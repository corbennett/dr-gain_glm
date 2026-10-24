"""Versioned datacube discovery and resumable spike-count extraction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from . import AREA_GROUPS, INCLUDED_AREAS
from .statistics import split_trials, stable_seed

STIMULI = ("is_vis_target", "is_vis_nontarget", "is_aud_target", "is_aud_nontarget")
WINDOWS = {"early": (0.0, 0.1), "late": (0.1, 0.5)}
BASELINE = (-0.2, 0.0)
UNIT_COLUMNS = (
    "unit_id",
    "session_id",
    "subject_id",
    "structure",
    "location",
    "default_qc",
    "firing_rate",
    "ccf_ap",
    "ccf_dv",
    "ccf_ml",
    "electrode_group_name",
    "peak_electrode",
    "peak_to_valley",
)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def atomic_parquet(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.write_parquet(temporary)
    temporary.replace(path)


def fetch_metadata(directory, version):
    """Use dr-datacube with the public cache; retain all discovery tables."""
    import dr_datacube as dc

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / "source.json"
    if stamp.exists() and json.loads(stamp.read_text())["version"] != version:
        raise ValueError("Metadata version mismatch; use a new metadata directory")
    config = getattr(dc, "datacube_config", getattr(dc, "config", None))
    config.version, config.use_cache = version, True
    if hasattr(config, "anon"):
        config.anon = True
    for remote, local in (
        ("session", "session_metadata"),
        ("units", "unit_metadata"),
        ("trials", "trials"),
        ("performance", "performance"),
    ):
        path = directory / f"{local}.parquet"
        if not path.exists():
            print(f"Downloading {remote} metadata", flush=True)
            lf = dc.get_lf(remote)
            if remote == "units":
                lf = lf.select(UNIT_COLUMNS)
            atomic_parquet(path, lf.collect())
    for kind, behavior in (("brainwide", True), ("naive", False)):
        path = directory / f"{kind}_cache_sessions.parquet"
        if not path.exists():
            function = getattr(
                dc, "get_sessions", getattr(dc, "get_session_table", None)
            )
            frame = function(
                kind, with_behavior_filter=behavior, only_in_data_asset=False
            )
            atomic_parquet(path, frame)
    path = directory / "nwb_sources.parquet"
    if not path.exists():
        atomic_parquet(path, pl.DataFrame({"nwb_path": dc.list_nwb_sources()}))
    if not stamp.exists():
        atomic_json(
            stamp,
            {
                "version": version,
                "retrieved_utc": datetime.now(UTC).isoformat(),
                "source": f"s3://aind-scratch-data/dynamic-routing/cache/nwb_components/{version}/consolidated",
                "cohort_source": "dr_datacube.get_sessions(..., only_in_data_asset=False)",
            },
        )


def select_areas(units, minimum_area=1000, included_areas=INCLUDED_AREAS):
    """Pool specified subdivisions before counting; preserve their source labels."""
    mapping = {
        child: parent for parent, children in AREA_GROUPS.items() for child in children
    }
    units = units.with_columns(pl.col("structure").alias("structure_original"))
    units = units.with_columns(pl.col("structure").replace(mapping))
    counts = (
        units.filter(pl.col("structure").is_not_null())
        .group_by("structure")
        .agg(
            pl.len().alias("n_units"),
            (pl.col("cohort") == "expert").sum().alias("n_expert"),
            (pl.col("cohort") == "naive").sum().alias("n_naive"),
            pl.col("session_id").n_unique().alias("n_sessions"),
            pl.col("subject_id").n_unique().alias("n_mice"),
            pl.col("structure_original")
            .unique()
            .sort()
            .str.join(",")
            .alias("source_structures"),
        )
        .with_columns(
            (
                (pl.col("n_units") >= minimum_area)
                | pl.col("structure").is_in(included_areas)
            ).alias("eligible"),
            pl.when(pl.col("n_units") >= minimum_area)
            .then(pl.lit("minimum_neuron_count"))
            .when(pl.col("structure").is_in(included_areas))
            .then(pl.lit("requested_inclusion"))
            .otherwise(pl.lit("below_threshold"))
            .alias("selection_reason"),
        )
        .sort("n_units", descending=True)
    )
    areas = counts.filter("eligible")["structure"].to_list()
    selected = units.filter(pl.col("structure").is_in(areas)).sort(
        "session_id", "unit_id"
    )
    return units, selected, counts


def build_manifest(
    metadata_dir,
    output_dir,
    minimum_area=1000,
    anatomy=None,
    included_areas=INCLUDED_AREAS,
):
    """Select areas before examining neural preferences and audit every session.

    The v0.0.289 'context_naive' keyword is known to mislabel expert sessions;
    the legacy 'context naive' keyword is used, as in dr-datacube. Also inventory
    naive sessions without the ccf keyword, rather than silently losing them.
    They enter an area analysis only if actual unit annotations are available.
    """
    metadata_dir, output_dir = Path(metadata_dir), Path(output_dir)
    sessions = pl.read_parquet(metadata_dir / "session_metadata.parquet")
    experts = pl.read_parquet(metadata_dir / "brainwide_cache_sessions.parquet")
    official_naive = pl.read_parquet(metadata_dir / "naive_cache_sessions.parquet")
    naive = sessions.filter(
        *[
            pl.col("keywords").list.contains(k)
            for k in ("context naive", "task", "ephys")
        ],
        *[~pl.col("keywords").list.contains(k) for k in ("issues", "templeton")],
    )
    rows = []
    for cohort, table in (("expert", experts), ("naive", naive)):
        for row in table.iter_rows(named=True):
            rows.append(
                {
                    "session_id": row["session_id"],
                    "subject_id": str(row["subject_id"]),
                    "cohort": cohort,
                    "official_datacube_cohort": cohort == "expert"
                    or row["session_id"] in official_naive["session_id"],
                    "has_ccf_keyword": "ccf" in row["keywords"],
                }
            )
    manifest = pl.from_dicts(rows).sort("cohort", "session_id")
    if manifest["session_id"].n_unique() != manifest.height:
        raise ValueError("Expert and naive cohorts overlap")
    units = pl.read_parquet(metadata_dir / "unit_metadata.parquet").select(UNIT_COLUMNS)
    units = units.with_columns(pl.col("subject_id").cast(pl.String))
    if anatomy is not None:
        extra = (
            pl.read_parquet(anatomy)
            if str(anatomy).endswith(".parquet")
            else pl.read_csv(anatomy)
        )
        if extra["unit_id"].n_unique() != extra.height:
            raise ValueError("External anatomy must contain one row per unit_id")
        columns = [
            c
            for c in ("structure", "location", "ccf_ap", "ccf_dv", "ccf_ml")
            if c in extra.columns
        ]
        units = units.join(
            extra.select("unit_id", *columns), on="unit_id", how="left", suffix="_new"
        )
        units = units.with_columns(
            *[pl.coalesce(f"{c}_new", c).alias(c) for c in columns]
        ).drop(*[f"{c}_new" for c in columns])
    units = units.filter(pl.col("default_qc").fill_null(False)).join(
        manifest.select("session_id", "cohort"),
        on="session_id",
        how="inner",
    )
    units, selected, counts = select_areas(units, minimum_area, included_areas)
    counts.write_csv(output_dir / "area_counts.csv")
    if selected["unit_id"].n_unique() != selected.height:
        raise ValueError("Duplicate unit IDs")
    paths = pl.read_parquet(metadata_dir / "nwb_sources.parquet").with_columns(
        pl.col("nwb_path")
        .str.split("/")
        .list.last()
        .str.split(".")
        .list.first()
        .alias("session_id"),
    )
    manifest = manifest.join(paths, on="session_id", how="left")
    trials = pl.read_parquet(metadata_dir / "trials.parquet")
    for frame, name in (
        (units, "n_qc_units"),
        (units.filter(pl.col("structure").is_not_null()), "n_anatomical_units"),
        (selected, "n_selected_units"),
        (trials, "n_trials"),
    ):
        manifest = manifest.join(
            frame.group_by("session_id").len(name), on="session_id", how="left"
        )
    manifest = manifest.with_columns(
        pl.col(
            "n_qc_units", "n_anatomical_units", "n_selected_units", "n_trials"
        ).fill_null(0)
    )
    reasons = []
    for row in manifest.iter_rows(named=True):
        if not row["n_qc_units"]:
            reason = "no_qc_units"
        elif not row["n_anatomical_units"]:
            reason = "missing_area_annotations"
        elif not row["n_selected_units"]:
            reason = "no_units_in_eligible_areas"
        elif not row["n_trials"]:
            reason = "missing_trials"
        elif row["nwb_path"] is None:
            reason = "missing_nwb"
        else:
            reason = "included"
        reasons.append(reason)
    manifest = manifest.with_columns(pl.Series("selection_status", reasons))
    manifest.write_csv(output_dir / "session_manifest.csv")
    atomic_parquet(output_dir / "selected_units.parquet", selected)
    return manifest, selected, trials


def prepare_trials(frame, seed):
    """Exclude instruction, catch, opto, and repeat trials; retain all outcomes."""
    frame = frame.sort("stim_start_time")
    frame = frame.with_columns(
        *[
            pl.col(column)
            .fill_null(False)
            .cast(pl.Float64)
            .shift(1)
            .fill_null(0)
            .alias(f"previous_{column}")
            for column in ("is_rewarded", "is_response", "is_vis_stim", "is_target")
        ]
    )
    valid = (
        ~pl.col("is_instruction").fill_null(True)
        & ~pl.col("is_opto").fill_null(False)
        & ~pl.col("is_repeat").fill_null(False)
        & pl.col("stim_start_time").is_finite()
        & pl.col("is_vis_rewarded").is_not_null()
    )
    frame = frame.filter(valid)
    flags = frame.select(STIMULI).fill_null(False).to_numpy()
    frame = frame.filter(pl.Series(flags.sum(axis=1) == 1))
    flags = frame.select(STIMULI).to_numpy()
    stimulus = flags.argmax(axis=1)
    context = frame["is_vis_rewarded"].to_numpy().astype(int)
    block = frame["block_index"].to_numpy()
    half = split_trials(stimulus, block, seed)
    times = frame["stim_start_time"].to_numpy()
    if len(times) < 40:
        raise ValueError("Fewer than 40 eligible trials")
    time_z = (times - times.mean()) / max(times.std(), 1)
    nuisance = np.column_stack(
        [
            np.ones(len(times)),
            time_z,
            np.eye(4)[stimulus, 1:],
            frame.select(
                [
                    f"previous_{c}"
                    for c in ("is_rewarded", "is_response", "is_vis_stim", "is_target")
                ]
            ).to_numpy(),
        ]
    )
    return frame, {
        "stimulus": stimulus,
        "context": context,
        "block": block,
        "half": half,
        "times": times,
        "nuisance": nuisance,
    }


def count_session(row, units, trials, cache_dir, seed):
    """Load each session once, save trial counts, and release the spike arrays."""
    import lazynwb

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{row['session_id']}.npz"
    unit_ids = np.asarray(units["unit_id"].to_list())
    frame, arrays = prepare_trials(
        trials, stable_seed(seed, row["session_id"], "halves")
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if (
                np.array_equal(cached["unit_ids"], unit_ids)
                and np.array_equal(cached["times"], arrays["times"])
                and np.array_equal(cached["half"], arrays["half"])
            ):
                return {k: cached[k] for k in cached.files}
        raise ValueError(f"Count-cache mismatch in {path}; use a fresh run directory")
    lazynwb.config.anon = True
    spikes = (
        lazynwb.scan_nwb(row["nwb_path"], "/units")
        .filter(pl.col("unit_id").is_in(unit_ids.tolist()))
        .select("unit_id", "spike_times")
        .collect()
        .sort("unit_id")
    )
    if spikes["unit_id"].to_list() != unit_ids.tolist():
        raise ValueError("NWB spike rows do not match selected unit IDs")
    for name in ("baseline", *WINDOWS):
        arrays[name] = np.empty((len(unit_ids), len(frame)), dtype=np.float32)
    for i, (_, values) in enumerate(spikes.iter_rows()):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite spikes for {unit_ids[i]}")
        if np.any(np.diff(values) < 0):
            values = np.sort(values)
        for name, (start, end) in {"baseline": BASELINE, **WINDOWS}.items():
            bounds = arrays["times"][:, None] + np.array([start, end])
            arrays[name][i] = np.diff(
                np.searchsorted(values, bounds, side="left"), axis=1
            )[:, 0] / (end - start)
    arrays["unit_ids"] = unit_ids
    arrays["response_time"] = frame["response_time"].to_numpy()
    arrays["trial_index"] = frame["trial_index"].to_numpy()
    arrays["is_response"] = frame["is_response"].fill_null(False).to_numpy()
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return arrays
