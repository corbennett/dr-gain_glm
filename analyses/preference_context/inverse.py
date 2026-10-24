"""P(preference | baseline context), with within-session enrichment controls.

Run: caffeinate -i .venv/bin/python analyses/preference_context/inverse.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from gain_glm.context_preferences.statistics import stable_seed
from gain_glm.preference_context.statistics import match_groups
from gain_glm.responsive_preferences.statistics import adjust_by_group, mouse_inference

GROUP = ["structure", "cohort", "window", "selection_half", "fit_subset"]
OUTCOMES = ("modality", "target", "modality_dominant", "target_dominant")


def add_indicators(units):
    return units.with_columns(
        (pl.col("modality_q") < 0.05).cast(pl.Float64).alias("modality"),
        (pl.col("target_q") < 0.05).cast(pl.Float64).alias("target"),
        (pl.col("dominant_class") == "modality_dominant")
        .cast(pl.Float64)
        .alias("modality_dominant"),
        (pl.col("dominant_class") == "target_dominant")
        .cast(pl.Float64)
        .alias("target_dominant"),
        (pl.col("glm_baseline_p") < 0.05).alias("baseline_context"),
    )


def conditional_counts(units):
    """All denominators are responsive cells, including both/neither preferences."""
    return units.group_by(*GROUP, "baseline_context").agg(
        pl.len().alias("n_cells"),
        *[pl.col(c).sum().alias(f"n_{c}") for c in OUTCOMES],
        (pl.col("modality") * pl.col("target")).sum().alias("n_both"),
        ((pl.col("modality") == 0) & (pl.col("target") == 0)).sum().alias("n_neither"),
    )


def analyze(units, seed=20261023, minimum=3):
    base = add_indicators(units)
    expanded = pl.concat(
        [
            base.with_columns(pl.lit("all").alias("fit_subset")),
            base.filter(pl.col("glm_converged")).with_columns(
                pl.lit("converged").alias("fit_subset")
            ),
        ]
    )
    pooled = pl.concat(
        [
            conditional_counts(expanded),
            conditional_counts(
                expanded.with_columns(pl.lit("All areas").alias("structure"))
            ),
        ]
    )
    rows, support = [], []
    for key, f in expanded.group_by(
        "session_id", "subject_id", *GROUP, maintain_order=True
    ):
        meta = dict(zip(["session_id", "subject_id", *GROUP], key))
        detected = f["baseline_context"].to_numpy()
        yes, no = np.flatnonzero(detected), np.flatnonzero(~detected)
        # Match on independently measured selection-half rate/response strength.
        features = np.log1p(
            f.select("selection_baseline_hz", "selection_response_rms_hz").to_numpy()
        )
        pairs, _, _ = match_groups(features, yes, no, caliper=1.5)
        support.append(
            {
                **meta,
                "n_baseline": len(yes),
                "n_no_baseline": len(no),
                "n_matched_pairs": len(pairs),
            }
        )

        def append(
            comparison, adjustment, metric, left, right, n_left, n_right, meta=meta
        ):
            rows.append(
                {
                    **meta,
                    "comparison": comparison,
                    "adjustment": adjustment,
                    "metric": metric,
                    "estimate": left - right,
                    "left": left,
                    "right": right,
                    "n_left": n_left,
                    "n_right": n_right,
                }
            )

        for label, idx in (("within_baseline", yes), ("within_no_baseline", no)):
            if len(idx) >= minimum:
                append(
                    label,
                    "unmatched",
                    "modality_minus_target",
                    float(f["modality"].to_numpy()[idx].mean()),
                    float(f["target"].to_numpy()[idx].mean()),
                    len(idx),
                    len(idx),
                )
        for adjustment, i, j in (
            ("unmatched", yes, no),
            ("matched", pairs[:, 0], pairs[:, 1]),
        ):
            if min(len(i), len(j)) < minimum:
                continue
            for outcome in OUTCOMES:
                values = f[outcome].to_numpy()
                append(
                    "baseline_enrichment",
                    adjustment,
                    outcome,
                    float(values[i].mean()),
                    float(values[j].mean()),
                    len(i),
                    len(j),
                )
    sessions = pl.from_dicts(rows)
    keys = [*GROUP, "comparison", "adjustment", "metric"]
    summaries = []
    for key, f in sessions.group_by(keys, maintain_order=True):
        stats = mouse_inference(
            f["estimate"].to_numpy(),
            f["subject_id"].to_numpy(),
            stable_seed(seed, *key),
        )
        outcomes = f.select("left", "right").to_numpy()
        if np.all(outcomes == 0) or np.all(outcomes == 1):
            stats.update(
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=np.nan,
                test="no_outcome_variation",
                population_inference=False,
            )
        per_mouse = f.group_by("subject_id").agg(
            pl.col("left").mean(), pl.col("right").mean()
        )
        summaries.append(
            {
                **dict(zip(keys, key)),
                **stats,
                "left": per_mouse["left"].mean(),
                "right": per_mouse["right"].mean(),
                "n_left": f["n_left"].sum(),
                "n_right": f["n_right"].sum(),
            }
        )
    summary = pl.from_dicts(summaries).with_columns(
        pl.when(pl.col("metric").str.ends_with("dominant"))
        .then(pl.lit("dominance"))
        .otherwise(pl.lit("detected_preference"))
        .alias("metric_family")
    )
    summary = adjust_by_group(
        summary,
        [
            "window",
            "selection_half",
            "fit_subset",
            "comparison",
            "adjustment",
            "metric_family",
        ],
        "p_value",
        "q_value",
    )
    return pooled, sessions, summary, pl.from_dicts(support)


def report(out, pooled, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = summary.filter(pl.col("selection_half") == 0, pl.col("fit_subset") == "all")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    areas = sorted(set(summary["structure"]))
    for window in ("early", "late"):
        f = main.filter(pl.col("window") == window, pl.col("adjustment") == "unmatched")
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 12), sharey=True, constrained_layout=True
        )
        for ax, comparison, title in zip(
            axes[0],
            ("within_baseline", "within_no_baseline"),
            (
                "Given detected baseline context encoding",
                "Given no detected baseline context encoding",
            ),
        ):
            for i, area in enumerate(areas):
                row = f.filter(
                    pl.col("structure") == area, pl.col("comparison") == comparison
                )
                if row.height:
                    r = row.row(0, named=True)
                    ax.plot(100 * np.array([r["left"], r["right"]]), [i, i], color=".7")
                    ax.scatter(
                        100 * r["left"],
                        i,
                        color="#3478b8",
                        label="Modality" if i == 0 else None,
                    )
                    ax.scatter(
                        100 * r["right"],
                        i,
                        color="#d77d28",
                        label="Target/nontarget" if i == 0 else None,
                    )
            ax.set_title(title)
            ax.set_xlim(0, 105)
            ax.set_xlabel("Preference detected (%); both can be present")
            ax.legend(loc="lower right", fontsize=8)
        for ax, metric in zip(axes[1], ("modality", "target")):
            for i, area in enumerate(areas):
                row = f.filter(
                    pl.col("structure") == area,
                    pl.col("comparison") == "baseline_enrichment",
                    pl.col("metric") == metric,
                )
                if row.height:
                    r = row.row(0, named=True)
                    color = "#20855e" if r["q_value"] < 0.05 else ".4"
                    ax.plot(
                        100 * np.array([r["ci_low"], r["ci_high"]]), [i, i], color=color
                    )
                    ax.scatter(100 * r["estimate"], i, color=color)
                    ax.annotate(
                        f" {r['n_mice']} mice",
                        (100 * r["estimate"], i),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=6,
                    )
                else:
                    ax.text(
                        0.03,
                        i,
                        "insufficient groups",
                        transform=ax.get_yaxis_transform(),
                        fontsize=7,
                    )
            ax.axvline(0, color="k", lw=0.7)
            ax.set_title(
                f"{metric.capitalize()} preference: enrichment with baseline encoding"
            )
            ax.set_xlabel(
                "Baseline context detected − not detected (percentage points)"
            )
        for ax in axes.flat:
            ax.set_yticks(range(len(areas)), areas)
            ax.set_ylim(len(areas) - 0.4, -0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"Expert responsive cells · {window} · equal session/mouse weights\n"
            "Bottom: common-session comparisons, 95% mouse intervals; green: q < .05"
        )
        fig.savefig(figures / f"{window}.png", dpi=150)
        plt.close(fig)
    rows = pooled.filter(
        pl.col("structure") == "All areas",
        pl.col("window") == "early",
        pl.col("selection_half") == 0,
        pl.col("fit_subset") == "all",
    ).sort("baseline_context", descending=True)
    lines = [
        "# Inverse conditional analysis",
        "",
        "Primary early expert responsive cells; pooled descriptive fractions:",
        "",
        "| Baseline context | Cells | Modality preference | Target/nontarget preference | Both |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows.iter_rows(named=True):
        n = r["n_cells"]
        lines.append(
            f"| {'Detected' if r['baseline_context'] else 'Not detected'} | {n} | "
            f"{100 * r['n_modality'] / n:.1f}% ({int(r['n_modality'])}) | "
            f"{100 * r['n_target'] / n:.1f}% ({int(r['n_target'])}) | {int(r['n_both'])} |"
        )
    lines += [
        "",
        "The two preference columns overlap. Baseline encoding uses the notebook's nominal GLM",
        "permutation p < .05; modality and target preference use their original joint q < .05.",
        "All denominators retain cells with both or neither preference. These probabilities are",
        "conditional on significant stimulus responsiveness; they do not describe all baseline cells.",
        "",
        "## Within-session early enrichment",
        "",
        "Equal sessions within mouse, then equal mice; at least 3 cells in each baseline group.",
        "",
        "| Area | Outcome | Difference pp [95% CI] | q | Mice |",
        "|---|---|---|---:|---:|",
    ]
    for r in (
        main.filter(
            pl.col("window") == "early",
            pl.col("comparison") == "baseline_enrichment",
            pl.col("adjustment") == "unmatched",
            pl.col("metric").is_in(["modality", "target"]),
        )
        .sort("structure", "metric")
        .iter_rows(named=True)
    ):
        lines.append(
            f"| {r['structure']} | {r['metric']} | {100 * r['estimate']:+.1f} "
            f"[{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] | {r['q_value']:.3g} | {r['n_mice']} |"
        )
    lines += [
        "",
        "See area_summary.csv for within-baseline M-versus-T comparisons, matching, convergence",
        "restriction, reversed splits, dominance labels, and late responses. No naive GLMs are available.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Preference given baseline context</title>"
        "<style>body{font:16px system-ui;max-width:1300px;margin:auto}img{width:100%}</style>"
        "<h1>Stimulus preference given baseline context encoding</h1><p><a href='REPORT.md'>Numerical report</a> · "
        "<a href='area_summary.csv'>All comparisons</a></p><p>Among stimulus-responsive cells; preferences may overlap. "
        "Both/neither remain in the denominators.</p><img src='figures/early.png'><img src='figures/late.png'>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/preference_context/glm")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/preference_context/inverse")
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == args.input_dir.resolve():
        raise ValueError("Use a separate output directory")
    if json.loads((args.input_dir / "status.json").read_text())["state"] != "complete":
        raise ValueError("Incomplete input analysis")
    out.mkdir(exist_ok=True, parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    source = args.input_dir / "linked_neurons.parquet"
    helpers = [
        Path(__file__),
        Path("src/gain_glm/preference_context/statistics.py"),
        Path("src/gain_glm/responsive_preferences/statistics.py"),
        Path("src/gain_glm/context_preferences/statistics.py"),
    ]
    provenance = {
        "inputs": {
            str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()
        },
        "code": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers},
        "seed": 20261023,
        "minimum_group_cells": 3,
        "matching_caliper": 1.5,
    }
    path = out / "provenance.json"
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError("Inputs/code changed; use a fresh output directory")
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "source_snapshot").mkdir(exist_ok=True)
    for p in helpers:
        (out / "source_snapshot" / str(p).replace("/", "_")).write_bytes(p.read_bytes())

    def status(state, **extra):
        (out / "status.json").write_text(
            json.dumps({"state": state, "pid": os.getpid(), **extra}, indent=2) + "\n"
        )

    try:
        status("analyzing")
        pooled, sessions, summary, support = analyze(pl.read_parquet(source))
        for name, frame in (
            ("conditional_counts", pooled),
            ("session_comparisons", sessions),
            ("area_summary", summary),
            ("comparison_support", support),
        ):
            frame.write_csv(out / f"{name}.csv")
        report(out, pooled, summary)
        status("complete", failures=[])
        print(f"Finished inverse conditional analysis: {out}")
    except Exception as error:
        status("failed", failures=[str(error)])
        raise


if __name__ == "__main__":
    main()
