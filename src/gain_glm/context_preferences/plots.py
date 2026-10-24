"""Standalone figures and a local HTML index; no web service required."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

COLORS = {"expert": "#2976ad", "naive": "#d97722"}
LABELS = {
    "baseline_modality": "Baseline vs modality",
    "baseline_target": "Baseline vs target (pooled)",
    "baseline_target_visual": "Baseline vs visual target",
    "baseline_target_auditory": "Baseline vs auditory target",
    "baseline_raw_modality": "Raw baseline vs modality",
}


def _interval(ax, row, y, color):
    x = row["estimate"]
    if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, lw=2)
    ax.plot(x, y, "o", color=color, mfc=color if row["q_value"] < 0.05 else "white")


def area_figure(area, window, units, summary, sessions):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    subset = units.filter(pl.col("structure") == area, pl.col("window") == window)
    stats = summary.filter(pl.col("structure") == area, pl.col("window") == window)
    ax = axes[0, 0]
    for cohort in ("expert", "naive"):
        x = subset.filter(pl.col("cohort") == cohort)["modality_hz"].to_numpy()
        if not len(x):
            continue
        limit = max(np.quantile(np.abs(subset["modality_hz"].to_numpy()), 0.99), 1)
        ax.hist(
            x,
            bins=np.linspace(-limit, limit, 45),
            density=True,
            histtype="step",
            lw=2,
            color=COLORS[cohort],
            label=f"{cohort} (n={len(x):,})",
        )
    ax.axvline(0, color=".5", lw=1)
    ax.set(
        xlabel="Visual − auditory evoked response (Hz)",
        ylabel="Density",
        title="Q1: continuous modality preference\nDisplay clipped at pooled 99th absolute percentile",
    )
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for cohort in ("expert", "naive"):
        frame = subset.filter(pl.col("cohort") == cohort)
        # Explicitly disjoint halves even in the illustrative scatter.
        ax.scatter(
            frame["baseline_h0"],
            frame["modality_h1"],
            s=5,
            alpha=0.18,
            color=COLORS[cohort],
            rasterized=True,
            label=cohort,
        )
    ax.axhline(0, color=".6", lw=0.7)
    ax.axvline(0, color=".6", lw=0.7)
    ax.set(
        xlabel="Baseline visual − auditory context, half 0 (Hz)",
        ylabel="Modality preference, half 1 (Hz)",
        title="Q2: independently estimated preferences\nPooled scatter is descriptive",
    )

    ax = axes[0, 2]
    names = list(LABELS)
    for i, name in enumerate(names):
        for j, cohort in enumerate(("expert", "naive")):
            frame = stats.filter(pl.col("metric") == name, pl.col("cohort") == cohort)
            if frame.height:
                _interval(
                    ax, frame.row(0, named=True), i + (j - 0.5) * 0.22, COLORS[cohort]
                )
    ax.axvline(0, color=".6", lw=1)
    ax.set(
        yticks=range(len(names)),
        yticklabels=[LABELS[n] for n in names],
        xlabel="Cross-half Spearman correlation",
        xlim=(-1, 1),
        title="Q2–3: session/mouse-weighted relationships\nFilled: q < .05; bars: mouse bootstrap 95% CI",
    )

    ax = axes[1, 0]
    for i, name in enumerate(("modality_power", "modality_generalization")):
        for j, cohort in enumerate(("expert", "naive")):
            frame = stats.filter(pl.col("metric") == name, pl.col("cohort") == cohort)
            if frame.height:
                _interval(
                    ax, frame.row(0, named=True), i + (j - 0.5) * 0.22, COLORS[cohort]
                )
    ax.axvline(0, color=".6", lw=1)
    ax.set(
        yticks=[0, 1],
        yticklabels=["Modality power", "Target → nontarget\ngeneralization"],
        xlabel="Cross-validated normalized product",
        title="Q1: signal beyond shuffled stimulus labels",
    )

    ax = axes[1, 1]
    for j, cohort in enumerate(("expert", "naive")):
        frame = stats.filter(
            pl.col("metric") == "modality_selective_fraction",
            pl.col("cohort") == cohort,
        )
        if frame.height:
            row = frame.row(0, named=True)
            _interval(ax, row, j, COLORS[cohort])
            ax.plot(row["null_mean"], j, "x", color="black")
    ax.set(
        yticks=[0, 1],
        yticklabels=["Expert", "Naive"],
        xlim=(0, 1),
        xlabel="Fraction with nominal modality p ≤ .05",
        title="Q1 & Q4: selective fractions\nBlack ×: simultaneous-cell shuffle expectation",
    )

    ax = axes[1, 2]
    spatial = sessions.filter(
        pl.col("structure") == area,
        pl.col("window") == window,
        pl.col("metric").is_in(["spatial_modality", "spatial_modality_within_probe"]),
    )
    rng = np.random.default_rng(0)
    for i, metric in enumerate(("spatial_modality", "spatial_modality_within_probe")):
        for j, cohort in enumerate(("expert", "naive")):
            frame = spatial.filter(
                pl.col("metric") == metric,
                pl.col("cohort") == cohort,
                pl.col("estimate").is_finite(),
            )
            y = i + (j - 0.5) * 0.24 + rng.normal(0, 0.025, frame.height)
            ax.scatter(
                frame["estimate"].to_numpy() - frame["null_mean"].to_numpy(),
                y,
                c=COLORS[cohort],
                s=18,
                alpha=0.6,
            )
    ax.axvline(0, color=".6", lw=1)
    ax.set(
        yticks=[0, 1],
        yticklabels=["Within session", "Within probe"],
        xlabel="Spatial covariance − permutation expectation",
        title="Q5: each point is one session\nNegative / near-zero / positive effects all retained",
    )
    n_mice = subset.group_by("cohort").agg(
        pl.col("subject_id").n_unique().alias("mice"),
        pl.col("session_id").n_unique().alias("sessions"),
    )
    text = "; ".join(
        f"{r['cohort']}: {r['mice']} mice, {r['sessions']} sessions"
        for r in n_mice.iter_rows(named=True)
    )
    fig.suptitle(
        f"{area} — {window} response | {text}\nBlue: expert; orange: naive. Single-mouse results have no population confidence interval.",
        fontsize=13,
    )
    return fig


def session_maps(path, area, units, stats):
    early = units.filter(pl.col("structure") == area, pl.col("window") == "early")
    late = units.filter(pl.col("structure") == area, pl.col("window") == "late")
    pages = 0
    with PdfPages(path) as pdf:
        for (sid,), frame in early.group_by("session_id", maintain_order=True):
            frame = frame.sort("unit_id").filter(
                pl.all_horizontal(
                    *[pl.col(c).is_finite() for c in ("ccf_ap", "ccf_dv", "ccf_ml")]
                )
            )
            if frame.height < 20:
                continue
            later = late.filter(pl.col("session_id") == sid).select(
                "unit_id", pl.col("modality_hz").alias("late_modality_hz")
            )
            frame = frame.join(later, on="unit_id", how="left")
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
            metrics = (
                (
                    "modality_hz",
                    "early",
                    "spatial_modality",
                    "Early modality preference",
                ),
                (
                    "late_modality_hz",
                    "late",
                    "spatial_modality",
                    "Late modality preference",
                ),
                (
                    "baseline_hz",
                    "baseline",
                    "spatial_baseline",
                    "Baseline context preference",
                ),
            )
            for col, (field, window, metric, label) in enumerate(metrics):
                values = frame[field].to_numpy()
                if not np.isfinite(values).any():
                    for ax in axes[:, col]:
                        ax.text(
                            0.5,
                            0.5,
                            "Context contrast unavailable",
                            ha="center",
                            transform=ax.transAxes,
                        )
                        ax.set_axis_off()
                    continue
                limit = max(float(np.nanquantile(np.abs(values), 0.95)), 0.01)
                for r, (x, y) in enumerate(
                    (("ccf_ap", "ccf_dv"), ("ccf_ml", "ccf_dv"))
                ):
                    ax = axes[r, col]
                    scatter = ax.scatter(
                        frame[x],
                        frame[y],
                        c=values,
                        vmin=-limit,
                        vmax=limit,
                        cmap="RdBu_r",
                        s=18,
                        alpha=0.85,
                        rasterized=True,
                    )
                    ax.set(
                        xlabel=f"{x.removeprefix('ccf_').upper()} (µm)",
                        ylabel="DV (µm)",
                    )
                    ax.invert_yaxis()
                    ax.set_aspect("equal", adjustable="datalim")
                    fig.colorbar(
                        scatter, ax=ax, label="Visual − auditory (Hz)", shrink=0.7
                    )
                result = stats.filter(
                    pl.col("session_id") == sid,
                    pl.col("structure") == area,
                    pl.col("window") == window,
                    pl.col("metric") == metric,
                )
                local = stats.filter(
                    pl.col("session_id") == sid,
                    pl.col("structure") == area,
                    pl.col("window") == window,
                    pl.col("metric") == metric + "_within_probe",
                )
                detail = "insufficient spatial support"
                if result.height:
                    a = result.row(0, named=True)
                    detail = f"I={a['estimate']:.3f}, q={a['q_value']:.3g}"
                if local.height:
                    a = local.row(0, named=True)
                    detail += f"; probe q={a['q_value']:.3g}"
                axes[0, col].set_title(f"{label}\n{detail}", fontsize=10)
            sites = frame.select("ccf_ap", "ccf_dv", "ccf_ml").unique().height
            fig.suptitle(
                f"{area} | {sid} | {frame['cohort'][0]} | {frame.height} neurons, {sites} recording sites\n"
                "Red: visual preference; blue: auditory preference. Color clipped at 95th absolute percentile.\n"
                "Coordinates estimate recording sites, not individual soma positions; tests use 3D distances.",
                fontsize=12,
            )
            pdf.savefig(fig, dpi=130)
            plt.close(fig)
            pages += 1
    return pages


def report(out, area_summary, session_stats, units, manifest, failures):
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Context and sensory preferences</title>",
        "<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px}img{width:100%}a{color:#176393}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style>",
        "<h1>Context and sensory preferences</h1>",
        (
            "<p>Area-wise analyses of QC-passing neurons. Effects are estimated from independent trial halves. "
            "Population estimates weight sessions equally within mice, then mice equally. "
            "A positive modality preference means higher baseline-subtracted firing for vision; suppression can also be selective.</p>"
        ),
        (
            "<p>Naive comparisons with fewer than three mice are descriptive. "
            "Significance of functional tuning does not establish anatomical pathways. "
            "An absent spatial effect is limited by the sampling geometry and recording-site resolution.</p>"
        ),
        (
            "<p><a href='area_summary.csv'>Area statistics</a> · <a href='session_statistics.csv'>Session statistics</a> · "
            "<a href='session_manifest.csv'>Cohort and exclusion audit</a> · <a href='REPORT.md'>Run notes</a></p>"
        ),
    ]
    notes = [
        "# Context and sensory preference analysis",
        "",
        f"Completed sessions represented: {units['session_id'].n_unique()}",
        f"Neurons represented: {units['unit_id'].n_unique():,}",
        f"Runtime failures: {len(failures)}",
        "",
        "## Cohort coverage",
        "",
    ]
    for cohort in ("expert", "naive"):
        frame = units.filter(pl.col("cohort") == cohort)
        notes.append(
            f"- {cohort}: {frame['unit_id'].n_unique():,} neurons, "
            f"{frame['session_id'].n_unique()} sessions, {frame['subject_id'].n_unique()} mice."
        )
    notes += [
        "",
        "## Interpretation",
        "",
        (
            "The primary window is 0–100 ms; the secondary window is 100–500 ms. "
            "Late responses may include action and reward effects. Baseline is −200–0 ms. "
            "Baseline context coefficients control linear session drift, upcoming stimulus identity, "
            "and previous reward/response/modality/target status, but are not a full behavior-adjusted GLM."
        ),
        "",
        (
            "Area p-values compare to the specified shuffle null conditional on recorded sessions. "
            "Mouse bootstrap intervals assess animal sampling uncertainty. Global area-summary BH q-values "
            "include all tested areas, cohorts, windows, and metrics; session q-values form a separate family. "
            "A small conditional p-value from one naive mouse does not establish a population trend."
        ),
        "",
        (
            "Stimulus generalization uses just two identities per modality. Spatial tests use a symmetric "
            "10-nearest-neighbor graph with a 500 µm maximum edge length. Within-probe tests center and shuffle "
            "within probes, providing a sensitivity check against probe-level sampling differences."
        ),
        "",
        "Missing anatomy is audited in session_manifest.csv; unassigned neurons are not pooled as a brain area.",
        "",
        "## Figures",
        "",
    ]
    for area in sorted(units["structure"].unique()):
        print(f"Rendering {area}", flush=True)
        parts.append(f"<h2>{escape(area)}</h2>")
        for window in ("early", "late"):
            fig = area_figure(area, window, units, area_summary, session_stats)
            name = f"{area}_{window}.png"
            fig.savefig(figures / name, dpi=130)
            plt.close(fig)
            parts.append(
                f"<img src='figures/{escape(name)}' alt='{escape(area)} {window} analysis'>"
            )
        map_name = f"{area}_individual_sessions.pdf"
        pages = session_maps(figures / map_name, area, units, session_stats)
        if pages:
            parts.append(
                f"<p><a href='figures/{escape(map_name)}'>Individual-session spatial maps ({pages} pages)</a></p>"
            )
        else:
            parts.append(
                "<p>No sessions have enough spatially annotated neurons for a map.</p>"
            )
        notes.append(
            f"- {area}: [early](figures/{area}_early.png), [late](figures/{area}_late.png), "
            f"[individual sessions](figures/{map_name}) ({pages} pages)."
        )
    (out / "index.html").write_text("\n".join(parts))
    (out / "REPORT.md").write_text("\n".join(notes) + "\n")
