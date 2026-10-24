"""Preference distributions and direct paired comparisons, with explicit units."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

COLORS = {"modality": "#2876ac", "target": "#da7e23"}


def cell_weights(frame):
    """Distributions give equal weight to mice, then sessions within mouse."""
    mice, sessions = frame["subject_id"].to_numpy(), frame["session_id"].to_numpy()
    weights = np.zeros(frame.height)
    for mouse in np.unique(mice):
        sids = np.unique(sessions[mice == mouse])
        for sid in sids:
            idx = (mice == mouse) & (sessions == sid)
            weights[idx] = 1 / (len(np.unique(mice)) * len(sids) * idx.sum())
    return weights


def interval(ax, row, position, color, horizontal=False):
    value = row["estimate"]
    lo, hi = row["ci_low"], row["ci_high"]
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot(
            [lo, hi] if horizontal else [position, position],
            [position, position] if horizontal else [lo, hi],
            color=color,
            lw=2,
        )
    ax.plot(
        value if horizontal else position,
        position if horizontal else value,
        "o",
        color=color,
        mfc=color if row["q_value"] < 0.05 else "white",
        ms=6,
    )


def area_figure(frame, summary, mice, count):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    weights = cell_weights(frame)
    for ax, suffix, label in (
        (axes[0, 0], "hz", "Held-out difference (Hz)"),
        (axes[0, 1], "scaled", "Held-out difference / selection response RMS"),
    ):
        values = [frame[f"{f}_{suffix}"].to_numpy() for f in COLORS]
        quantile = 1.0 if frame.height < 100 else 0.995
        limit = max(float(np.quantile(np.abs(np.concatenate(values)), quantile)), 0.1)
        bins = np.linspace(-limit, limit, 51)
        for factor, value in zip(COLORS, values):
            ax.hist(
                value,
                bins=bins,
                weights=weights,
                density=True,
                histtype="step",
                color=COLORS[factor],
                lw=2,
                label=factor,
            )
        ax.axvline(0, color=".5", lw=1)
        ax.set(
            xlabel=label,
            ylabel="Mouse-weighted density",
            title="Signed preferences in responsive cells\n"
            + (
                "All cells shown"
                if quantile == 1
                else "Range: pooled 99.5th absolute percentile"
            ),
        )
        ax.legend(fontsize=9)
    ax = axes[0, 2]
    m, t = (
        frame["modality_scaled"].abs().to_numpy(),
        frame["target_scaled"].abs().to_numpy(),
    )
    limit = max(
        float(np.quantile(np.r_[m, t], 1.0 if frame.height < 100 else 0.995)), 0.1
    )
    ax.scatter(m, t, s=6, alpha=0.25, color="#526775", rasterized=True)
    ax.plot([0, limit], [0, limit], "--", color=".5")
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="|Modality difference| / response RMS",
        ylabel="|Target difference| / response RMS",
        title="Same-cell comparison (descriptive)\nBelow diagonal: larger modality difference",
    )
    ax.set_aspect("equal", adjustable="box")
    ax = axes[1, 0]
    for i, factor in enumerate(COLORS):
        row = summary.filter(pl.col("metric") == f"{factor}_power").row(0, named=True)
        interval(ax, row, i, COLORS[factor])
    ax.axhline(0, color=".5", lw=1)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Modality", "Target vs nontarget"],
        xlim=(-0.5, 1.5),
        ylabel="Cross-validated squared preference / response RMS²",
        title="Reproducible preference strength\nFilled: trial-shuffle q < .05; bars: mouse 95% CI",
    )
    ax = axes[1, 1]
    differences = mice.filter(pl.col("metric") == "power_difference")[
        "estimate"
    ].to_numpy()
    rng = np.random.default_rng(0)
    ax.scatter(
        differences,
        rng.uniform(-0.08, 0.08, len(differences)),
        color="#526775",
        alpha=0.7,
        s=24,
    )
    row = summary.filter(pl.col("metric") == "power_difference").row(0, named=True)
    interval(ax, row, 1, "#35256d", horizontal=True)
    ax.axvline(0, color=".5", lw=1)
    p_text = (
        f"paired mouse q={row['q_value']:.3g}"
        if np.isfinite(row["q_value"])
        else "single-mouse comparison is descriptive"
    )
    ax.set(
        yticks=[0, 1],
        yticklabels=["Individual mice", "Mean and 95% CI"],
        ylim=(-0.3, 1.3),
        xlabel="Modality strength − target strength (normalized)",
        title=f"Direct comparison: positive = modality stronger\n{p_text}",
    )
    ax = axes[1, 2]
    m, t = (frame[f"{f}_q"].to_numpy() < 0.05 for f in COLORS)
    categories = [m & ~t, ~m & t, m & t, ~m & ~t]
    fractions = [float(weights[v].sum()) for v in categories]
    ax.bar(
        range(4),
        fractions,
        color=[COLORS["modality"], COLORS["target"], "#77639e", "#aaa"],
    )
    ax.set(
        xticks=range(4),
        xticklabels=["Modality\nonly", "Target\nonly", "Both", "Neither\ndetected"],
        ylim=(0, 1),
        ylabel="Fraction of selected cells (mouse weighted)",
        title="Held-out selectivity at unit q < .05\nNeither detected does not establish equal responses",
    )
    first = frame.row(0, named=True)
    fig.suptitle(
        f"{first['structure']} | {first['cohort']} | {first['window']} response | "
        f"{frame.height:,}/{count['n_tested']:,} responsive cells; {frame['subject_id'].n_unique()} mice\n"
        "Blue = visual − auditory; orange = target − nontarget. Selection and evaluation use separate trials.",
        fontsize=14,
    )
    return fig


def overview(summary, cohort, window):
    frame = summary.filter(pl.col("cohort") == cohort, pl.col("window") == window)
    areas = sorted(frame["structure"].unique())
    fig, axes = plt.subplots(
        1, 3, figsize=(15, max(6, len(areas) * 0.43)), layout="constrained"
    )
    for i, area in enumerate(areas):
        for j, factor in enumerate(COLORS):
            f = frame.filter(
                pl.col("structure") == area,
                pl.col("selection_half") == 0,
                pl.col("metric") == f"{factor}_power",
            )
            if f.height:
                interval(
                    axes[0],
                    f.row(0, named=True),
                    i + (j - 0.5) * 0.2,
                    COLORS[factor],
                    horizontal=True,
                )
        for ax, split in zip(axes[1:], (0, 1)):
            f = frame.filter(
                pl.col("structure") == area,
                pl.col("selection_half") == split,
                pl.col("metric") == "power_difference",
            )
            if f.height:
                interval(ax, f.row(0, named=True), i, "#35256d", horizontal=True)
    for ax in axes:
        ax.axvline(0, color=".5", lw=1)
        ax.set(yticks=range(len(areas)), yticklabels=areas)
        ax.invert_yaxis()
    axes[0].set(
        title="Reproducible preference strength\nBlue: modality; orange: target",
        xlabel="Normalized cross-validated squared preference",
    )
    axes[1].set(
        title="Primary paired difference\nSelect half 0; evaluate half 1",
        xlabel="Modality − target strength",
    )
    axes[2].set(
        title="Reversed-split sensitivity\nSelect half 1; evaluate half 0",
        xlabel="Modality − target strength",
    )
    fig.suptitle(
        f"Stimulus-responsive cells | {cohort} | {window}\n"
        "Dots: equal-mouse means; bars: mouse bootstrap 95% CI; filled: q < .05. Positive differences favor modality."
    )
    return fig


def report(out, units, summary, mice, counts):
    out = Path(out)
    alpha = json.loads((out / "configuration.json").read_text())["response_alpha"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    html = [
        "<!doctype html><meta charset='utf-8'><title>Preferences of stimulus-responsive cells</title>",
        "<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}img{width:100%}a{color:#176393}</style>",
        "<h1>Preferences of stimulus-responsive cells</h1>",
        (
            "<p>Cells are selected by a response-versus-baseline test on one trial half. Both preference measures use the other half. "
            "Responsiveness includes excitation and suppression in any tested stimulus/context condition. "
            "Primary: 0–100 ms; secondary: 100–500 ms (can include action and reward effects).</p>"
        ),
        (
            "<p>Modality = (VT + VN − AT − AN)/2. Target = (VT + AT − VN − AN)/2. "
            "Both use equal stimulus/context weighting and a common response-strength scale. "
            "Raw distributions and absolute-value scatterplots are descriptive; the direct strength test uses independent evaluation-split products.</p>"
        ),
        (
            "<p><a href='REPORT.md'>Results summary</a> · <a href='area_summary.csv'>Area statistics</a> · "
            "<a href='selection_counts.csv'>Selection counts</a> · <a href='selection_overlap.csv'>Split-selection overlap</a> · "
            "<a href='../../../analyses/responsive_preferences/README.md'>Methods</a></p>"
        ),
    ]
    for cohort in sorted(units["cohort"].unique()):
        html.append(f"<h2>{escape(cohort)} overview</h2>")
        for window in ("early", "late"):
            fig = overview(summary, cohort, window)
            name = f"overview_{cohort}_{window}.png"
            fig.savefig(figures / name, dpi=130)
            plt.close(fig)
            html.append(f"<img src='figures/{name}' alt='{cohort} {window} overview'>")
    primary = units.filter(pl.col("selection_half") == 0)
    for key, frame in primary.group_by(
        "structure", "cohort", "window", maintain_order=True
    ):
        area, cohort, window = key
        filters = [
            pl.col("structure") == area,
            pl.col("cohort") == cohort,
            pl.col("window") == window,
            pl.col("selection_half") == 0,
        ]
        print(f"Rendering {area} {cohort} {window}", flush=True)
        fig = area_figure(
            frame,
            summary.filter(*filters),
            mice.filter(*filters),
            counts.filter(*filters).row(0, named=True),
        )
        name = f"{area}_{cohort}_{window}.png"
        fig.savefig(figures / name, dpi=130)
        plt.close(fig)
        html.extend(
            [
                f"<h2>{escape(area)} · {cohort} · {window}</h2>",
                f"<img src='figures/{name}' alt='{escape(area)} {cohort} {window}'>",
            ]
        )
    (out / "index.html").write_text("\n".join(html))
    notes = [
        "# Stimulus-responsive cells: modality versus target preference",
        "",
        (
            "Primary results use responsiveness selection on half 0 and preference estimation on half 1. "
            "The reverse split is a sensitivity analysis with a separately selected population; the two are not independent replications."
        ),
        "",
        (
            "Responsiveness: a two-sided paired-rate t test for each stimulus/context condition, Bonferroni correction "
            f"within cell, then BH q ≤ {alpha:g} within area × cohort × window × selection half. "
            "Trial independence and the approximate t-test calibration are assumptions. "
            "Area effects average sessions within mice, then weight mice equally."
        ),
        "",
    ]
    for cohort in sorted(units["cohort"].unique()):
        notes.extend([f"## {cohort}: primary 0–100 ms", ""])
        c = counts.filter(
            pl.col("cohort") == cohort,
            pl.col("window") == "early",
            pl.col("selection_half") == 0,
        )
        a = summary.filter(
            pl.col("cohort") == cohort,
            pl.col("window") == "early",
            pl.col("selection_half") == 0,
        )
        notes.append(
            f"Selected {c['n_selected'].sum():,} / {c['n_tested'].sum():,} QC cells."
        )
        if cohort == "naive":
            notes.append(
                "Only one anatomically annotated naive mouse is available. These observations are descriptive; no population strength-comparison p-value is assigned."
            )
        for metric, label in (
            ("modality_power", "Reproducible modality preference"),
            ("target_power", "Reproducible target/nontarget preference"),
        ):
            passed = a.filter(pl.col("metric") == metric, pl.col("q_value") < 0.05)[
                "structure"
            ].to_list()
            notes.append(
                f"{label} at conditional shuffle q < .05: {', '.join(passed) or 'none'}."
            )
        d = a.filter(pl.col("metric") == "power_difference")
        for positive, label in (
            (True, "Modality stronger"),
            (False, "Target preference stronger"),
        ):
            condition = pl.col("ci_low") > 0 if positive else pl.col("ci_high") < 0
            passed = d.filter(pl.col("q_value") < 0.05, condition)[
                "structure"
            ].to_list()
            notes.append(
                f"{label}, requiring paired mouse q < .05 and a same-direction 95% mouse interval: {', '.join(passed) or 'none'}."
            )
        notes += [
            "",
            "| Area | Responsive / QC | Mice | Modality strength | Target strength | Difference [95% CI] | Paired q | Reverse difference (q) |",
            "|---|---:|---:|---:|---:|---|---:|---|",
        ]
        for row in c.sort("structure").iter_rows(named=True):
            area = row["structure"]
            stats = {
                r["metric"]: r
                for r in a.filter(pl.col("structure") == area).iter_rows(named=True)
            }
            if "power_difference" not in stats:
                notes.append(
                    f"| {area} | 0 / {row['n_tested']} | 0 | — | — | — | — | — |"
                )
                continue
            diff = stats["power_difference"]
            reverse = summary.filter(
                pl.col("structure") == area,
                pl.col("cohort") == cohort,
                pl.col("window") == "early",
                pl.col("selection_half") == 1,
                pl.col("metric") == "power_difference",
            )
            reverse_text = "unavailable"
            if reverse.height:
                r = reverse.row(0, named=True)
                reverse_text = f"{r['estimate']:.3g} ({r['q_value']:.3g})"
            notes.append(
                f"| {area} | {row['n_selected']} / {row['n_tested']} | {diff['n_mice']} | "
                f"{stats['modality_power']['estimate']:.3g} | {stats['target_power']['estimate']:.3g} | "
                f"{diff['estimate']:.3g} [{diff['ci_low']:.3g}, {diff['ci_high']:.3g}] | {diff['q_value']:.3g} | {reverse_text} |"
            )
        notes.append("")
    notes += [
        (
            "Strength is the cross-validated squared contrast divided by the same independently estimated response RMS² for both factors. "
            "It is not a fraction of cells, correlation, or classification accuracy. Negative estimates are retained as sampling noise around weak signals. "
            "The tables also report the unnormalized paired comparison in Hz², and descriptive held-out absolute differences in Hz."
        ),
        "",
        (
            "A significant modality contrast does not establish a distinct anatomical pathway. A nonsignificant modality-versus-target comparison does not establish equivalence. "
            "A target stimulus is the potentially rewarded stimulus identity, averaged across contexts; this comparison does not isolate learned target status from physical stimulus identity."
        ),
        "",
        "[Figures](index.html) · [Area statistics](area_summary.csv) · [Mouse estimates](mouse_estimates.csv) · [Selection overlap](selection_overlap.csv)",
    ]
    (out / "REPORT.md").write_text("\n".join(notes) + "\n")
