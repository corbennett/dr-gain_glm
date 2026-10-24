"""Clearly label descriptive cell fractions versus paired animal comparisons."""

from __future__ import annotations

import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

COLORS = {
    "modality_only": "#3478b8",
    "target_only": "#d77d28",
    "both": "#8b62a8",
    "neither": "#999999",
    "modality_dominant": "#3478b8",
    "target_dominant": "#d77d28",
}


def _effect(ax, frame, metric, percent=False):
    scale = 100 if percent else 1
    labels = []
    for i, (comparison, adjustment) in enumerate(
        (
            ("exclusive", "unmatched"),
            ("exclusive", "matched"),
            ("dominant", "unmatched"),
            ("dominant", "matched"),
        )
    ):
        row = frame.filter(
            pl.col("comparison") == comparison,
            pl.col("adjustment") == adjustment,
            pl.col("metric") == metric,
        )
        labels.append(f"{comparison}, {adjustment}")
        if not row.height:
            ax.text(
                0.02,
                i,
                "insufficient groups",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
            )
            continue
        r = row.row(0, named=True)
        if r["test"] == "no_outcome_variation":
            ax.text(
                0.02,
                i,
                "no detected context cells",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
            )
            continue
        x = scale * r["estimate"]
        low, high = scale * r["ci_low"], scale * r["ci_high"]
        color = "#20855e" if r["q_value"] < 0.05 else "#555555"
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [i, i], color=color)
        ax.scatter(x, i, color=color, s=26)
        ax.annotate(
            f" {r['n_mice']} mice",
            (x, i),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(4), labels, fontsize=8)
    ax.set_ylim(3.7, -0.6)
    ax.set_xlabel(
        "Modality − target: percentage points"
        if percent
        else "Modality − target: baseline power"
    )


def report(out, units, summary, mice, classes):
    del mice, classes
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    primary = units.filter(
        pl.col("selection_half") == 0, pl.col("preference_quarter") == 0
    )
    main = summary.filter(
        pl.col("selection_half") == 0, pl.col("preference_quarter") == 0
    )
    pages = []
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    for key, u in primary.group_by(
        "structure", "cohort", "window", maintain_order=True
    ):
        area, cohort, window = key
        f = main.filter(
            pl.col("structure") == area,
            pl.col("cohort") == cohort,
            pl.col("window") == window,
        )
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        for ax, scheme, labels in (
            (
                axes[0, 0],
                "exclusive",
                ("modality_only", "target_only", "both", "neither"),
            ),
            (axes[0, 1], "dominant", ("modality_dominant", "target_dominant")),
        ):
            for i, label in enumerate(labels):
                part = u.filter(pl.col(f"{scheme}_class") == label)
                n = part.height
                encoded = int(part["encoding_probability"].sum()) if n else 0
                height = 100 * encoded / n if n else 0
                ax.bar(i, height, color=COLORS[label], alpha=0.8)
                ax.text(i, height + 1.5, f"{encoded}/{n}", ha="center", fontsize=8)
            ax.set_xticks(range(len(labels)), [x.replace("_", "\n") for x in labels])
            ax.set_ylim(0, 105)
            ax.set_ylabel("Baseline context detected (%)")
            ax.set_title("Pooled cells, descriptive; labels = encoded / tested")
        _effect(axes[0, 2], f, "encoding_probability", percent=True)
        axes[0, 2].set_title(
            "Within-session comparison, equal mouse weight\n95% mouse intervals; green: area q < .05"
        )
        _effect(axes[1, 0], f, "block_pair_power")
        axes[1, 0].set_title(
            "Context modulation repeated across block pairs\nNegative estimates retained"
        )
        _effect(axes[1, 1], f, "linear_encoding_probability", percent=True)
        axes[1, 1].set_title("Sensitivity: linear drift instead of cubic drift")
        ax = axes[1, 2]
        for i, (half, quarter) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            part = summary.filter(
                pl.col("structure") == area,
                pl.col("cohort") == cohort,
                pl.col("window") == window,
                pl.col("selection_half") == half,
                pl.col("preference_quarter") == quarter,
                pl.col("comparison") == "dominant",
                pl.col("adjustment") == "unmatched",
                pl.col("metric") == "encoding_probability",
            )
            if part.height:
                r = part.row(0, named=True)
                if r["test"] == "no_outcome_variation":
                    ax.text(
                        0.02,
                        i,
                        "no detected context cells",
                        transform=ax.get_yaxis_transform(),
                        fontsize=8,
                    )
                    continue
                ax.plot(
                    100 * np.array([r["ci_low"], r["ci_high"]]), [i, i], color="#555555"
                )
                ax.scatter(
                    100 * r["estimate"],
                    i,
                    color="#20855e" if r["q_value"] < 0.05 else "#555555",
                )
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(
            range(4),
            [
                "Primary (0, 0)",
                "Swap evaluation quarters",
                "Reverse selection (1, 0)",
                "Reverse + swap (1, 1)",
            ],
        )
        ax.set_ylim(3.6, -0.6)
        ax.set_xlabel("Modality-dominant − target-dominant (pp)")
        ax.set_title(
            "Split sensitivity: overlapping reanalyses\nDo not treat these as independent replications"
        )
        fig.suptitle(
            f"{area} · {cohort} · {'0–100 ms' if window == 'early' else '100–500 ms'} preference\n"
            f"{u.height} responsive cells; baseline −200–0 ms on separate trials",
            fontsize=14,
        )
        name = f"{area}_{cohort}_{window}.png"
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
        pages.append((f"{area} / {cohort} / {window}", name))
    for cohort in primary["cohort"].unique().sort():
        for window in ("early", "late"):
            f = main.filter(pl.col("cohort") == cohort, pl.col("window") == window)
            areas = (
                primary.filter(pl.col("cohort") == cohort)["structure"]
                .unique()
                .sort()
                .to_list()
            )
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(14, max(5, 0.42 * len(areas))),
                sharey=True,
                constrained_layout=True,
            )
            for ax, comparison, adjustment in zip(
                axes,
                ("exclusive", "dominant", "dominant"),
                ("unmatched", "unmatched", "matched"),
            ):
                for i, area in enumerate(areas):
                    row = f.filter(
                        pl.col("structure") == area,
                        pl.col("comparison") == comparison,
                        pl.col("adjustment") == adjustment,
                        pl.col("metric") == "encoding_probability",
                    )
                    if not row.height:
                        ax.text(
                            0.03,
                            i,
                            "insufficient groups",
                            transform=ax.get_yaxis_transform(),
                            fontsize=7,
                        )
                        continue
                    r = row.row(0, named=True)
                    if r["test"] == "no_outcome_variation":
                        ax.text(
                            0.03,
                            i,
                            "no detected context cells",
                            transform=ax.get_yaxis_transform(),
                            fontsize=7,
                        )
                        continue
                    color = "#20855e" if r["q_value"] < 0.05 else "#555555"
                    ax.plot(
                        100 * np.array([r["ci_low"], r["ci_high"]]), [i, i], color=color
                    )
                    ax.scatter(100 * r["estimate"], i, color=color)
                ax.set_yticks(range(len(areas)), areas)
                ax.set_ylim(len(areas) - 0.4, -0.6)
                ax.axvline(0, color="k", lw=0.7)
                ax.set_title(f"{comparison}, {adjustment}")
                ax.set_xlabel("Baseline context probability: M − T (pp)")
            fig.suptitle(
                f"{cohort} · {window} · within-session differences, equal mouse weight\n"
                "95% mouse bootstrap intervals; green: q < .05; primary split"
            )
            name = f"overview_{cohort}_{window}.png"
            fig.savefig(figures / name, dpi=150)
            plt.close(fig)
            pages.insert(0, (f"Overview / {cohort} / {window}", name))
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(14, max(5, 0.42 * len(areas))),
                sharey=True,
                constrained_layout=True,
            )
            for ax, comparison, adjustment, metric in zip(
                axes,
                ("dominant", "dominant", "continuous"),
                ("unmatched", "matched", "partial_rank"),
                ("block_pair_power", "block_pair_power", "block_pair_correlation"),
            ):
                for i, area in enumerate(areas):
                    row = f.filter(
                        pl.col("structure") == area,
                        pl.col("comparison") == comparison,
                        pl.col("adjustment") == adjustment,
                        pl.col("metric") == metric,
                    )
                    if not row.height:
                        ax.text(
                            0.03,
                            i,
                            "insufficient support",
                            transform=ax.get_yaxis_transform(),
                            fontsize=7,
                        )
                        continue
                    r = row.row(0, named=True)
                    color = "#20855e" if r["q_value"] < 0.05 else "#555555"
                    ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color)
                    ax.scatter(r["estimate"], i, color=color)
                    ax.annotate(
                        f" {r['n_mice']} mice",
                        (r["estimate"], i),
                        xytext=(2, 4),
                        textcoords="offset points",
                        fontsize=6,
                    )
                ax.set_yticks(range(len(areas)), areas)
                ax.set_ylim(len(areas) - 0.4, -0.6)
                ax.axvline(0, color="k", lw=0.7)
                ax.set_title(f"{comparison}, {adjustment}")
                ax.set_xlabel(
                    "M − T: cross-block baseline strength"
                    if metric == "block_pair_power"
                    else "Preference balance vs baseline strength: partial ρ"
                )
            fig.suptitle(
                f"{cohort} · {window} · baseline context strength without a cell-level threshold\n"
                "95% mouse bootstrap intervals; green: q < .05; primary split; <3 mice descriptive"
            )
            name = f"continuous_{cohort}_{window}.png"
            fig.savefig(figures / name, dpi=150)
            plt.close(fig)
            pages.insert(
                0, (f"Continuous context strength / {cohort} / {window}", name)
            )
    lines = [
        "# Preference type and baseline context encoding",
        "",
        "Baseline detection uses a block-clustered CR2/Satterthwaite test with cubic time drift,",
        "BH q < .05 within area/cohort/window/split. Only 4–6 context blocks are available;",
        "absence of detection is not evidence of absent encoding. Preferences and baseline use",
        "disjoint evaluation quarters; responsive-cell selection uses the other half.",
        "",
        "Primary: early responses, selection half 0, preference quarter 0. Differences below",
        "compare groups within the same sessions (at least 3 cells per group), then average",
        "sessions within mouse and mice equally. Intervals bootstrap mice. One naive mouse",
        "is descriptive only. Each sensitivity comparison has its own across-area FDR family.",
        "",
    ]
    for comparison, adjustment in (
        ("exclusive", "unmatched"),
        ("dominant", "unmatched"),
        ("dominant", "matched"),
    ):
        lines += [
            f"## Expert early: {comparison}, {adjustment}",
            "",
            "| Area | M context % | T context % | Difference pp [95% CI] | q | M / T cells | Mice |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
        f = main.filter(
            pl.col("cohort") == "expert",
            pl.col("window") == "early",
            pl.col("comparison") == comparison,
            pl.col("adjustment") == adjustment,
            pl.col("metric") == "encoding_probability",
        ).sort("structure")
        for r in f.iter_rows(named=True):
            lines.append(
                f"| {r['structure']} | {100 * r['left']:.1f} | {100 * r['right']:.1f} | "
                f"{100 * r['estimate']:+.1f} [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] | "
                f"{r['q_value']:.3g} | {r['n_left']} / {r['n_right']} | {r['n_mice']} |"
            )
        lines += [
            "",
            "Areas omitted here lack qualifying paired sessions; see comparison_support.csv.",
            "",
        ]
    lines += [
        "See area_summary.csv for both windows, all four splits, linear-drift and block-pair",
        "strength checks, and the continuous partial-rank analysis. class_counts.csv records",
        "all detected and overlapping preference groups. matched_pairs.csv audits matching.",
        "",
        "Interpret this as an association with pre-stimulus context-related activity, not a test",
        "of causal priming or of the original GLM's context-baseline dropout significance.",
    ]
    primary_expert = primary.filter(
        pl.col("cohort") == "expert", pl.col("window") == "early"
    )
    n_detected = int(primary_expert["encoding_probability"].sum())
    n_target = primary_expert.filter(pl.col("exclusive_class") == "target_only").height
    n_modality = primary_expert.filter(
        pl.col("exclusive_class") == "modality_only"
    ).height
    conclusion = [
        (
            f"Primary early expert sample: {primary_expert.height:,} responsive cells, "
            f"including {n_modality:,} modality-only and {n_target:,} target-only cells. "
            f"The cubic-drift baseline test detects {n_detected:,} cells."
        ),
        "",
    ]
    if n_detected == 0:
        conclusion += [
            (
                "**The encoding-probability comparison is inconclusive:** the baseline test has "
                "no detections after correction. It does not show that encoding is absent or "
                "equally likely in the two groups. Degenerate all-zero bootstrap intervals "
                "and their p-values are suppressed. Use the continuous checks below."
            ),
            "",
        ]
    lines[2:2] = conclusion
    lines += [
        "",
        "## Primary early expert continuous checks",
        "",
        "| Area | Comparison | Adjustment | Effect [95% CI] | q | Mice |",
        "|---|---|---|---|---:|---:|",
    ]
    check = main.filter(
        pl.col("cohort") == "expert",
        pl.col("window") == "early",
        (
            (pl.col("comparison") == "dominant")
            & (pl.col("metric") == "block_pair_power")
        )
        | (pl.col("comparison") == "continuous"),
    ).sort("structure", "comparison", "adjustment")
    for r in check.iter_rows(named=True):
        lines.append(
            f"| {r['structure']} | {r['comparison']} | {r['adjustment']} | "
            f"{r['estimate']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{r['q_value']:.3g} | {r['n_mice']} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Preference and context</title>"
        "<style>body{font:16px system-ui;max-width:1500px;margin:30px auto}img{width:100%}section{margin:45px 0}</style>"
        "<h1>Preference type and baseline context encoding</h1><p><a href='REPORT.md'>Numerical report</a> · "
        "<a href='area_summary.csv'>All comparisons</a></p><p>Separate trials for selection, preference, and baseline. "
        "Baseline tests cluster by context block. Pooled bar charts are descriptive; inference compares groups "
        "within sessions, then weights mice equally. Naive results represent one mouse.</p>"
        + "".join(
            f"<section><h2>{html.escape(title)}</h2><img loading='lazy' src='figures/{name}'></section>"
            for title, name in pages
        )
    )
