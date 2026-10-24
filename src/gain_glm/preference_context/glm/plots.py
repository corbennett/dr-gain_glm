"""Saved-GLM association figures, including overlap and fit convergence."""

from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from gain_glm.preference_context.plots import COLORS, _effect


def report(out, units, summary, coverage):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    primary = units.filter(pl.col("selection_half") == 0)
    main = summary.filter(pl.col("selection_half") == 0)
    pages = []
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    for key, u in primary.group_by("structure", "window", maintain_order=True):
        area, window = key
        f = main.filter(pl.col("structure") == area, pl.col("window") == window)
        all_fits = f.filter(pl.col("fit_subset") == "all")
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
                k = int(part["glm_nominal_probability"].sum()) if n else 0
                h = 100 * k / n if n else 0
                ax.bar(i, h, color=COLORS[label])
                ax.text(i, h + 2, f"{k}/{n}", ha="center", fontsize=8)
            ax.set_xticks(range(len(labels)), [x.replace("_", "\n") for x in labels])
            ax.set_ylim(0, 110)
            ax.set_ylabel("GLM baseline context detected (%)")
            ax.set_title(
                "Notebook definition: permutation p < .05\nPooled counts, descriptive"
            )
        _effect(axes[0, 2], all_fits, "glm_nominal_probability", percent=True)
        axes[0, 2].set_title(
            "Within-session difference; equal mouse weight\n95% mouse intervals; green: area q < .05"
        )
        _effect(axes[1, 0], all_fits, "glm_fdr_probability", percent=True)
        axes[1, 0].set_title("Sensitivity: cell-level baseline FDR < .05")
        _effect(axes[1, 1], all_fits, "glm_delta_r2")
        axes[1, 1].set_xlabel("Modality − target: baseline GLM ΔR²")
        axes[1, 1].set_title(
            "Continuous baseline dropout effect\nPermutation-observed mean-fold ΔR²"
        )
        _effect(
            axes[1, 2],
            f.filter(pl.col("fit_subset") == "converged"),
            "glm_nominal_probability",
            percent=True,
        )
        axes[1, 2].set_title("Sensitivity: converged fits and baseline dropout")
        fig.suptitle(
            f"{area} · expert · {'0–100 ms' if window == 'early' else '100–500 ms'} preference\n"
            f"{u.height} responsive cells; saved GLMs use overlapping trial data",
            fontsize=14,
        )
        name = f"{area}_{window}.png"
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
        pages.append((f"{area} / {window}", name))
    for window in ("early", "late"):
        f = main.filter(
            pl.col("window") == window,
            pl.col("fit_subset") == "all",
            pl.col("metric") == "glm_nominal_probability",
        )
        areas = primary["structure"].unique().sort().to_list()
        fig, axes = plt.subplots(
            1, 3, figsize=(14, 8), sharey=True, constrained_layout=True
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
                color = "#20855e" if r["q_value"] < 0.05 else "#555555"
                ax.plot(
                    100 * np.array([r["ci_low"], r["ci_high"]]), [i, i], color=color
                )
                ax.scatter(100 * r["estimate"], i, color=color)
                ax.annotate(
                    f" {r['n_mice']} mice",
                    (100 * r["estimate"], i),
                    xytext=(2, 5),
                    textcoords="offset points",
                    fontsize=6,
                )
            ax.axvline(0, color="k", lw=0.7)
            ax.set_yticks(range(len(areas)), areas)
            ax.set_ylim(len(areas) - 0.4, -0.6)
            ax.set_title(f"{comparison}, {adjustment}")
            ax.set_xlabel("Baseline context encoding: modality − target (pp)")
        fig.suptitle(
            f"Expert {window} preferences versus saved GLM baseline context labels\n"
            "Baseline permutation p < .05; 95% mouse intervals; green: comparison q < .05; <3 mice descriptive"
        )
        name = f"overview_{window}.png"
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
        pages.insert(0, (f"Overview / {window}", name))
    lines = [
        "# Preference type versus the notebook's GLM baseline-context labels",
        "",
        "The primary baseline label is the notebook's nominal permutation p < .05.",
        "FDR across neurons and converged-only fits are sensitivity checks. Preference",
        "classification uses the original full held-out half; response selection uses",
        "the other half. Saved GLMs use overlapping trial data, so this is a retrospective",
        "association analysis, not an independent-trial validation or causal test.",
        "",
        "All 35,170 expert QC cells in the 17 areas have GLM results. The 406 naive cells",
        "have none and are explicitly excluded from this GLM-linked comparison.",
        "",
    ]
    for window in ("early", "late"):
        u = primary.filter(pl.col("window") == window)
        lines += [
            f"## {window.capitalize()} responses: primary selection split",
            "",
            (
                f"{u.height:,} responsive cells; {int(u['glm_nominal_probability'].sum()):,} "
                "have baseline-context permutation p < .05."
            ),
            "",
        ]
        for comparison, adjustment in (
            ("exclusive", "unmatched"),
            ("dominant", "unmatched"),
            ("dominant", "matched"),
        ):
            lines += [
                f"### {comparison}, {adjustment}",
                "",
                "| Area | M context % | T context % | Difference pp [95% CI] | q | M / T cells | Mice |",
                "|---|---:|---:|---|---:|---:|---:|",
            ]
            f = main.filter(
                pl.col("window") == window,
                pl.col("fit_subset") == "all",
                pl.col("comparison") == comparison,
                pl.col("adjustment") == adjustment,
                pl.col("metric") == "glm_nominal_probability",
            ).sort("structure")
            for r in f.iter_rows(named=True):
                lines.append(
                    f"| {r['structure']} | {100 * r['left']:.1f} | {100 * r['right']:.1f} | "
                    f"{100 * r['estimate']:+.1f} [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] | "
                    f"{r['q_value']:.3g} | {r['n_left']} / {r['n_right']} | {r['n_mice']} |"
                )
            lines += [
                "",
                "Only common eligible sessions contribute (minimum 3 cells per group).",
                "",
            ]
    lines += [
        "See area_summary.csv for the reversed preference split, FDR labels, continuous",
        "GLM effect sizes, partial correlations, and the converged-fit subset.",
        "",
        "Nominal cell-level p < .05 is a population label, not FDR control over cells.",
        "Area q-values correct both categorical comparisons across areas within each",
        "window/split/fit-subset/adjustment/outcome family. Pooled bars and paired mouse",
        "estimates use different populations/weights and need not be equal.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>GLM baseline context and preference</title>"
        "<style>body{font:16px system-ui;max-width:1500px;margin:30px auto}img{width:100%}section{margin:45px 0}</style>"
        "<h1>Saved GLM baseline context versus stimulus preference</h1><p><a href='REPORT.md'>Numerical report</a> · "
        "<a href='area_summary.csv'>All comparisons</a></p><p>Notebook baseline label: permutation p &lt; .05. "
        "Within-session comparisons weight mice equally. Saved GLMs and preferences use overlapping trials. "
        "No naive fits are available.</p>"
        + "".join(
            f"<section><h2>{html.escape(title)}</h2>"
            f"<img loading='lazy' src='figures/{name}'></section>"
            for title, name in pages
        )
    )
