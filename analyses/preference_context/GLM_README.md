# Link stimulus preference to the notebook's baseline-context population

This is the direct analysis of the user's question using the **existing GLM
definition of baseline context encoding**. It complements the
[trial-disjoint check](README.md), whose conservative block-level baseline test
has too little power for reliable cell classification in this dataset.

## Inputs and baseline labels

`glm_analysis.ipynb` currently loads the newer fits under
`/Volumes/programs-2/mindscope/workgroups/dynamicrouting/corbettb/gain_glm/100926`
and saves/loads `/Users/corbettb/Documents/scratch/gain_glm_df_100626.parquet`.
Despite the older date in the parquet filename, its scalar permutation results
match sampled records in the newer JSON directory. The user's corrected mount
also exposes `091626`, but those older fits lack the baseline permutation results
used by the notebook. We use the notebook's existing parquet, not those older fits.

The full parquet is input-hashed. `source_audit.json` checks identical session
sets and exact equality of four baseline-permutation fields for the first unit
in five spread-out session JSONs. This is a source check, not a full reparse of
every remote JSON. A local scalar QC table is saved for reproducibility.

All **35,170 expert QC cells** in the selected 17 areas have finite GLM baseline
permutation results. The **406 naive QC cells have no saved GLM fits**. Missing
fits are excluded explicitly, never counted as non-encoding. Anatomical grouping
comes from the previously audited dataset, including SCm = SCiw + SCig + SCdg.

The notebook creates its `*_significant` flags using **permutation p < .05**, not
FDR-adjusted p-values. The direct comparison preserves that label:

```
baseline encoding = context_baseline_dropout_permutation_p_value < 0.05
```

This is a nominal cell-level population definition. Group-comparison p-values
receive separate across-area FDR correction. We also recompute baseline-cell BH
q-values over **all GLM-covered QC cells within area/cohort**, before selecting
responsive cells or preference classes. This is an explicit stricter sensitivity
analysis, not the notebook's primary population definition. There are only 62–81
unique shifts per cell in the included dataset; the coarse p-value resolution
limits cell-level FDR power. No included cells pass this stricter correction.

The continuous outcome is
`context_baseline_dropout_permutation_observed_delta_r2`, the mean-fold observed
baseline-dropout effect used for the permutation analysis. This is the observed
statistic on the permutation-compatible evaluation rows, which can differ from
the ordinary dropout `delta_r2`; both are retained. Negative effects are retained.

## Preferences and comparisons

Use `results/responsive_preferences/full/selected_neurons.parquet` unchanged.
Responsiveness is selected on one half of trials, and both preferences are
estimated and tested on the other half. The contrast definitions, balanced
contexts, conditional trial shuffles, and joint modality/target FDR are described
in the [responsive-cell guide](../responsive_preferences/README.md).

This uses the **full evaluation half**, giving more preference-classification
power than the separate baseline-quarter analysis. Primary early responses retain
3,900 expert cells. Repeat for late responses and reverse the selection/evaluation
halves as a sensitivity check. Do not count repeated rows as additional neurons.

- **Exclusive:** modality-only versus target/nontarget-only; both and neither
  detected are displayed separately. “Only” means only that preference was
  detected, not proof that the other preference is zero.
- **Dominant:** among cells with either preference detected, |M| > |T| versus
  |T| > |M|; this retains cells selective for both. This is an estimated dominance
  label, not an individual-cell significance test of the strength difference.

The saved GLMs were fitted using overlapping trial data. Consequently, this
comparison reproduces the user's baseline-population definition but **does not
provide an independent-trial validation** of the association. It cannot establish
causal priming. The separate trial-disjoint analysis retains that stricter design.

## Session/mouse inference and sensitivity checks

Compare the two populations only within the same area/session, with at least
three cells in each group. Average the difference equally across sessions within
mouse, then mice equally. Report the two group fractions using exactly those
same sessions and weights. Pooled bar charts are descriptive and use different
weights; their fractions can differ from the paired estimates.

Bootstrap mice 10,000 times for 95% intervals. Test zero difference with two-sided
mouse sign flips (exact for ≤15 mice, otherwise 19,999 assignments); this assumes
symmetric mouse effects under the null. Fewer than three mice get no population
p-value. When every cell has the same binary outcome, suppress the misleading
zero-width bootstrap interval and mark the comparison uninformative.

Correct both categorical comparisons together across areas within each
window × preference split × fit subset × adjustment × outcome family. The
continuous partial-correlation test has a separate across-area family. This is
not one global FDR over every exploratory sensitivity analysis.

Sensitivity checks:

- One-to-one matching within area/session on log(1 + selection-half baseline
  firing rate), log(1 + response RMS), and log(1 + sqrt(M² + T²)). Standardize
  within the two groups, apply Euclidean caliper 1.5, and require ≥3 matched pairs.
  Matching never sees baseline GLM outcomes. Retain match IDs and feature balance.
- Baseline-cell BH q < .05 instead of the notebook's nominal p < .05.
- Continuous observed baseline GLM ΔR² instead of a cell-level threshold.
- Converged fits only: final fit, full-model cross-validation, and baseline
  dropout metric must all report convergence. The primary comparison matches
  the notebook's inclusion of all finite tests; it does not silently discard
  nonconverged fits. There are 17,726 fully converged QC cells and 3,281 of the
  3,900 primary early responsive cells meet this stricter criterion.
- Partial Spearman correlation of `(M²−T²)/(M²+T²)` with baseline GLM ΔR²,
  controlling the same ranked matching variables, within sessions containing
  ≥10 responsive cells; aggregate and test at mouse level.
- Reversed response-selection/evaluation halves.

Matching and within-session comparison reduce measured confounding. They can
also sharply reduce support, especially for the rare target-only group. All
exclusions and sample sizes are available in `comparison_support.csv`.

## Run and outputs

```bash
caffeinate -i .venv/bin/python -u -m gain_glm.preference_context.glm

# Optional explicit inputs/output:
caffeinate -i .venv/bin/python -u -m gain_glm.preference_context.glm \
  --glm-parquet /Users/corbettb/Documents/scratch/gain_glm_df_100626.parquet \
  --output-dir results/preference_context/glm

.venv/bin/python -m pytest tests/test_preference_context_glm.py \
  tests/test_preference_context.py tests/test_responsive_preferences.py \
  tests/test_context_preferences.py
```

Implementation: `src/gain_glm/preference_context/glm/`, with `analysis.py` for
linkage/comparisons, `plots.py` for reports, and `__main__.py` for execution and
provenance. Default output: `results/preference_context/glm/`.

- `index.html`, `figures/`: 34 area/window figures and early/late overviews.
- `REPORT.md`: primary mouse-weighted comparisons for both windows.
- `linked_neurons.parquet`: responsive cells, original preferences and labels,
  saved GLM metrics, strict convergence flags, and matching covariates.
- `glm_qc_cells.parquet`, `glm_coverage.csv`: all included QC cells and coverage.
- `class_counts.csv`, `session_classes.csv`: overlap and descriptive fractions.
- `area_summary.csv`, `session_comparisons.csv`, `mouse_estimates.csv`: estimates,
  confidence intervals, p/q, comparison definitions, and sample support.
- `comparison_support.csv`, `matched_pairs.csv`: exclusions and matching audit.
- `configuration.json`, `provenance.json`, `source_snapshot/`, `source_audit.json`,
  `status.json`: reproducibility and completion records.

Identical runs may be rerun; changed code or data require a fresh output directory.
The new analysis does not alter the notebook or original GLM fits.
