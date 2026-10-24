# Modality versus target preference among stimulus-responsive cells

The completed dataset is summarized in the [initial results](RESULTS_2026-10-23.md),
including the different conclusions for early and late responses.

This analysis asks whether **stimulus-responsive cells prefer one modality**, and
whether **that preference is stronger than their target-versus-nontarget
preference**. It separates response detection from preference estimation, and
compares the two preferences in the same cells using matched contrasts.

Open the [figure report](../../results/responsive_preferences/full/index.html)
or the [results summary](../../results/responsive_preferences/full/REPORT.md)
after running the analysis. These are new outputs; the earlier context and
spatial analysis is retained separately.

## Data and scope

The input is the completed trial-rate cache from
`results/modality_context/full_expanded`. It contains 35,576 QC-passing neurons
from 81 good-behavior expert sessions and one anatomically annotated naive
session. The same 17 areas are retained: ACAd, ACAv, CA1, CA3, CP, DG, MOp, MOs,
MRN, ORBl, ORBvl, PL, SCm, SSp, SSs, SUB, and VISp. SCm remains exactly SCiw +
SCig + SCdg. Area eligibility is determined before responsiveness selection;
areas are not dropped merely because fewer than 1,000 responsive cells remain.

Trial exclusions and cohorts are inherited from the
[previous analysis](../context_preferences/README.md). No additional download
is required. Both increases and decreases in firing count as responses.
Primary response window: **0–100 ms**. Secondary window: **100–500 ms**. Baseline:
**−200–0 ms**. The late window can include action and reward effects. These are
event-aligned rate analyses, not fits of the behavior-adjusted gain GLM.

## Selection without reusing preference-estimation trials

1. Use the cached split into halves, balanced within actual block × stimulus.
2. On **half 0**, compute each trial's post-stimulus minus prestimulus firing
   rate. Rates account for the different pre/post window durations.
3. Within each stimulus × context condition, test whether the mean paired rate
   difference is zero, using a **two-sided one-sample t test**. Conditions are
   tested separately so excitation, suppression, and context-specific responses
   can all qualify. Require at least five selection trials per condition.
4. A cell's any-response p-value is `min(1, K × min(condition p))`, where K is
   the number of available conditions (normally eight). This Bonferroni test
   accounts for looking for a response in any condition. Apply BH correction
   across all cells in the same **area × cohort × window × selection half**.
   Select cells at response **q ≤ .05**.
5. Estimate both preferences exclusively on **half 1**. The selection test and
   response-strength denominator never use these trials.
6. Repeat with the halves reversed as a prespecified sensitivity analysis.
   The two analyses have different selected populations. They are not combined
   by taking their union, averaging their neuron rows, or choosing the more
   significant result. The reverse split is not an independent replication.

The paired-rate t test is a test of a mean response, with approximate calibration
for non-Gaussian spike-count differences and an independence assumption across
trials. Bonferroni and BH correction do not repair a misspecified single-test
model. Synthetic checks include a stationary Poisson null at several rates with
the actual unequal pre/post durations, plus recovery of excitation and suppression.
See the primary [SciPy t-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
and [multiple-testing documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

The same selection rule is applied to both preferences: cells are not selected
for having a modality or target difference. A cell responding equally to all
four stimuli can therefore qualify. Selection on half the trials reduces power,
but makes the subsequent comparison interpretable without selection-induced
inflation. Selection uncertainty remains; counts and split overlap are reported.

## Matched preference definitions

Let VT, VN, AT, and AN be the held-out mean baseline-subtracted responses to the
visual target, visual nontarget, auditory target, and auditory nontarget.
Each available context receives equal weight.

| Quantity | Definition | Positive direction |
|---|---|---|
| Modality preference M | `(VT + VN − AT − AN)/2` | Higher response to vision |
| Target preference T | `(VT + AT − VN − AN)/2` | Higher response to targets |

The coefficients have identical norms and both contrasts use the same trials.
“Higher” means a higher signed evoked firing rate; it can mean less suppression.
These are not contrasts of absolute response magnitudes. “Target” denotes the
potentially rewarded stimulus identity, averaged across both contexts, including
its presentations in the irrelevant context. This design cannot separate learned
target status from the physical identities assigned as targets.

Two distribution views are provided:

- **Hz:** signed held-out differences, in the same physical units for M and T.
- **Relative to response strength:** divide each contrast by the same cell's
  selection-half RMS response over stimulus × context conditions. The RMS uses
  signed responses squared, thus including suppression. It is floored at **1 Hz**
  to prevent an almost-zero denominator from creating extreme relative preferences.
  This scale uses no evaluation trials. The resulting values are not bounded
  selectivity indices and can exceed one.

Both histograms weight mice equally, sessions equally within mice, and selected
cells equally within sessions. The same-cell scatter compares absolute held-out
differences. These distributions and absolute differences are descriptive:
finite-trial noise can still broaden them even after independent cell selection.
The normalized distribution's shape also depends on the denominator: a cell
responding equally to both visual stimuli and zero to both auditory stimuli
approaches a normalized modality difference of √2 when the 1 Hz floor is inactive.
Peaks near such values are not, by themselves, evidence for discrete cell classes.

## Are the preferences reproducible, and which is stronger?

Split the **evaluation half** again into A and B, balanced within block × stimulus.
Require at least two trials per stimulus/context in each evaluation split.
Let S be the selection-derived response RMS, floored at 1 Hz. Calculate:

```
modality strength = M_A × M_B / S²
target strength   = T_A × T_B / S²
paired difference = (M_A × M_B − T_A × T_B) / S²
```

Products of disjoint-trial estimates avoid the positive noise contribution that
would arise from squaring a noisy estimate. Negative estimated strengths are
retained, not clipped or square-rooted. The estimand is mean squared preference,
not mean absolute preference or the fraction of cells with a preference.
Positive paired differences favor modality; negative differences favor targets.

For each area and session, average over its selected cells. Average sessions
equally within mice, then mice equally. This uses the same cells and weights for
both factors. A session with no selected cells does not contribute to the
conditional preference estimate, but remains in the response-selection audit.

There are two distinct inferential tests:

- **Preference versus a stimulus-label null.** Shuffle modality labels within
  actual block × target status × evaluation split. For the target contrast,
  shuffle target status within block × modality × evaluation split. Thus each
  null preserves the other factor. The selection set and response scale stay
  fixed. All simultaneously recorded cells share a shuffle. The area test
  compares the cross-validated strength with 1,999 such shuffles, using the
  same session/mouse weights. Its p-value is conditional on the sampled sessions.
- **Modality strength versus target strength.** Compute their paired difference
  for every mouse. Obtain a 95% interval by 10,000 mouse bootstrap draws. Test
  zero paired difference by sign-flipping whole mouse effects, exactly for at
  most 15 mice and with 19,999 Monte Carlo assignments otherwise. This test
  assumes symmetric mouse effects under the null. It does not treat neurons as
  independent animals. Fewer than three mice receive no population-comparison
  p-value. An additional paired comparison in **unnormalized Hz²** checks whether
  the conclusion depends on response-strength normalization.

Area q-values use one family across all finite tested area × cohort × window ×
metric hypotheses within each split orientation. Primary and reverse orientations
have separate, prespecified families. Figure fill denotes q < .05; bars show
mouse bootstrap intervals. For the reported stronger-preference conclusions,
require both paired q < .05 and a same-direction mouse interval excluding zero.

The cell-level modality and target p-values use the full held-out contrast and
their respective shuffle distributions. Correct **both factors jointly** within
each area × cohort × window × selection half. Category plots show modality-only,
target-only, both, and neither detected at unit q < .05. Neither detected does
not mean the cell has equal responses. Different p-values or detected fractions
alone are not used as the primary evidence that one preference is stronger.

The naive data contain only one mouse with usable area assignments, so naive
preference-strength comparisons remain descriptive. Selection and inference
are otherwise identical across cohorts.

## Running and resuming

```bash
# Full cached dataset, with a persistent log and recorded PID.
.venv/bin/python scripts/launch_responsive_preferences.py
tail -f results/responsive_preferences/full/run.log

# Equivalent foreground run.
caffeinate -i .venv/bin/python -u -m gain_glm.responsive_preferences

# Smoke test; response-selection FDR still uses the full source dataset.
caffeinate -i .venv/bin/python -u -m gain_glm.responsive_preferences \
  --output-dir results/responsive_preferences/smoke --permutations 99 \
  --session 626791_2022-08-15 --session 746439_2025-01-27

.venv/bin/python -m pytest tests/test_responsive_preferences.py tests/test_context_preferences.py
```

Identical runs reuse screening and evaluation caches. Source-code hashes,
input-data hashes, parameters, package versions, and a source snapshot are saved.
Changed data, code, or statistical configuration require a fresh output directory.
The run updates `status.json`; failures terminate with a traceback and failed
status rather than silently reducing the selected dataset. `caffeinate -i`
prevents idle system sleep while the process runs.

## Organization and outputs

Implementation is in `src/gain_glm/responsive_preferences/`:

- `statistics.py`: response detection, balanced held-out contrasts, matched nulls,
  and paired mouse inference.
- `analysis.py`: two-stage analysis, multiple-testing correction, and aggregation.
- `plots.py`: preference distributions, direct comparisons, and report generation.
- `__main__.py`: provenance, resumable execution, and status reporting.

Key output files:

- `index.html`: area figures and across-area overview plots.
- `REPORT.md`: primary conclusions, counts, confidence intervals, and reverse-split results.
- `selection_all_units.parquet`: every screened cell, condition-level response
  p-values and means, selection q-value, eligibility, and held-out estimates.
- `selected_neurons.parquet`: selected cells with both held-out preferences,
  response normalization, strengths, and selectivity p/q values.
- `selection_counts.csv`: responsive and tested counts for each area/cohort/window/split.
- `selection_overlap.csv`: how often cells are selected in both orientations.
- `area_summary.csv`: estimates, intervals, p/q, test type, mice, sessions, and counts.
- `session_summary.csv` / `.parquet`: session effects; parquet retains shuffle distributions.
- `mouse_estimates.csv`: the actual animal-level effects used for paired comparisons.
- `figures/overview_<cohort>_<window>.png`: both preference strengths, the primary
  paired comparison, and the reverse-split comparison.
- `figures/<area>_<cohort>_<window>.png`: six-panel figures for the primary orientation.

Selected-cell rows appear separately for each window and split orientation;
do not count those repetitions as distinct neurons. The 0–100 ms window and
selection-half-0 orientation are the designated primary analysis.
