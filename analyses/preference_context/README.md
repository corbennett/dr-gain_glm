# Does sensory preference predict baseline context encoding?

This analysis asks whether **modality-preferring cells are more likely to encode
context before the stimulus than target/nontarget-preferring cells**, within each
area. It also compares the strength of context modulation that repeats across
different context blocks, avoiding exclusive reliance on cell-level thresholds.

For the direct comparison using **the notebook's saved GLM baseline-context
labels**, see [GLM_README.md](GLM_README.md). Those results are in
`results/preference_context/glm/`. The analysis described below is the additional
trial-disjoint baseline check in `results/preference_context/full/`.

The implementation is `src/gain_glm/preference_context/`. It reuses the complete
cached 17-area expert/naive dataset and the independently selected responsive
cells from [responsive_preferences](../responsive_preferences/README.md). It does
not download or refit the full stimulus/time-bin GLM. Its baseline test is a new
trial-rate analysis, not the notebook's dropout test. After the user supplied the
corrected `/Volumes/programs-2/mindscope` mount, the saved GLM comparison was added
as a separate analysis, preserving the trial-disjoint results and their provenance.

## Data and three disjoint trial sets

The included areas are ACAd, ACAv, CA1, CA3, CP, DG, MOp, MOs, MRN, ORBl, ORBvl,
PL, SCm, SSp, SSs, SUB, and VISp. SCm pools SCiw, SCig, and SCdg. The cached source
contains 81 expert sessions and one anatomically annotated naive session, with
35,170 expert and 406 naive QC cells in these areas. Additional naive sessions
without usable anatomical assignments remain excluded, as documented in the
[original dataset guide](../context_preferences/README.md).

The primary analysis uses early responses (0–100 ms) and the original selection
half 0. Late responses (100–500 ms) are a separate analysis and may reflect
movement, reward, or other post-stimulus events. Baseline is −200–0 ms.

1. **Selection half:** use the existing significant-response selection (any of
   the eight stimulus/context conditions, two-sided paired-rate tests, Bonferroni
   across conditions and BH across units within area/cohort/window). This retains
   excitation and suppression. The primary early population is unchanged:
   3,900 expert and 44 naive cells. This half also supplies baseline firing rate,
   baseline within-block variability, and response strength for matching/scaling.
2. **Preference quarter:** half of the remaining trials, balanced within actual
   block × stimulus. Classify modality and target preference using only this set.
3. **Baseline quarter:** the other evaluation quarter. Measure baseline context
   effects using only this set. These baseline samples never enter response
   selection or preference classification for this orientation.

All splits are balanced within actual block × stimulus, with deterministic seeds.
Repeat after swapping evaluation quarters and after reversing the selection
half, yielding four orientations. These reuse data and are sensitivity checks,
not four independent replications. The default seed matches the source analysis.
Disjoint trials prevent direct reuse of the same spike-count noise; they do not
remove persistent shared state or establish independence between nearby trials.

## Two ways to define the compared populations

On the preference quarter, average baseline-subtracted rates equally across
contexts and compute matched contrasts:

```
M = (visual target + visual nontarget − auditory target − auditory nontarget) / 2
T = (visual target + auditory target − visual nontarget − auditory nontarget) / 2
```

M > 0 means higher visual response; T > 0 means higher target response. Negative
values prefer auditory stimuli or nontargets. Both signs count as a preference.
These compare signed firing rates, including less suppression, rather than
absolute response magnitudes. Target identity and learned target status are
confounded by the task's fixed physical stimulus assignments.

Use 1,999 conditional trial shuffles for each contrast: modality labels shuffle
within actual block × target status × evaluation quarter; target labels within
block × modality × evaluation quarter. All simultaneously recorded cells share
each shuffle. Correct both factor p-values jointly with BH within
area × cohort × window × orientation. Significance is q < .05.

- **Exclusive comparison:** modality-only versus target-only cells. Cells
  selective for both and for neither are displayed separately. A nonsignificant
  preference is undetected, not proven absent.
- **Dominance comparison:** among cells with at least one detected preference,
  compare |M| > |T| with |T| > |M|. This retains the overlapping population and
  is useful when target-only cells are rare. Dominance uses point estimates;
  it does not claim that the difference between |M| and |T| is significant for
  each individual cell. Exact ties are unclassified.

Quarter-trial classification has less power than the previous half-trial analysis;
its labels and counts need not be identical. Response selection is unchanged.

## Baseline context measurement

Fit baseline firing rate by OLS to visual-versus-auditory context, an intercept,
linear/quadratic/cubic session time, current stimulus identity, and previous-trial
reward, response, visual stimulus, and target indicators. Current stimulus is
included to absorb accidental imbalance, not to imply prediction of the upcoming
stimulus. The context coefficient is visual-context minus auditory-context Hz.

Context is assigned by blocks, so cell-level significance uses **CR2 cluster-robust
variance with actual blocks as clusters and a Satterthwaite t reference**. The
working covariance has unit diagonal and within-block correlation 0.5 (a fixed
random-intercept working model), independent between blocks. Both the adjustment
and degrees of freedom use the full regression design and this covariance.
The observed variance still uses empirical cluster scores. This follows the
[small-sample CR2 method](https://arxiv.org/abs/1601.01981), also implemented by
the authors' [clubSandwich package](https://github.com/jepusto/clubSandwich).

Apply BH to baseline p-values across all responsive cells within
area/cohort/window/orientation, **before** separating preference classes. Baseline
encoding is operationally defined as baseline q < .05. A sensitivity analysis
uses linear time drift instead of cubic drift, with a separate baseline FDR family.

There are only 4–6 context blocks per session. Effective degrees of freedom can
be very small, so these tests are conservative and often underpowered. Neither
non-detection nor zero detected fractions establishes absence of context coding.
CR2 remains a small-sample approximation; unmodeled dependence between blocks,
nonlinear state changes, and task-related behavior can still affect estimates.
This analysis measures association with context, not a causal priming mechanism.

**Threshold-free check:** split baseline trials into disjoint adjacent block pairs
(two or three pairs per session). Within each pair, estimate the context contrast
controlling the listed nuisance variables and a local linear time trend. Average
products of contrasts from different pairs, divided by the square of the cell's
selection-half within-block baseline SD (floored at 1 Hz). Negative products are
retained. A positive mean estimates squared context modulation that is stable
across pairs, assuming errors in different pairs are independent. This avoids
squaring trial noise and does not require cell-level significance. Pair-specific
coefficients and scales are saved. Their uncertainty can still be substantial.

## Comparing groups without treating cells as independent mice

Only compare groups **within the same area and session**, requiring at least
three cells in each group. Compute the modality-minus-target difference in:

- probability of detected baseline context encoding (primary cubic-drift model);
- probability under the linear-drift sensitivity model;
- baseline context strength repeated across block pairs.

Average session differences equally within mouse, then mice equally. Bootstrap
mice 10,000 times for 95% intervals. Test zero difference by two-sided mouse sign
flips (exact up to 15 mice, otherwise 19,999 assignments), assuming symmetric
mouse effects under the null. Fewer than three mice get no population p-value.
When both groups have no detections (or detection is universal), there is no
outcome variation: suppress the otherwise degenerate bootstrap interval and
p-value, and label the comparison uninformative. A zero-width interval would
misleadingly imply that equal encoding probabilities had been established.
Primary and sensitivity orientations are analyzed separately. For each
window/orientation/adjustment/outcome, BH jointly corrects both categorical
comparisons across all estimable area/cohort tests. The continuous correlation
has its own across-area family. These are separate families, not one global FDR.

Also repeat the comparisons after one-to-one matching within area/session on
log(1 + baseline firing rate), log(1 + response RMS), and
log(1 + sqrt(M² + T²)). Standardize features across the two groups within that
session. Use minimum-distance assignment with a Euclidean caliper of 1.5,
prioritizing the number of admissible pairs; require at least three matched pairs.
Matching never accesses baseline-test outcomes. Save cell IDs, distances, and
standardized feature differences so balance and exclusions can be audited.
Matching reduces measured power/rate differences; it does not guarantee removal
of all confounding and can substantially reduce the population being compared.

A continuous sensitivity check uses all responsive cells in a session (minimum
10): partial Spearman correlation of `(M² − T²)/(M² + T²)` with cross-block baseline
strength, controlling the same three ranked matching variables. Average and test
these correlations at mouse level. Noisy preference estimates can attenuate or
otherwise distort this association; it is not an error-free latent selectivity test.

Pooled cell-fraction bar charts are **descriptive**. The paired estimates use
common eligible sessions and their own weights, so they can differ from the bars.
Counts and missing comparisons are explicit. The one annotated naive mouse
supports descriptive comparisons only.

## Run, resume, and outputs

```bash
.venv/bin/python scripts/launch_preference_context.py
tail -f results/preference_context/full/run.log

# Equivalent foreground run, also preventing idle sleep:
caffeinate -i .venv/bin/python -u -m gain_glm.preference_context

# Smoke test only; classification FDR is over this subset:
caffeinate -i .venv/bin/python -u -m gain_glm.preference_context \
  --output-dir results/preference_context/smoke --permutations 199 \
  --session 626791_2022-08-15 --session 746439_2025-01-27

.venv/bin/python -m pytest tests/test_preference_context.py \
  tests/test_responsive_preferences.py tests/test_context_preferences.py
```

Default inputs are `results/modality_context/full_expanded` and
`results/responsive_preferences/full`. Inputs must be complete and compatible.
The CLI saves hashes of source code and input data, a source snapshot, package
versions, parameters, per-session caches, and status. Identical runs resume;
changed code, inputs, or parameters require a new output directory. Failures
stop with an explicit failed status. `caffeinate -i` keeps the computer awake.

Implementation files:

- `statistics.py`: CR2 and block-pair estimators, class definitions, matching.
- `analysis.py`: disjoint-trial analysis, FDR, session and mouse comparisons.
- `plots.py`: area figures, overview figures, HTML index, numerical report.
- `__main__.py`: execution, caching, provenance, and status.

Output files:

- `index.html`, `figures/`: area-specific six-panel figures and overviews of
  both detection probabilities and continuous context strength.
- `REPORT.md`: primary expert comparison tables and interpretation notes.
- `selected_neurons.parquet`: trial counts, effects, p/q values, class labels,
  baseline degrees of freedom, pair coefficients, and scales for each orientation.
- `class_counts.csv`: counts including both/neither and detected context cells.
- `comparison_support.csv`: per-session group sizes, matched pairs, eligibility.
- `matched_pairs.csv`: matched unit IDs, distances, and remaining feature imbalance.
- `session_classes.csv`: descriptive class fractions and context strengths.
- `session_comparisons.csv`, `mouse_estimates.csv`, `area_summary.csv`: the actual
  paired effects, counts, intervals, p/q, and mouse/session support.

Do not count rows repeated across windows/orientations as distinct neurons.
Tests verify the CR2 formula against an analytically tractable block-mean test,
block-shock null calibration, absence of baseline-outcome leakage into preferences,
drift rejection, recovery of simulated associations, overlap handling, and matching.
