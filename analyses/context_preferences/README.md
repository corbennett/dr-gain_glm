# Modality, baseline context, target preference, and spatial organization

The completed expanded dataset is described in the
[initial results](RESULTS_2026-10-22.md).

This analysis addresses the five questions using trial-level spikes from the
Dynamic Routing datacube. It does not require a cell to pass a GLM significance
or fit-quality threshold. All `default_qc` neurons with anatomical assignments
are counted when choosing areas, before their preferences are examined.
The default area threshold is **1,000 neurons across the selected sessions and
cohorts combined**. Comparisons use that common set of areas; the naive cohort
does not separately need 1,000 neurons per area. At the user's request, **MRN,
SCm, ACAd, ACAv, and ORBvl are included even below this threshold**. SCm pools
**SCiw, SCig, and SCdg** before area selection and every downstream calculation;
SCdw and superficial SC subdivisions are not included in this pool. Original
subdivision labels are retained in `structure_original` and `location`. This
also means that SCm spatial tests can detect organization between its component
subdivisions, in addition to organization within them.

The expanded public-cache cohort contains **17 areas and 35,576 QC-passing
neurons**: the original twelve areas meeting the threshold plus ACAd (973),
ACAv (965), ORBvl (956), SCm (905: 436 SCiw + 370 SCig + 99 SCdg), and MRN (599).
Counts are across expert and naive sessions combined; available cells per
session and cohort are reported separately. `--include-areas` replaces the
exception list; supplying it with no values uses the threshold alone.

## Run and resume

From the repository root, using its existing virtual environment:

```bash
caffeinate -i .venv/bin/python -u -m gain_glm.context_preferences \
  --output-dir results/modality_context/full_expanded \
  --metadata-dir results/modality_context/discovery \
  --permutations 1999 --threads 2
```

The process prints progress and updates `status.json` after every session.
Completed session files and spike-count caches are reused on an identical run.
Configuration changes require a different output directory. Interrupted sessions
are recomputed; successful sessions are retained even if another session fails.
Failures produce an `incomplete` status and a nonzero exit code, not silent
success. Keep the computer connected to power; `caffeinate -i` prevents idle
sleep while the process is alive, but does not override closing the lid.

Useful options:

```bash
# Inspect the complete cohort, exclusions, and area selection without spikes.
.venv/bin/python -m gain_glm.context_preferences --discover-only

# Small validation run; area counts still come from the full dataset.
.venv/bin/python -m gain_glm.context_preferences \
  --output-dir results/modality_context/smoke_expanded --permutations 99 \
  --session 746439_2025-01-27 --session 626791_2022-08-15

# Recreate reports using the original configuration and completed session files.
.venv/bin/python -m gain_glm.context_preferences --summarize-only

# Optional additional anatomy, keyed by unique unit_id.
# Columns: structure, location, ccf_ap, ccf_dv, ccf_ml (coordinates in µm).
.venv/bin/python -m gain_glm.context_preferences \
  --anatomy /path/to/annotations.parquet --output-dir results/modality_context/with_anatomy
```

For a detached background run with a recorded PID and persistent `run.log`:

```bash
.venv/bin/python scripts/launch_context_preferences.py
tail -f results/modality_context/full_expanded/run.log
```

The launcher records the launched command's PID in `launch.json`; `status.json`
records the worker's PID and completed sessions. Results
are local files; nothing is uploaded. Data are read anonymously from the public
S3 cache. The default datacube version is pinned to `v0.0.289`.

Dependencies are available through the `dynamic-routing` and `analysis` extras
(`pip install -e '.[dynamic-routing,analysis]'`). Run its synthetic checks with:

```bash
.venv/bin/python -m pytest tests/test_context_preferences.py
```

## Cohorts and eligibility

- **Expert:** the dr-datacube `brainwide` selector with its behavior filter,
  evaluated over the full public cache (`only_in_data_asset=False`). The selector
  requires good performance in both contexts and excludes known issue sessions.
  All eligible non-instruction trials in passing sessions enter the analysis;
  this is a session-level behavior criterion, not a good-block-only analysis.
- **Naive:** dr-datacube's context-naive metadata, without imposing expert
  behavior criteria. The discovery audit also includes naive task/ephys sessions
  without the `ccf` keyword so their missing anatomy remains visible. Known
  issue sessions and Templeton sessions are excluded. For this cache version,
  use `context naive` (with a space): `context_naive` is known to erroneously
  label some expert sessions, as documented in dr-datacube's selector source.
- **Anatomy limitation:** the public cache inspected on 2026-10-22 has only one
  naive session with both QC-passing neurons and area assignments. Additional
  naive recordings have missing anatomical labels. They cannot contribute to
  an area-specific analysis without supplementary annotations. A single naive
  mouse provides a descriptive comparison, not population-level evidence about
  learning. The code supports additional annotated naive sessions automatically.
- **Trials:** exclude instruction, catch/unrecognized stimulus, optogenetic, and
  repeat trials. Include correct and incorrect outcomes. Each available context
  must have at least five trials of every stimulus in each trial half. Sessions
  with one context can contribute sensory preferences but have undefined context
  contrasts. Trial halves are randomized within block × stimulus, with a stable
  per-session seed. All neurons in a session use identical trials and halves.

`session_manifest.csv` and `area_counts.csv` expose all selections. The metadata
snapshot, datacube version, seeds, parameters, and optional anatomy-file hash
are retained. The explicitly defined SCm grouping is applied before selection;
no other smaller areas are merged. `area_counts.csv` records source subdivisions
and the reason for each area's inclusion or exclusion.

## Definitions and tests

Count baseline spikes from **−200 to 0 ms**, primary evoked spikes from
**0 to 100 ms**, and secondary evoked spikes from **100 to 500 ms**. All intervals
are half-open. Subtract the same trial's baseline rate from each evoked rate.
Average stimulus identities and contexts equally, regardless of trial counts.
The late window can include action/reward effects; it is a timing sensitivity
analysis, not isolated sensory encoding. This pipeline does not replace the
behavior-adjusted gain GLM.

Let VT, VN, AT, AN denote the resulting visual target, visual nontarget, auditory
target, and auditory nontarget mean evoked rates:

| Quantity | Definition / positive direction |
|---|---|
| Modality | `(VT + VN − AT − AN) / 2`; higher firing for vision |
| Target | `(VT + AT − VN − AN) / 2`; higher firing for targets |
| Visual target | `VT − VN` |
| Auditory target | `AT − AN` |
| Modality on targets | `VT − AT` |
| Modality on nontargets | `VN − AN` |
| Baseline context | Visual-context coefficient in baseline-rate OLS |

Baseline OLS includes an intercept, linear session time, upcoming stimulus
identity, and previous-trial reward, response, modality, and target status.
History is computed before removing trials. The coefficient is a visual minus
auditory context difference in Hz. An unadjusted, stimulus-balanced baseline
contrast is also provided. Movement and arousal are not fully controlled;
context blocks can still covary with slow state changes. The pipeline neither
defines cells as baseline-significant nor requires a baseline significance test.

### 1. Modality preference beyond chance

The stimulus-label null shuffles visual/auditory assignments within each
**block × target status × trial half**. It preserves target effects, block
composition, half assignment, and simultaneous-cell dependence. Each shuffle is
shared across every neuron in the session.

- Unit modality p-values are two-sided permutation ranks (conservative for
  ties). BH q-values are calculated separately within area × cohort × window.
  Cells remain continuous observations even when their tuning is unresolved.
- The selective fraction uses nominal unit p ≤ .05 and compares the observed
  fraction with fractions under those same joint permutations. It does not use
  a binomial test treating correlated neurons as independent trials.
- **Modality power:** mean product of the modality contrasts from independent
  halves. This estimates reproducible variation without squaring noisy estimates.
- **Generalization:** cross-half product of modality preference on targets and
  nontargets, averaged in both directions. Positive values support a common
  modality preference across the two stimulus identities.
- Products are normalized by each neuron's block-demeaned evoked-rate SD,
  floored at 1 Hz. The scale is invariant to the stimulus-label permutations.
  Signed Hz contrasts and cross-half reliability are also retained.
- Crossvalidated visual and auditory response energies include suppressive as
  well as excitatory responses. They are secondary descriptive fields and are
  never rectified to make noisy energies appear positive.

The design distinguishes functional modality selectivity from arbitrary
cell-to-cell differences. With just two stimuli per modality, it cannot prove
anatomically distinct pathways or generalization to all possible stimuli.

### 2–3. Baseline context versus modality or target preference

Compute Spearman correlations **across cells within each area and session**,
pairing baseline half 0 with sensory half 1 and vice versa, then average the two
correlations. This prevents shared baseline-subtraction noise from creating an
artificial relationship. Split halves are not independent sessions and do not
remove all shared state or drift effects.

Test baseline against modality, pooled target preference, visual target
preference, and auditory target preference separately; the latter two prevent
oppositely directed relationships from cancelling. An additional statistic
uses baseline preference in one half to choose a modality, then measures the
target-preference difference between that modality and the opposite modality
in the other half. Positive `preferred_modality_target_advantage` means stronger
target preference in the baseline-preferred modality. Raw-baseline and
within-probe permutation analyses provide sensitivity checks.

Session-level population statistics require at least 10 cells in that area;
smaller groups still have unit-level estimates but are omitted from correlations
and session-level population inference. The relationship null permutes the paired baseline halves between cells in
the same session/area. The within-probe variant restricts this permutation to
each probe. It tests whether the relationship exceeds what probe membership
alone can explain; consult its excess-over-null value as well as its raw rho.

### 4. Expert versus naive

Use identical definitions, windows, and selected areas for both cohorts.
`cohort_comparison.csv` reports estimates, signs, and differences. A population
comparison and mouse-bootstrap difference CI require at least three mice in
each cohort. No population CI is manufactured for a single naive mouse.
The naive context label is the scheduled reward context; it does not imply the
mouse inferred or used that context.

### 5. Spatial segregation

Within each session and area, form the symmetric union of each neuron's ten
nearest neighbors, discarding edges longer than 500 µm. Distances use all three
CCF coordinates. Require at least 20 cells and five distinct recording sites.

The statistic is a cross-half Moran-type covariance: neighboring neurons'
preferences are multiplied using opposite halves, centered, and normalized by
the geometric mean of the halves' sums of squares. The null shuffles each
neuron's two preference estimates together across recording positions. A
second analysis uses only within-probe edges, centers within each probe, and
shuffles within probes. Spatial tests are one-sided for clustering; negative
effects are retained descriptively rather than called evidence for segregation.

Plots show **individual sessions**, including nonsignificant ones, with
anatomical projections, effect sizes, and multiplicity-corrected q-values.
Pooled spatial estimates average within-session effects; they do not mistake
between-session differences in probe placement for local segregation. The
excess-over-null confidence interval quantifies the aggregate effect's range.
Failure to reject clustering is not proof of anatomical mixing: electrode-site
coordinates and limited probe coverage constrain spatial resolution.

## Aggregation and multiplicity

First average session statistics equally within a mouse, then average mice
equally. Confidence intervals resample mice (2,000 bootstrap draws); cells are
not the population sampling unit. Area permutation p-values are conditional on
the recorded sessions and their specified nulls; mouse intervals address a
different question, uncertainty across animals. Fewer than three mice is
flagged as insufficient for population interpretation, and a single mouse has
no bootstrap CI.

BH correction uses all finite area × cohort × window × metric hypotheses as
one explicit area-summary family. All session tests form a separate family.
Unit modality tests have their own area × cohort × window families. Read the
effect sizes and CIs alongside p/q values, especially for negative results.

## Organization and outputs

Implementation is in `src/gain_glm/context_preferences/`:

- `data.py`: datacube metadata, cohort/area selection, trial filtering, spike counts.
- `statistics.py`: contrasts, shuffles, independent-half associations, spatial tests,
  and mouse-weighted inference; independent of NWB access.
- `analysis.py`: per-session analyses, FDR, aggregation, and cohort comparisons.
- `plots.py`: area figures, individual-session map PDFs, and the HTML report.
- `__main__.py`: CLI, resumable execution, progress, and failure reporting.

Open `results/modality_context/full_expanded/index.html` after the run. Outputs include:

- `unit_preferences.parquet`: all continuous estimates, trial halves, modality p/q,
  coordinates, and metadata; one row per cell and response window.
- `session_statistics.csv` / `.parquet`: effects and p/q for each session/area;
  parquet also retains null distributions for reproducible aggregation.
- `area_summary.csv`: cohort/area estimates, mouse intervals, null effects, counts.
- `cohort_comparison.csv`: expert–naive comparisons with support flags.
- `figures/<area>_<window>.png`: six-panel area summaries.
- `figures/<area>_individual_sessions.pdf`: spatial maps for every supported session.
- `counts/` and `sessions/`: resumable trial counts and session analysis products.
- `configuration.json`, `provenance.json`, `source_snapshot/`, `status.json`, manifests, and `REPORT.md`: reproducibility,
  execution status, exclusions, and interpretation notes.
