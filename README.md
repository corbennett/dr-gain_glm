# gain-glm

`gain-glm` fits bilinear encoding models whose temporal kernels can be scaled
by trial-level variables such as context, value, attention, or arousal.

For predictor $p$ on trial $n$,

$$
y(t,n) = b_0 + \sum_p g_{p,n}(X_p * K_p)(t) + \epsilon(t,n)
$$

$$
g_{p,n} = g_{0,p} + \sum_v g_{v,p}V_{v,n}.
$$

The package separates four things that used to be mixed together:

1. `ModelSpec` declares the time grid, fitted rows, predictors, gains, groups,
   and default reduced-model comparisons.
2. `ModelData` supplies named arrays on one time grid.
3. `PreparedDesign` caches session-shared convolutional design matrices.
4. `EvaluationResult` contains one fit, trial-held-out CV, and any requested
   reduced-model comparisons.

Model specifications and prepared designs are immutable. Fitting one unit does
not modify anything reused by another unit.

## Installation

Install the numerical core:

```bash
pip install -e .
```

Install the Dynamic Routing/NWB workflow or plotting helpers when needed:

```bash
pip install -e '.[dynamic-routing,plot]'
```

## Define a model

Predictors refer to named data sources. A predictor lists its gains directly;
there is no separate `gain_modulated` flag or target list to keep synchronized.

```python
from gain_glm import Dropout, Event, Gain, ModelSpec, Signal

model = ModelSpec(
    predictors=(
        Event(
            "cue",
            window=(0.0, 1.0),
            n_basis=10,
            gains=("context",),
            groups=("stimulus",),
        ),
        Signal(
            "running",
            window=(-1.0, 1.0),
            n_basis=10,
            normalize="zscore",
            groups=("behavior",),
        ),
    ),
    gains=(Gain("context"),),
    name="cue_gain",
    dt=0.01,
    fit_window=(-0.5, 1.0),
    fit_events=("cue",),
    dropouts=(
        Dropout.gain("context"),
        Dropout.gain_terms("context", "cue", name="cue_context_gain"),
        Dropout.group("behavior"),
        Dropout.predictors("running", name="running_only"),
    ),
)
```

`Event("cue", ...)` reads `data.events["cue"]`; `Signal("running", ...)`
reads `data.signals["running"]`; and `Gain("context")` reads
`data.trial_values["context"]`. Use `source="another_name"` when the model
term and input should have different names.

`dt` fixes the model's sampling interval. When `fit_window` is present,
`fit_events` names the event sources around which that window selects training
and scoring rows. These are separate from each predictor's `window`, which
defines the temporal support of its convolutional kernel.

## Supply data and prepare once

```python
from gain_glm import ModelData, TimedSignal, compile_design

data = ModelData(
    dt=0.01,
    trial_index=trial_index,       # length T, integer trial for every bin
    events={"cue": cue_times},    # times relative to bin zero
    signals={
        "running": TimedSignal(running_values, running_times),
    },
    trial_values={
        "context": context,       # one value per trial
    },
)

prepared = compile_design(model, data)
```

Irregular continuous signals are interpolated onto bin centers by default.
Use `align="bin"` on a `Signal` to average samples within bins. Signals can be
centered or z-scored as part of their declaration. A signal can also declare
`orthogonalize_against="predictor_name"`. During preparation, that resampled
signal is residualized against the named signal before its temporal design
block is constructed. The least-squares projection is estimated on the
prepared fit rows after centering both signals, so the signal's mean is
preserved.

`compile_design` validates that `data.dt` matches `model.dt`, resolves the
model's event-anchored fit window, and validates all source names and dimensions
before doing an expensive fit. Its output can be reused for every target
recorded in the same session. Passing `fit_mask=` explicitly replaces the
model's declared fit window for that prepared design.

## Fit, cross-validate, and evaluate dropouts

```python
from gain_glm import CVConfig, FitConfig

result = prepared.evaluate(
    y,
    fit=FitConfig(
        regularizer="ridge",
        max_iter=100,
    ),
    cv=CVConfig(folds=5, seed=0),
)

print(result.train_r2)
print(result.cv.r2)
print(result.cv.r2_per_fold)
print(result.dropouts["context"].delta_r2)
print(result.fit.gain_table())
kernel = result.fit.kernel("cue")
```

When `dropouts` is omitted, `evaluate()` uses the comparisons declared by the
model. Passing a nonempty `dropouts=` sequence overrides those defaults;
passing `dropouts=()` explicitly disables reduced-model comparisons.

Every requested dropout uses the same full-model CV folds and full-model fit
within each fold. This avoids recomputing CV, a fit summary, and each ΔR²
separately. Gain-only dropouts reuse that fold's selected gain penalty so the
comparison changes only the requested gain terms. Results include separate
convergence diagnostics for the final all-data fit, every full-model CV fold,
and every iterative reduced-model fold; the legacy top-level `converged` field
continues to describe only the final all-data fit.

### Current fitting procedure

The implementation separates preparation, parameter fitting, and held-out
evaluation. The details below describe the current behavior, including the
defaults.

#### Design rows and fitted rows

`compile_design` constructs each predictor block across all `data.n_time` time
bins. An event or continuous signal is shifted over its declared lag window and
projected onto its basis, producing one block $B_p$ with shape
`(n_time, n_basis)`. Trial values are also broadcast across all bins according
to `data.trial_index`. A target is not centered, standardized, or smoothed by
the core fitter; `y` is used as supplied.

The prepared `fit_mask` does not shorten these cached arrays. It selects rows
only when fitting and scoring. Consequently:

- preparation and prediction span the complete time grid;
- the full-data fit uses only selected rows;
- outer-CV training and test sets use only selected rows within their trials;
- `FittedModel.predict()` returns a prediction for every time bin; a target
  history array is also required when the model contains `History`.

Passing `mask=` directly to `fit()` or `evaluate()` replaces the prepared
`fit_mask`; the two masks are not automatically intersected.

For predictor $p$, let

$$
d_p(t) = B_p(t)\beta_p
$$

be its kernel drive. The fitted prediction is

$$
\hat y(t) = b_0
+ \sum_{p\ \mathrm{without\ gains}} d_p(t)
+ \sum_{p\ \mathrm{with\ gains}}
\left[g_{0,p} + \sum_v g_{v,p}V_v(t)\right]d_p(t),
$$

where $V_v(t)$ is the value for the trial containing time bin $t$. Thus, if
a globally convolved event response crosses a trial boundary, its gain at each
row is the gain of the trial assigned to that row.

#### Alternating kernel and gain updates

The bilinear parameters are estimated by alternating linear regressions:

1. Gain offsets are initialized to one and trial-variable gain coefficients to
   zero. This makes every gain-modulated predictor initially enter with unit
   gain.
2. Holding gains fixed, the fitter constructs a kernel design whose columns
   are the predictor basis blocks multiplied by their current time-varying
   gains. It fits all kernel coefficients and an unpenalized intercept.
3. Each gain-modulated reconstructed kernel is divided by its Euclidean norm.
   Its gain offset and all of its trial-variable gain coefficients are
   multiplied by the same norm, preserving the prediction while fixing the
   otherwise arbitrary kernel/gain scale.
4. Holding the normalized kernels and intercept fixed, the fitter computes
   every predictor drive. The intercept and predictors without gains are
   treated as fixed offsets. Gain offsets and trial-variable gain coefficients
   are then fitted together with Ridge regression and no additional intercept.
5. Steps 2–4 repeat until convergence or `max_iter` is reached. Convergence is
   declared after every retained trial-variable gain coefficient changes by no
   more than `tol` for `patience` consecutive iterations. Gain offsets are not
   currently included in this convergence check. A model with no retained
   trial-variable gain coefficients finishes after one alternating update.
6. Finally, gains are refitted once against the final normalized kernels and
   intercept. This final refit reuses the gain alpha selected during the first
   gain update.

`FitConfig.regularizer` controls only the kernel regression:

- `"ridge"` uses an L2 penalty and is the default;
- `"lasso"` uses an L1 penalty and warm-starts later ALS iterations.

The gain regression always uses Ridge. Kernel basis coefficients are penalized
by the selected kernel regularizer; gain offsets and trial-variable gain
coefficients are Ridge-penalized. The overall model intercept is not.

#### Automatic alpha selection

`kernel_alpha` and `gain_alpha` are regularization strengths, not optimization
learning rates. Supplying either value skips automatic selection for that
parameter. When a value is `None`, the fitter searches `FitConfig.alphas`
(25 values from $10^{-3}$ through $10^3$ by default) during the first
corresponding ALS update and then holds the selected value fixed.

`FitConfig.inner_cv_folds` is passed to the scikit-learn alpha selector. It is
separate from the trial-held-out outer CV configured by `CVConfig`:

- With Ridge and `inner_cv_folds=None`, `RidgeCV` uses its efficient
  leave-one-observation-out procedure. With scikit-learn's default scoring,
  alpha is selected by negative mean squared error.
- With Ridge and an integer such as `inner_cv_folds=5`, `RidgeCV` uses ordinary
  row-based K-fold splits. Because the code does not set `scoring`,
  scikit-learn uses R² in this case.
- With Lasso and `inner_cv_folds=None`, `LassoCV` uses its default five
  row-based folds and selects alpha by validation mean squared error. An
  integer changes the number of row-based folds.

Inner alpha selection is therefore not currently trial-aware: bins from one
trial may contribute to both its training and validation rows. The outer model
evaluation described next does keep complete trials together.

#### Full fit and trial-held-out evaluation

`evaluate()` first fits one model to all selected rows and reports its training
R². It then obtains the unique trial IDs, optionally permutes them using
`CVConfig.seed`, and divides the trial IDs into `CVConfig.folds` groups. Every
time bin from a trial belongs to the same outer training or test set. These
outer folds are trial-aware but are not stratified by trial condition.

Each outer training fold runs a new ALS fit and therefore selects its own
regularization strengths when they were not explicitly supplied. R² is
computed on the selected test rows of that fold. `CVResult.r2` is the
unweighted mean of the per-fold R² values.

With a `History` predictor, the history design is generated from the complete
target before rows are selected. `CVConfig(gap_history=True)` removes boundary
rows whose requested history reaches across the outer train/test assignment;
the default is `False`.

### Reduced-model semantics

The automatic refitting rule is based on what is removed:

- Gain-only dropout: retain the full model's fold-specific kernels and
  intercept, then independently refit the reduced gain coefficients. This asks
  for the incremental gain contribution conditional on the learned response
  shapes. If `gain_alpha=None`, each reduced gain model selects its own Ridge
  alpha from that fold's training rows.
- Predictor-specific gain dropout: `Dropout.gain_terms()` removes one gain's
  coefficients only from the selected predictors, then performs the same
  fixed-kernel gain refit. Gain offsets and every unselected gain coefficient
  remain in the reduced model.
- Predictor or group dropout: fully refit the reduced bilinear model. Remaining
  kernels, gains, and the intercept may adapt, and automatically selected
  regularization strengths are chosen independently for that reduced fit.
- Mixed gain and predictor dropout: fully refit the reduced model.

Use `Dropout.gain("context", refit="full")` to ask the alternative scientific
question in which all remaining parameters adapt after removing a gain.

Each `DropoutResult` records the resolved predictor names, gain names, refit
strategy, reduced R², and ΔR² for every fold. It also reports
`reduced_r2_pooled` and `delta_r2_pooled`, computed by pooling all held-out
residuals and normalizing them once against the TSS of all held-out targets.
The full model's corresponding score is `result.cv.r2_pooled` (serialized as
`cv_r2_pooled`). The existing `r2` and `delta_r2` fields remain the unweighted
means of the fold-specific scores.

For every outer fold, reduced models are evaluated on the same test rows as the
full model. The reported per-fold quantity is

$$
\Delta R^2 = R^2_{\mathrm{full}} - R^2_{\mathrm{reduced}}.
$$

## Model variants

A model variant changes the full hypothesis; a dropout tests terms nested in a
given full hypothesis. Immutable transforms preserve the model's declared
dropouts. If a transform removes terms referenced by a dropout, pass the
compatible comparison set explicitly:

```python
no_behavior = model.without_group(
    "behavior",
    name="no_behavior",
    dropouts=tuple(
        dropout
        for dropout in model.dropouts
        if dropout.name not in {"behavior", "running_only"}
    ),
)

long_cue = model.replace(
    "cue",
    window=(0.0, 2.0),
    n_basis=20,
)
```

## Dynamic Routing workflow

The experiment-specific NWB schema and default model library live in one
module:

```python
from gain_glm import CVConfig
from gain_glm.dynamic_routing import DEFAULT_MODEL, load_session, prepare

session = load_session(nwb_path, dt=DEFAULT_MODEL.dt)
prepared = prepare(session, DEFAULT_MODEL)

result = prepared.evaluate(
    y,
    cv=CVConfig(folds=5, seed=0),
)
```

Available full-model declarations are exposed through `MODELS`:

- `default`
- `no_face`
- `no_hit_long_stim`
- `all_response`
- `only_baseline`

The Dynamic Routing `default` model represents each stimulus with separate
early (0–0.1 s) and late (0.1–1 s) kernels, both modulated by context. Its
default dropouts compare the context gain on all early stimulus kernels and on
all late stimulus kernels separately, in addition to the whole-context-gain and
context-baseline comparisons.

Fit a complete session from the command line:

```bash
gain-glm-fit \
  --nwb-path s3://path/session.nwb \
  --session-id 668755_2023-08-31 \
  --output-dir results \
  --model default \
  --dt 0.025 \
  --folds 5 \
  --fold-seed 0
```

Omitting `--dropout` uses the selected model's declared comparisons. Providing
one or more `--dropout` arguments replaces them for that run; `--no-dropouts`
disables them. Omitting `--dt` uses the selected model's declared time-bin
width. Omitting `--fold-seed` leaves trials in trial-ID order, while supplying
an integer reproducibly randomizes whole trials among folds.

Model comparison and SLURM launchers are in `scripts/` and take model names as
arguments instead of requiring source edits. The SLURM launcher forwards the
same time-grid and outer-CV options to every session job:

```bash
python scripts/submit_slurm.py \
  --model default \
  --dt 0.025 \
  --folds 5 \
  --fold-seed 0 \
  --dry-run
```

Remove `--dry-run` after inspecting the generated commands to submit the jobs.

## Spike history

History is an explicit predictor rather than a constructor option:

```python
from gain_glm import History

model_with_history = model.add(
    History(window=(0.01, 1.0), n_basis=10),
)
```

Use `CVConfig(gap_history=True)` to remove fold-boundary rows whose history
would include target values from the other side of a train/test boundary.

## Repository layout

```text
src/gain_glm/
    model.py             public specifications and fitted results
    data.py              named inputs and time-grid helpers
    design.py            validation and design compilation
    _solver.py           private ALS implementation
    evaluation.py        CV and reduced-model comparisons
    dynamic_routing.py   NWB adapter and default model declarations
    batch.py             multi-unit fitting, comparison, and CLI
    plotting.py          optional plotting helpers

scripts/                 comparison and cluster entry points
examples/                small runnable examples
tests/                   numerical and API tests
```

Run the synthetic example and tests from a checkout:

```bash
python examples/synthetic.py
python -m unittest discover -s tests -v
```
