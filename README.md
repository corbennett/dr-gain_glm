# gain-glm

`gain-glm` fits bilinear encoding models whose temporal kernels can be scaled
by trial-level variables such as context, value, attention, or arousal.

For predictor \(p\) on trial \(n\),

\[
y(t,n) = b_0 + \sum_p g_{p,n}(X_p * K_p)(t) + \epsilon(t,n)
\]

\[
g_{p,n} = g_{0,p} + \sum_v g_{v,p}V_{v,n}.
\]

The package separates four things that used to be mixed together:

1. `ModelSpec` declares predictors, gains, and useful predictor groups.
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
from gain_glm import Event, Gain, ModelSpec, Signal

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
)
```

`Event("cue", ...)` reads `data.events["cue"]`; `Signal("running", ...)`
reads `data.signals["running"]`; and `Gain("context")` reads
`data.trial_values["context"]`. Use `source="another_name"` when the model
term and input should have different names.

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

prepared = compile_design(model, data, fit_mask=peri_stimulus_mask)
```

Irregular continuous signals are interpolated onto bin centers by default.
Use `align="bin"` on a `Signal` to average samples within bins. Signals can be
centered or z-scored as part of their declaration.

`compile_design` validates all source names and dimensions before doing an
expensive fit. Its output can be reused for every target recorded in the same
session.

## Fit, cross-validate, and request dropouts

```python
from gain_glm import CVConfig, Dropout, FitConfig

result = prepared.evaluate(
    y,
    fit=FitConfig(
        regularizer="ridge",
        max_iter=50,
    ),
    cv=CVConfig(folds=5, seed=0),
    dropouts=(
        Dropout.gain("context"),
        Dropout.group("behavior"),
        Dropout.predictors("running", name="running_only"),
    ),
)

print(result.train_r2)
print(result.cv.r2)
print(result.cv.r2_per_fold)
print(result.dropouts["context"].delta_r2)
print(result.fit.gain_table())
kernel = result.fit.kernel("cue")
```

Every requested dropout uses the same full-model CV folds and full-model fit
within each fold. This avoids recomputing CV, a fit summary, and each ΔR²
separately.

### Reduced-model semantics

The automatic refitting rule is based on what is removed:

- Gain-only dropout: retain the full model's fold-specific kernels and
  intercept, then independently refit the reduced gain coefficients. This asks
  for the incremental gain contribution conditional on the learned response
  shapes.
- Predictor or group dropout: fully refit the reduced bilinear model. Remaining
  kernels, gains, and the intercept may adapt.
- Mixed gain and predictor dropout: fully refit the reduced model.

Use `Dropout.gain("context", refit="full")` to ask the alternative scientific
question in which all remaining parameters adapt after removing a gain.

Each `DropoutResult` records the resolved predictor names, gain names, refit
strategy, reduced R², and ΔR² for every fold.

## Model variants

A model variant changes the full hypothesis; a dropout tests terms nested in a
given full hypothesis. Immutable transforms make variants explicit:

```python
no_behavior = model.without_group("behavior", name="no_behavior")

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
from gain_glm import CVConfig, Dropout
from gain_glm.dynamic_routing import DEFAULT_MODEL, load_session, prepare

session = load_session(nwb_path, dt=0.025)
prepared = prepare(session, DEFAULT_MODEL, fit_window=(-0.5, 1.0))

result = prepared.evaluate(
    y,
    cv=CVConfig(folds=5, seed=0),
    dropouts=(
        Dropout.gain("context"),
        Dropout.group("face"),
    ),
)
```

Available full-model declarations are exposed through `MODELS`:

- `default`
- `no_face`
- `no_hit_long_stim`
- `all_response`

Fit a complete session from the command line:

```bash
gain-glm-fit \
  --nwb-path s3://path/session.nwb \
  --session-id 668755_2023-08-31 \
  --output-dir results \
  --model default \
  --dropout gain:context \
  --dropout group:face
```

Model comparison and SLURM launchers are in `scripts/` and take model names as
arguments instead of requiring source edits.

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
    dynamic_routing.py   NWB adapter, default models, batch fitting
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
