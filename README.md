# gain_glm

A Python implementation of a **bilinear encoding GLM with trial-by-trial gain
modulation**, generalising the model in Pan-Vazquez et al.
(*bioRxiv 2025.11.04.685995*).

The model fits a neural (or behavioural) time series $y(t)$ as a sum of
convolutional kernels, where each kernel can be scaled trial-by-trial by an
arbitrary number of latent "gain" variables (e.g. value, attention, arousal):

$$
y(t, n) = b_0
        + \sum_{p \in P_g} g_{p,n}\,(X_p * K_p)(t)
        + \sum_{p \in P_o} (X_p * K_p)(t)
        + \varepsilon(t, n)
$$

$$
g_{p,n} = \beta^g_{0,p} + \sum_v \beta^g_{v,p}\, V_{v,n}
$$

where $P_g$ are the gain-modulated predictors, $P_o$ are the plain linear
ones, and $*$ is convolution.

- $X_p$: 1D predictor (event impulses or a continuous signal)
- $K_p$: temporal kernel for predictor $p$, parameterised in a small basis
- $g_{p,n}$: scalar gain on trial $n$ for predictor $p$
- $V_{v,n}$: trial-by-trial gain variable (any number per predictor)

The unknowns — kernels ($K_p$) and gains ($g_{p,n}$) — are **bilinear** in
$y$ (linear in either set given the other), so the fit alternates between
two penalised regressions (Park & Pillow 2013; Runyan et al. 2017;
Pan-Vazquez et al. 2025).

---

## Contents

- [Why bilinear?](#why-bilinear)
- [Quick start](#quick-start)
- [Conceptual walkthrough](#conceptual-walkthrough)
- [Module reference](#module-reference)
- [The ALS fit loop in detail](#the-als-fit-loop-in-detail)
- [Statistical tools](#statistical-tools)
- [File layout](#file-layout)

---

## Why bilinear?

A standard encoding GLM lets each predictor contribute a fixed kernel:
the impulse response to a cue is whatever the average response is. But many
phenomena — value coding, attention, satiety, learning — are best described
as **multiplicative gains on a stable kernel shape**: the timing and shape of
the response are stereotyped, but its amplitude depends on trial variables.

The bilinear GLM separates "what the response looks like" ($K_p$, fit
across all trials) from "how big it is on this trial" ($g_{p,n}$, fit per
predictor per trial). This dramatically reduces the parameter count
compared to fitting a separate kernel per trial, and gives directly
interpretable trial-by-trial gain coefficients.

---

## Quick start

```python
import numpy as np
from bilinear_glm import (
    BilinearGLM, EventPredictor, ContinuousPredictor, GainModulator,
)

dt = 0.01                                # 10 ms bins
y          = ...                         # shape (T,)  signal to model
trial_idx  = ...                         # shape (T,)  trial index per bin
cue_times  = ...                         # shape (n_events,)  in seconds
speed      = ...                         # shape (T,)  continuous regressor
V1         = ...                         # shape (n_trials,)  trial values

model = BilinearGLM(
    predictors=[
        EventPredictor("cue", cue_times, window=(0.0, 1.0),
                       n_basis=10, basis="cosine", gain_modulated=True),
        ContinuousPredictor("speed", speed, window=(0.0, 0.5),
                            n_basis=8,  basis="cosine"),
    ],
    gains=[GainModulator("V1", V1, modulates=["cue"])],
    dt=dt,
    kernel_regularizer="ridge",
)
model.fit(y, trial_idx, verbose=True)

print("R² :", model.score(y, trial_idx))
print("CV R²:", model.cross_val_score(y, trial_idx, n_folds=5))
print(model.gain_table())
K = model.kernel("cue")                  # time-domain kernel for the cue
```

See [example.py](example.py) for a full end-to-end demo with synthetic data,
ground-truth recovery, and ΔR² model comparisons.

---

## Conceptual walkthrough

The model lives in three roughly orthogonal pieces:

1. **Predictors** specify *what* to convolve. Each predictor `p` carries a
   1D signal — either an event impulse train (`EventPredictor`) or a dense
   signal (`ContinuousPredictor`) — together with a temporal `window`, a
   basis family, and a flag for whether it is gain-modulated. Predictors
   are the columns of the design matrix.

2. **Gains** specify *how* trial-by-trial scaling enters. A `GainModulator`
   carries a length-`n_trials` value array and a list of predictor names it
   modulates. Multiple gains can target the same predictor and a single
   gain can target many predictors — the relationship is many-to-many.
   Per-predictor gain coefficients (offset + one slope per modulator) make
   up the second parameter block of the bilinear model.

3. **The bilinear model itself** holds the predictor/gain specs, builds
   design matrices on demand, runs the alternating fit, and exposes
   accessors for kernels, gains, R², and significance tests.

### Convolutional design matrices

For each predictor $p$ we want the linear contribution at time $t$ to be

$$
D_p(t) = (X_p * K_p)(t) = \sum_{\tau \in \text{lags}} K_p(\tau)\, X_p(t - \tau)
$$

We parameterise the kernel in a small basis ($n_{\text{basis}} \approx 8\text{--}10$
raised-cosine bumps) so that

$$
K_p(\tau) = \sum_j \beta_{p,j}\, b_j(\tau)
\qquad\Longleftrightarrow\qquad
K_p = B_p\, \beta_p
$$

with $B_p$ of shape $(n_{\text{lag}}, n_{\text{basis}})$. Substituting,

$$
D_p(t) = \sum_j \beta_{p,j}\, (X_p * b_j)(t)
$$

So the per-predictor design block is $(T, n_{\text{basis}})$: each column
is the predictor convolved with one basis bump. Fitting $\beta_p$ is a
regression over $n_{\text{basis}}$ parameters per predictor — far fewer
than the $n_{\text{lag}}$-long kernel itself, which gives implicit
smoothness.

### Bilinearity and ALS

Once we add gain modulation, the model is

$$
y(t) \approx b_0 + \sum_p g_p(t)\, (X_p * K_p)(t)
$$

where $g_p(t)$ is the gain expansion evaluated at the trial of bin $t$.
This is bilinear: linear in $\beta_p$ if gains are fixed (gains just
rescale design columns), and linear in the gain coefficients if kernels
are fixed ($D_p(t)$ becomes a known regressor that we multiply by
$V_v(t)$). So we **alternate**:

- **Step A**: hold gains fixed; fit kernels (lasso or ridge).
- **Step B**: rescale each gain-modulated kernel to unit norm and absorb
  the scale into the gain offset (gauge fix — the bilinear product is
  invariant under $(\beta_p, g_p) \to (c\,\beta_p,\, g_p/c)$, so we pin a
  scale).
- **Step C**: hold kernels fixed; fit gain coefficients (ridge).
- Repeat until gain coefficients stabilise.

---

## Module reference

Everything lives in [bilinear_glm.py](bilinear_glm.py). The pieces, in
roughly the order you would encounter them:

### Basis functions

| Function | What it does |
|---|---|
| [linear_cosine_basis](bilinear_glm.py#L38) | `(n_lag_bins, n_basis)` raised-cosine basis evenly spaced across the lag window. Mirrors the Pillow lab `raisedCosineBasis` MATLAB code. Each bump is a half-period cosine of width `2 × center spacing`. |
| [log_cosine_basis](bilinear_glm.py#L62) | Same idea but bumps are uniform in `log(t + offset)` — peaks bunch near `t=0`, used for spike-history kernels (Pillow et al. 2008) where fast dynamics dominate. |
| [identity_basis](bilinear_glm.py#L87) | `I_{n_lag_bins}` — no basis projection, one parameter per lag. Useful for very short windows or as a no-regularisation control. |
| [make_basis](bilinear_glm.py#L91) | String-dispatch wrapper picking one of the above by name (`"cosine"`, `"log_cosine"`, `"identity"`). |

### Predictor & gain specs (dataclasses)

| Class | Purpose |
|---|---|
| [EventPredictor](bilinear_glm.py#L106) | Sparse regressor — event times in seconds, kernel window, basis size, gain flag. The corresponding 1D series is an impulse train at the event bins. |
| [ContinuousPredictor](bilinear_glm.py#L118) | Dense regressor — a length-`T` signal sampled at `dt`. Used for things like running speed or, with `values=y`, a spike-history term. |
| [GainModulator](bilinear_glm.py#L131) | Length-`n_trials` array of trial values plus a list of predictor names it scales. `modulates=None` means "scale every gain-modulated predictor". |

### Design-matrix helpers

| Function | What it does |
|---|---|
| [_lags_in_samples](bilinear_glm.py#L145) | Converts a `(t_low, t_high)` seconds window into the integer lag axis used for convolution. Supports acausal lags (`t_low < 0`). |
| [_event_series](bilinear_glm.py#L151) | Bins event times onto the `y` grid: `1` (or `k` for `k` coincident events) at each event bin, `0` elsewhere. |
| [_design_block](bilinear_glm.py#L162) | Builds the $(T, n_{\text{basis}})$ design block for one predictor. Internally forms the lag matrix $L_{t,i} = X_p(t - \text{lags}[i])$ and right-multiplies by $B_p$. Equivalent (and could be implemented as) $n_{\text{basis}}$ 1D convolutions of the series with each basis bump. |

### The `BilinearGLM` class

`BilinearGLM` ([bilinear_glm.py:187](bilinear_glm.py#L187)) wires the above
together. Methods, grouped by role:

#### Construction & bookkeeping

| Method | What it does |
|---|---|
| [__init__](bilinear_glm.py#L202) | Stores predictors/gains, validates that every gain targets known gain-modulated predictors, precomputes the lag arrays and basis matrices, and lays out the coefficient vectors. Sets up two index maps: `_beta_off` for slicing $\beta$ (one segment per predictor) and `_gain_offset_idx` / `_gain_var_idx` for slicing the gain vector (one offset per gain-modulated predictor, plus one slope per $(v, p)$ pair). |

#### Design construction

These run every fit / predict call to build the per-bin design.

| Method | What it does |
|---|---|
| [_input_series](bilinear_glm.py#L282) | Returns `{pred_name: 1D series of length T}`. Event predictors are binned; continuous predictors are passed through. Accepts an `overrides` dict to swap in alternative inputs (used by `predict` for spike-history terms or pseudosession analyses). |
| [_design_blocks](bilinear_glm.py#L306) | Wraps `_design_block` over all predictors. Returns `{pred_name: (T, n_basis) block}`. |
| [_gain_per_t](bilinear_glm.py#L313) | Materialises each gain modulator from per-trial values to per-bin values, by indexing $V_v[\text{trial\_idx}[t]]$. Returns `{gain_name: (T,) values}`. |

#### Per-iteration helpers

| Method | What it does |
|---|---|
| [_build_kernel_design](bilinear_glm.py#L326) | Builds the **kernel-step** design matrix of shape $(T, n_{\beta})$. For non-modulated predictors, the design block is copied as-is. For modulated predictors, it is multiplied row-wise by $w_p(t) = \beta^g_{0,p} + \sum_v \beta^g_{v,p}\, V_v(t)$ using the current gain coefficients. The fit then learns $\beta$ against this gain-weighted design. |
| [_kernel_drives](bilinear_glm.py#L344) | Computes $D_p(t) = (\text{block}_p\, \beta_p)(t)$ for every predictor — the "kernel response" before any gain scaling. Used both to evaluate the model and to build the gain-step design. |
| [_normalise_kernels](bilinear_glm.py#L352) | After the kernel step, rescales each gain-modulated $\beta_p$ so the time-domain kernel $K_p = B_p\, \beta_p$ has unit $L_2$ norm, and multiplies all of that predictor's gain coefficients by the same factor. This is a gauge fix: the model is invariant under $(\beta_p, g_p) \to (\beta_p/c,\, c\, g_p)$, so we pin $\lVert K_p \rVert = 1$ and let $g_p$ carry the amplitude. Without this, kernels and gains can drift to compensating extremes and the gain values become uninterpretable. |
| [_build_gain_design](bilinear_glm.py#L372) | Builds the **gain-step** design matrix of shape $(T, n_g)$. Each gain-modulated predictor $p$ contributes one column $D_p(t)$ (which multiplies $\beta^g_{0,p}$) and one column $V_v(t)\, D_p(t)$ per modulator $v$ (which multiplies $\beta^g_{v,p}$). The intercept and non-modulated drives have already been subtracted from the target before fitting this design. |
| [_initial_gain_vec](bilinear_glm.py#L391) | Returns the "identity gain" starting point: every offset $= 1$, every modulator slope $= 0$. This makes the first kernel step a standard linear GLM, so the ALS loop is well-conditioned from the start. |

#### Fit / predict / score

| Method | What it does |
|---|---|
| [_fit_als](bilinear_glm.py#L400) | Core ALS loop. Operates on pre-built design blocks and gain arrays so it can be called with any subset of time bins (used by `cross_val_score` for the train fold). Returns `(beta, gain_vec, b0, history)`. See [the section below](#the-als-fit-loop-in-detail). |
| [fit](bilinear_glm.py#L506) | Public fit entry. Builds the full-data design once and calls `_fit_als`. Populates `self.beta_`, `self.gain_`, `self.intercept_`, `self.history_`. |
| [cross_val_score](bilinear_glm.py#L530) | Trial-held-out k-fold CV. Partitions trials into `n_folds` contiguous groups; for each fold, refits ALS on training rows only (but builds the design over the full session so convolutions that span trial boundaries are handled correctly) and scores on test rows. Returns mean R²; per-fold scores in `self.cv_scores_`. Does **not** update fitted attributes. |
| [_predict_from_state](bilinear_glm.py#L597) | Given pre-built blocks, gains, and a $(\beta, g, b_0)$ triple, returns the model's predicted $(T,)$ time course. Sums intercept + each non-modulated drive + each modulated drive weighted by $w_p(t)$. Used by both `predict` and the CV path. |
| [predict](bilinear_glm.py#L612) | Public predict. Builds blocks / gains for the requested `(y, trial_idx)` and calls `_predict_from_state` with the fitted parameters. Accepts `event_overrides`, `continuous_overrides`, and `gain_overrides` to swap in alternative inputs without refitting — useful for spike-history simulation or pseudosession-style null distributions. |
| [score](bilinear_glm.py#L638) | R² of the fitted model. |

#### Accessors

| Method | What it returns |
|---|---|
| [kernel(name)](bilinear_glm.py#L647) | The time-domain kernel $K_p(\tau) = B_p\, \beta_p$, shape $(n_{\text{lag}},)$. |
| [lags(name)](bilinear_glm.py#L653) | The integer lag axis for predictor $p$. |
| [lags_seconds(name)](bilinear_glm.py#L657) | Same axis in seconds. |
| [gain_offset(name)](bilinear_glm.py#L660) | Scalar gain offset $\beta^g_{0,p}$. |
| [gain_coefficient(gain_name, pred_name)](bilinear_glm.py#L663) | Scalar slope $\beta^g_{v,p}$. |
| [gain_table()](bilinear_glm.py#L666) | Nested `{pred: {"offset": ..., gain_var: ...}}` for inspection. |

#### Statistical comparisons

| Method | What it does |
|---|---|
| [delta_r2](bilinear_glm.py#L678) | ΔR² between the full fitted model and a reduced model that zeros out specified gain slopes and/or whole predictor blocks. Kernels and offsets are **not** refit — this matches the paper's "kernels held at the final iteration; only the targeted gains are removed" comparison. |
| [pseudosession_test](bilinear_glm.py#L718) | Circular-shift permutation test for a single gain variable. Holds kernels and intercept fixed at fitted values; refits only the gain coefficients for the real and shifted gain series; reports the observed ΔR² and the empirical p-value against the null distribution of shifts. |
| [_delta_r2_remove](bilinear_glm.py#L771) | Internal helper used by `pseudosession_test`. Refits gain coefficients on a column-removed copy of the gain design and returns the resulting ΔR². |

---

## The ALS fit loop in detail

The body of [`_fit_als`](bilinear_glm.py#L400) is short but does a lot of
work per iteration. One iteration:

**1. Kernel step (gains fixed).** Build $X = $ `_build_kernel_design(blocks, gain_t, gain_vec)`,
shape $(T, n_\beta)$. Each modulated block's columns have been scaled by
the current per-bin gain $w_p(t)$, so fitting $\beta$ against $X$
recovers kernels assuming the gains are correct. Fit with either:

- `kernel_regularizer="lasso"` — `LassoCV` (or `Lasso` if `kernel_alpha` is
  fixed) for sparsity in basis coefficients. Required if you want sparse
  kernels.
- `kernel_regularizer="ridge"` — `RidgeCV` (efficient SVD-based GCV when
  `cv_folds=None`, else k-fold) for smooth fits. Much faster than lasso.

The intercept $b_0$ is fit alongside.

**2. Gauge fix.** `_normalise_kernels` rescales $\beta$ so every modulated
kernel has unit $L_2$ norm, absorbing the discarded scale into the matching
gain coefficients. This pins the parameterisation so the gain values are
the unique scale, not a $\beta$/gain product that could drift.

**3. Gain step (kernels fixed).** Compute kernel drives $D_p(t)$ for every
predictor. Subtract the intercept and the sum of non-modulated drives from
$y$ to form the target:

$$
\text{target}(t) = y(t) - b_0 - \sum_{p \in P_o} D_p(t)
$$

Build the gain design $Z$ (one column per gain coefficient — $D_p$ for
the offset, $V_v\, D_p$ for each slope). Fit the gain vector by ridge
regression against the target, no intercept (the intercept is already in
$\text{offset}(t)$).

**4. Convergence check.** Track the change in the modulator slopes across
iterations (the offsets are intentionally allowed to drift with the gauge
fix). If $\max |\Delta \beta^g_{v,p}| \le \text{tol}$ for `patience`
consecutive iterations, stop.

The loop builds an MSE / alpha history in `self.history_` so you can verify
convergence post-hoc.

### Alpha selection

`kernel_alpha` and `gain_alpha` can each be:
- `None` (default) — picked by CV from `self.alphas` (a log-spaced grid).
  For ridge with `cv_folds=None`, this is closed-form leave-one-out via
  SVD — essentially free. For lasso, k-fold is required and slower.
- A fixed float — skips CV entirely; useful for inner CV loops where the
  outer loop has already picked alphas.

Once chosen on the first iteration, the same alphas are reused for
subsequent ALS iterations — re-running CV every iteration would be
wasteful and unstable.

---

## Statistical tools

Two ways to test whether a gain variable matters:

**`delta_r2`** is a simple drop-one comparison: how much $R^2$ does the
full model lose if we zero out the slopes for gain $v$ (or the whole
block of predictor $p$)? Cheap, deterministic, no refit.

$$
\Delta R^2 = R^2_{\text{full}} - R^2_{\text{reduced}}
$$

**`pseudosession_test`** is the paper's circular-shift permutation test:
randomly rotate the trial-value array, refit only the gain coefficients,
record the $\Delta R^2$, repeat. Compare the observed $\Delta R^2$ to
the null distribution. Slower (one ridge fit per permutation) but
accounts for trial-temporal autocorrelation in the gain variable.

For nested model comparisons that *refit kernels* under the null
(stronger but much slower), refit a new `BilinearGLM` with the relevant
gains/predictors removed and compare CV $R^2$ values.

---

## File layout

```
gain_glm/
├── bilinear_glm.py    # the model
├── example.py         # synthetic-data demo with ground-truth recovery
├── main.py            # placeholder entry point
├── pyproject.toml     # uv / pip project metadata
└── README.md          # this file
```

Run the demo:

```sh
uv run example.py
```

It builds a 250-trial synthetic session with a cue event, an action event,
running speed, and two trial-value variables, fits the model, and prints
recovered vs. true gains and kernel cosine similarities.

---

## References

- Pan-Vazquez et al., *bioRxiv 2025.11.04.685995v1* — the bilinear gain model
  this code generalises.
- Park & Pillow, *PLOS Computational Biology* 2013 — bilinear GLM with ALS
  fitting.
- Runyan et al., *Nature* 2017 — gain-modulated encoding in cortex.
- Pillow et al., *Nature* 2008 — raised-cosine and log-cosine bases for
  spike-history kernels.
