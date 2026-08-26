"""Bilinear encoding model with trial-by-trial gain modulation.

Generalisation of the bilinear GLM in Pan-Vazquez et al. (bioRxiv 2025.11.04.685995):

    y(t, n) = b0
            + sum_{p in P_g}  g_{p,n} * (X_p * K_p)(t)
            + sum_{p in P_o}            (X_p * K_p)(t)
            + eps(t, n)

    g_{p,n} = beta^g_{0,p} + sum_v beta^g_{v,p} * V_{v,n}

P_g are gain-modulated predictors; P_o are linear (no gain). Each kernel K_p is
parameterised in a temporal basis B_p:  K_p(tau) = B_p[tau, :] @ beta_p.
Predictors can be event-based (sparse impulses) or continuous (dense signal);
both reduce to a 1D series convolved with the basis. A spike-history term is
just a continuous predictor whose series is y itself.

Fit by alternating ridge / lasso regression (Park & Pillow 2013; Runyan et al.
2017; Pan-Vazquez et al. 2025): fix gains, fit kernels with L1; normalise each
gain-modulated kernel to unit L2 norm so its amplitude is captured by its gain
offset; fix kernels, fit gains with L2, with the intercept and non-modulated
predictors entering as an offset. Iterate until gains stop moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def linear_cosine_basis(n_basis: int, n_lag_bins: int) -> np.ndarray:
    """Linearly spaced raised-cosine basis. Shape (n_lag_bins, n_basis).

    Mirrors the Pillow lab raisedCosineBasis MATLAB package: bumps span the
    full window, each is a half-period cosine of width 2*spacing, peaks
    uniformly spaced.
    """
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    if n_basis == 1:
        return np.ones((n_lag_bins, 1))
    centers = np.linspace(0.0, n_lag_bins - 1, n_basis)
    spacing = centers[1] - centers[0]
    width = 2.0 * spacing
    t = np.arange(n_lag_bins, dtype=float)
    B = np.zeros((n_lag_bins, n_basis))
    for j, c in enumerate(centers):
        x = (t - c) * np.pi / width
        bump = (np.cos(np.clip(x, -np.pi, np.pi)) + 1.0) / 2.0
        bump[np.abs(x) > np.pi] = 0.0
        B[:, j] = bump
    return B


def log_cosine_basis(n_basis: int, n_lag_bins: int,
                     log_offset: float = 1.0) -> np.ndarray:
    """Log-time raised-cosine basis (peaks bunched near t=0).

    Used for spike history kernels in Pillow et al. (2008) and in the present
    paper for the 10-cosine, 10ms-1s history term.
    """
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    if n_basis == 1:
        return np.ones((n_lag_bins, 1))
    t = np.arange(n_lag_bins, dtype=float)
    nlin = np.log(t + log_offset)
    centers = np.linspace(nlin[0], nlin[-1], n_basis)
    spacing = centers[1] - centers[0]
    width = 2.0 * spacing
    B = np.zeros((n_lag_bins, n_basis))
    for j, c in enumerate(centers):
        x = (nlin - c) * np.pi / width
        bump = (np.cos(np.clip(x, -np.pi, np.pi)) + 1.0) / 2.0
        bump[np.abs(x) > np.pi] = 0.0
        B[:, j] = bump
    return B


def identity_basis(n_lag_bins: int) -> np.ndarray:
    return np.eye(n_lag_bins)


def make_basis(name: str, n_basis: int, n_lag_bins: int) -> np.ndarray:
    if name == "cosine":
        return linear_cosine_basis(n_basis, n_lag_bins)
    if name == "log_cosine":
        return log_cosine_basis(n_basis, n_lag_bins)
    if name == "identity":
        return identity_basis(n_lag_bins)
    raise ValueError(f"unknown basis {name!r}")


# ---------------------------------------------------------------------------
# Predictor & gain specs
# ---------------------------------------------------------------------------

@dataclass
class EventPredictor:
    """Sparse event regressor: response is impulses at event times convolved
    with a temporal kernel."""
    name: str
    times: np.ndarray            # event times in seconds (absolute, in y's clock)
    window: tuple                # (lag_start_sec, lag_end_sec) relative to event
    n_basis: int = 8
    basis: str = "cosine"
    gain_modulated: bool = True


@dataclass
class ContinuousPredictor:
    """Dense regressor: arbitrary 1D signal convolved with a temporal kernel.

    `values` is the signal. By default it must already be sampled at the y
    bin grid (length T, same `dt`). If `times` is also supplied, the signal
    is treated as (timestamps in seconds, value at each timestamp) and is
    resampled onto the y bin grid at fit/predict time:

    - `align="interp"` (default): linear interpolation at bin centers
      (np.interp). Edge behavior: clamped to the first/last sample.
      Best for smoothly-varying signals (velocity, calcium, BOLD).
    - `align="bin"`: average all samples whose timestamps fall within
      each bin (empty bins → 0). Best for already-discretised count-like
      signals sampled densely.

    `normalize` is applied to the binned series before it enters the
    regression (and before plotting in `plot_fit`). It makes kernels on
    different-unit predictors comparable and gives ridge / lasso penalties a
    consistent meaning across predictors. Computed per call from the
    current bin series — at predict time with new data, the stats are
    re-derived from that new data, so for strict ML hygiene either keep
    `normalize="none"` and z-score the input yourself with stored fit-time
    stats, or only predict on the same session.

    - `"none"` (default): no rescaling.
    - `"center"`: subtract the mean.
    - `"zscore"`: subtract the mean, divide by std (degenerate-std → center).

    `outlier_zscore` rejects samples whose absolute z-score (computed on the
    raw values, ignoring existing NaNs) exceeds the threshold. Rejected
    samples are set to NaN and then either linearly interpolated over (the
    length-T path) or dropped before resampling (the (values, times) path).
    Default 5.0; pass `None` to disable.

    To use as a spike-history term, pass values=y and window=(dt, T_history)
    (the auto `spike_history` constructor option does this for you).
    """
    name: str
    values: np.ndarray           # signal: length-T at dt, or arbitrary length if `times` is set
    window: tuple
    n_basis: int = 8
    basis: str = "cosine"
    gain_modulated: bool = False
    times: Optional[np.ndarray] = None   # optional sample timestamps in seconds (same clock as y)
    align: str = "interp"                # "interp" | "bin" — resampling mode when `times` is set
    normalize: str = "none"              # "none" | "center" | "zscore" — applied to the binned series
    outlier_zscore: Optional[float] = 5.0  # |z| > threshold → NaN before resampling/interp; None to disable


def _is_event(p) -> bool:
    """Predictor-kind check by class name. Robust to module / class reloads
    (`isinstance` compares class identity, which breaks when the module is
    re-executed in an interactive session — old objects keep pointing at the
    previous class object). Comparing by class name survives reloads."""
    return type(p).__name__ == "EventPredictor"


def _is_continuous(p) -> bool:
    return type(p).__name__ == "ContinuousPredictor"


@dataclass
class GainModulator:
    """Trial-by-trial scalar that multiplies one or more gain-modulated kernels.

    `modulates=None` means: every gain-modulated predictor.
    """
    name: str
    values: np.ndarray           # length-n_trials trial-by-trial values
    modulates: Optional[Sequence[str]] = None


@dataclass
class _CrossValidationResult:
    """Internal result shared by all trial-held-out evaluation methods."""

    r2_per_fold: np.ndarray
    delta_r2_per_fold: Optional[np.ndarray]
    n_iter_per_fold: np.ndarray


# ---------------------------------------------------------------------------
# Convolutional design-matrix helpers
# ---------------------------------------------------------------------------

def _lags_in_samples(window, dt) -> np.ndarray:
    lo = int(np.floor(window[0] / dt))
    hi = int(np.ceil(window[1] / dt))
    return np.arange(lo, hi + 1)


def _event_series(times: np.ndarray, dt: float, T: int) -> np.ndarray:
    """1 at the bin containing each event, 0 elsewhere (counts if multiple)."""
    s = np.zeros(T)
    if times is None:
        return s
    bins = np.floor(np.asarray(times, dtype=float) / dt).astype(int)
    bins = bins[(bins >= 0) & (bins < T)]
    np.add.at(s, bins, 1.0)
    return s


def _design_block(series: np.ndarray, lags: np.ndarray,
                  basis: np.ndarray) -> np.ndarray:
    """Build (T, n_basis) block via lag matrix × basis matmul.

    L[t, i] = series[t - lags[i]]  (zero-padded at edges)
    out = L @ basis,  shape (T, n_basis)
    """
    T = series.size
    n_lag_bins = lags.size
    L = np.zeros((T, n_lag_bins))
    for i, lag in enumerate(lags):
        if lag >= 0:
            if lag < T:
                L[lag:, i] = series[: T - lag]
        else:
            absl = -lag
            if absl < T:
                L[: T - absl, i] = series[absl:]
    return L @ basis


class TimeBins(NamedTuple):
    """Container for an epoch's bin grid (returned by `make_time_bins`)."""
    edges: np.ndarray   # (T+1,) bin boundaries: bin t covers [edges[t], edges[t+1])
    centers: np.ndarray # (T,) bin centers: edges[:-1] + dt/2
    T: int              # number of bins = floor((end - start) / dt)
    dt: float           # bin width in seconds
    start: float        # epoch start in seconds (= edges[0])
    end: float          # epoch end requested (may exceed edges[-1] by < dt)


def make_time_bins(start: float, end: float, dt: float) -> TimeBins:
    """Build the bin grid for an epoch covering [start, end) at width `dt`.

    `T = floor((end - start) / dt)`; bin `t` covers
    `[start + t*dt, start + (t+1)*dt)`. If `end - start` isn't an exact
    multiple of `dt`, the trailing partial bin is dropped (consistent with
    `bin_spike_times` and the rest of the codebase, which always use
    `np.floor` for bin counts).

    Parameters
    ----------
    start, end : epoch start and end in seconds
    dt         : bin width in seconds

    Returns
    -------
    TimeBins with `.edges`, `.centers`, `.T`, `.dt`, `.start`, `.end`.
    """
    if end <= start:
        raise ValueError(
            f"end ({end}) must be greater than start ({start})")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    T = int(np.floor((end - start) / dt))
    edges = start + np.arange(T + 1) * dt
    centers = edges[:-1] + dt / 2.0
    return TimeBins(edges=edges, centers=centers, T=T, dt=dt,
                    start=float(start), end=float(end))


def bin_spike_times(spike_times: np.ndarray, start: float, end: float,
                    dt: float, *,
                    smooth_sigma: Optional[float] = None,
                    output: str = "counts") -> np.ndarray:
    """Bin spike times in [start, end) into an array suitable as `y` for
    BilinearGLM.

    The model uses a Gaussian likelihood (ridge/lasso), so for sparse
    low-count spike trains you typically want to either coarsen `dt` or
    smooth with `smooth_sigma` to make residuals more Gaussian. For dense
    or already-continuous signals (calcium ΔF/F, BOLD, LFP) you don't need
    this function — pass the signal as `y` directly.

    Parameters
    ----------
    spike_times  : array of spike times in seconds (same clock as `start`)
    start        : start of the binning window in seconds (becomes bin 0)
    end          : end of the binning window in seconds (exclusive)
    dt           : bin width in seconds
    smooth_sigma : if not None, convolve the binned counts with a Gaussian
        of this standard deviation (in seconds). Edges use 'same' mode so
        the output stays length T.
    output       : "counts" (default) — spike counts per bin.
                   "rate"             — counts / dt (instantaneous rate, Hz).

    Returns
    -------
    y : (T,) array of binned spike counts (or rate, optionally smoothed),
        where T = floor((end - start) / dt). Spike times outside
        [start, end) are dropped.
    """
    if output not in ("counts", "rate"):
        raise ValueError(
            f"output must be 'counts' or 'rate', got {output!r}")
    if end <= start:
        raise ValueError(
            f"end ({end}) must be greater than start ({start})")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    T = int(np.floor((end - start) / dt))
    y = np.zeros(T)
    if spike_times is not None and len(spike_times) > 0:
        rel = np.asarray(spike_times, dtype=float) - start
        bins = np.floor(rel / dt).astype(int)
        bins = bins[(bins >= 0) & (bins < T)]
        np.add.at(y, bins, 1.0)
    if smooth_sigma is not None and smooth_sigma > 0:
        sigma_bins = smooth_sigma / dt
        radius = max(1, int(np.ceil(4.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma_bins) ** 2)
        k = k / k.sum()
        y = np.convolve(y, k, mode="same")
    if output == "rate":
        y = y / dt
    return y


def make_trial_idx(starts: np.ndarray, ends: np.ndarray, dt: float, *,
                   T: Optional[int] = None,
                   fill: Optional[int] = None) -> np.ndarray:
    """Build a (T,) bin-to-trial map from per-trial start / end times.

    Bin `t` covers time `[t*dt, (t+1)*dt)`. A bin is assigned to trial `n`
    if its center falls in `[starts[n], ends[n])`; bin indices for trial `n`
    are `floor(starts[n]/dt) : floor(ends[n]/dt)`.

    Parameters
    ----------
    starts : (n_trials,) trial start times in seconds (same clock as `y`)
    ends   : (n_trials,) trial end times in seconds (exclusive)
    dt     : bin width in seconds
    T      : total number of bins. If None, inferred from `max(ends) / dt`.
    fill   : integer to use for bins not covered by any trial. If None
             (default), an uncovered bin raises ValueError — pass an
             explicit sentinel (e.g. `-1`) if you'll mask those bins out
             before fitting.

    Returns
    -------
    trial_idx : (T,) int array. `trial_idx[t]` is the trial number owning
                bin `t`. Suitable to pass to `BilinearGLM.fit` etc.

    Raises
    ------
    ValueError if trials overlap, or if `fill is None` and bins are
    uncovered.
    """
    starts = np.asarray(starts, dtype=float).ravel()
    ends = np.asarray(ends, dtype=float).ravel()
    if starts.size != ends.size:
        raise ValueError(
            f"starts and ends must have equal length, "
            f"got {starts.size} and {ends.size}")
    if np.any(ends <= starts):
        raise ValueError("each trial must have end > start")

    if T is None:
        T = int(np.floor(ends.max() / dt)) if ends.size else 0

    trial_idx = np.full(T, -1, dtype=int)
    for n in range(starts.size):
        a = max(0, int(np.floor(starts[n] / dt)))
        b = min(T, int(np.floor(ends[n] / dt)))
        if a >= b:
            continue
        overlap = trial_idx[a:b] >= 0
        if overlap.any():
            i = a + int(np.where(overlap)[0][0])
            raise ValueError(
                f"trials {int(trial_idx[i])} and {n} overlap at bin {i} "
                f"(time {i * dt:.3f}s)")
        trial_idx[a:b] = n

    if (trial_idx < 0).any():
        if fill is None:
            n_unc = int((trial_idx < 0).sum())
            first = int(np.where(trial_idx < 0)[0][0])
            raise ValueError(
                f"{n_unc} bins are not covered by any trial "
                f"(first uncovered: bin {first} at {first * dt:.3f}s). "
                f"Pass `fill=<int>` to fill them with a sentinel (you'll "
                f"typically also pass a `fit_mask` to exclude them).")
        trial_idx[trial_idx < 0] = int(fill)

    return trial_idx


def windows_mask(event_times: np.ndarray, T: int, dt: float,
                 window: tuple = (-1.0, 2.0)) -> np.ndarray:
    """Boolean mask of length T marking bins within `window` of each event.

    Use as `fit_mask` in BilinearGLM.fit/score/cross_val_score to restrict the
    regression to peri-event windows while still building the convolutional
    design over the full session — eliminates trial-boundary kernel leakage
    without padding tricks.

    Parameters
    ----------
    event_times : array of event times in seconds (same clock as y)
    T           : total number of bins in the session
    dt          : bin width in seconds
    window      : (lo, hi) in seconds relative to each event (lo may be negative)

    Returns
    -------
    mask : (T,) bool array, True for bins to be fit.
    """
    mask = np.zeros(T, dtype=bool)
    if event_times is None or len(event_times) == 0:
        return mask
    lo = int(np.floor(window[0] / dt))
    hi = int(np.ceil(window[1] / dt))
    bins = np.floor(np.asarray(event_times, dtype=float) / dt).astype(int)
    for t0 in bins:
        a = max(0, t0 + lo)
        b = min(T, t0 + hi + 1)
        if a < b:
            mask[a:b] = True
    return mask


# ---------------------------------------------------------------------------
# Bilinear GLM
# ---------------------------------------------------------------------------

class BilinearGLM:
    """Bilinear encoding model with arbitrary trial-by-trial gain modulators.

    Data layout
    -----------
    `y` is a length-T signal sampled at `dt`. `trial_idx` (passed at fit /
    predict / CV time, **not** at construction) is a length-T integer array
    mapping every bin to a trial number — `trial_idx[t]` says which trial
    owns bin `t`. `GainModulator.values` is length-`n_trials`, one scalar
    per trial; the per-bin gain is `V[trial_idx[t]]`, broadcasting trial
    values to bins. This is the only place trial structure enters the model.

    Parameters
    ----------
    predictors : list of EventPredictor / ContinuousPredictor
    gains      : list of GainModulator
    dt         : time-bin width (seconds)
    kernel_regularizer : "lasso" | "ridge"
    kernel_alpha       : if None, picked by CV from `alphas`
    gain_alpha         : if None, picked by CV from `alphas`
    alphas             : grid for CV alpha selection
    cv_folds           : number of folds for CV alpha selection
    spike_history      : if not None, auto-add a spike-history regressor
        whose values are `y` itself. Pass `True` for sensible defaults
        (window=(dt, 1.0), n_basis=10, basis="log_cosine", non-modulated,
        named "history"), or a dict to override any of those:
        `spike_history={"window": (dt, 0.5), "n_basis": 8}`. The predictor
        is appended to `predictors`, accessible as `model.kernel("history")`
        and friends. At predict-time on new data, pass that data as the
        history series via `continuous_overrides={"history": new_y}` if
        you don't want the fit-time `y` reused.
    """

    def __init__(self,
                 predictors: Sequence,
                 gains: Sequence[GainModulator] = (),
                 *,
                 dt: float = 0.01,
                 kernel_regularizer: str = "lasso",
                 kernel_alpha: Optional[float] = None,
                 gain_alpha: Optional[float] = None,
                 alphas: Optional[np.ndarray] = None,
                 cv_folds: Optional[int] = None,
                 spike_history=None):
        self.dt = float(dt)
        self.predictors = list(predictors)

        # Optionally append an auto-configured spike-history predictor. Its
        # values are filled in from `y` at fit/predict time via the override
        # mechanism, so the placeholder length-1 array here is never used.
        self._history_predictor_name = None
        if spike_history is not None and spike_history is not False:
            cfg = {} if spike_history is True else dict(spike_history)
            h_name           = cfg.pop("name", "history")
            h_window         = cfg.pop("window", (self.dt, 1.0))
            h_n_basis        = cfg.pop("n_basis", 10)
            h_basis          = cfg.pop("basis", "log_cosine")
            h_gain_modulated = cfg.pop("gain_modulated", False)
            if cfg:
                raise ValueError(
                    f"unknown spike_history options: {sorted(cfg)}")
            if h_window[0] < self.dt - 1e-9:
                raise ValueError(
                    f"spike_history window starts at {h_window[0]} s but "
                    f"must be >= dt ({self.dt} s) — otherwise the model "
                    f"would use y(t) to predict y(t)")
            self.predictors.append(ContinuousPredictor(
                name=h_name, values=np.zeros(1),
                window=h_window, n_basis=h_n_basis, basis=h_basis,
                gain_modulated=h_gain_modulated,
                # y itself is the input here — leave its values untouched so
                # legitimate peaks (bursts, transients) aren't clipped.
                outlier_zscore=None,
            ))
            self._history_predictor_name = h_name

        self.gains = list(gains)
        self.kernel_regularizer = kernel_regularizer
        self.kernel_alpha = kernel_alpha
        self.gain_alpha = gain_alpha
        self.alphas = (np.logspace(-5, 5, 21) if alphas is None
                       else np.asarray(alphas, dtype=float))
        # None  → RidgeCV uses efficient SVD-based leave-one-out (GCV) — fast
        # int   → k-fold CV (required for LassoCV; Ridge will use joblib, slow)
        self.cv_folds = cv_folds

        names = [p.name for p in self.predictors]
        if len(set(names)) != len(names):
            raise ValueError("duplicate predictor names")
        self._pred_idx = {p.name: i for i, p in enumerate(self.predictors)}
        for g in self.gains:
            if g.modulates is None:
                continue
            for n in g.modulates:
                if n not in self._pred_idx:
                    raise ValueError(f"gain {g.name!r} modulates unknown predictor {n!r}")
                if not self.predictors[self._pred_idx[n]].gain_modulated:
                    raise ValueError(f"gain {g.name!r} modulates non-gain predictor {n!r}")

        self._lags = {p.name: _lags_in_samples(p.window, self.dt)
                      for p in self.predictors}
        self._basis = {p.name: make_basis(p.basis, p.n_basis,
                                          self._lags[p.name].size)
                       for p in self.predictors}

        # parameter offsets in the kernel coefficient vector
        sizes = [p.n_basis for p in self.predictors]
        self._beta_off = np.cumsum([0] + sizes)
        self._beta_size = int(self._beta_off[-1])

        # gain-related bookkeeping
        self._modulated = [p for p in self.predictors if p.gain_modulated]
        self._mod_names = [p.name for p in self._modulated]
        # which gains modulate predictor p?
        self._gains_for = {p.name: [] for p in self._modulated}
        for g in self.gains:
            targets = (g.modulates if g.modulates is not None
                       else self._mod_names)
            for n in targets:
                self._gains_for[n].append(g.name)

        # gain coefficient vector index:
        # [offset_p1, offset_p2, ..., (gain_v, p_k) pairs in order]
        self._gain_offset_idx = {}
        cnt = 0
        for p in self._modulated:
            self._gain_offset_idx[p.name] = cnt
            cnt += 1
        self._gain_var_idx = {}      # (var_name, pred_name) -> int
        for p in self._modulated:
            for vname in self._gains_for[p.name]:
                self._gain_var_idx[(vname, p.name)] = cnt
                cnt += 1
        self._gain_size = cnt

        # populated by fit()
        self.beta_ = None            # (beta_size,)
        self.beta_history_ = None    # last set of kernel coefs per iter
        self.gain_ = None            # (gain_size,)
        self.intercept_ = 0.0
        self.history_ = None         # list of dicts per iteration (loss, etc.)
        self.fit_mask_ = None        # (T,) bool — bins used in last fit, or None

    # ----- design construction ----------------------------------------------

    def _continuous_to_bins(self, values, T, *,
                              times=None, align: str = "interp",
                              normalize: str = "none",
                              outlier_zscore: Optional[float] = 5.0,
                              ) -> np.ndarray:
        """Resample a continuous signal onto the y bin grid.

        If `times` is None, `values` must already be length T; any NaN
        entries are linearly interpolated over (using surrounding finite
        values; the nearest finite value is held constant at the edges).

        Otherwise `(values, times)` are interpolated (`align="interp"`) or
        averaged per bin (`align="bin"`) onto bins `[t*dt, (t+1)*dt)` for
        t=0..T-1. Sample pairs where either `values` or `times` is NaN are
        dropped before resampling. `times` is sorted defensively (np.interp
        requires it).

        `outlier_zscore`: if set, samples whose |z| (computed on the raw
        finite values) exceeds the threshold are marked NaN before the
        NaN-handling step. None disables.

        `normalize` is applied to the binned series: "none" (default),
        "center" (subtract mean), or "zscore" (subtract mean, divide by std).
        """
        arr = np.asarray(values, dtype=float).ravel()
        arr = self._mark_outliers(arr, outlier_zscore)
        if times is None:
            if arr.size != T:
                raise ValueError(
                    f"continuous values have length {arr.size}, expected "
                    f"T={T} (pass `times=` to resample irregularly-sampled "
                    f"data onto the bin grid)")
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                if nan_mask.all():
                    raise ValueError(
                        "continuous values are all NaN — cannot interpolate")
                idx = np.arange(T)
                arr = np.where(nan_mask,
                               np.interp(idx, idx[~nan_mask], arr[~nan_mask]),
                               arr)
            return self._normalise_continuous(arr, normalize)
        times_arr = np.asarray(times, dtype=float).ravel()
        if times_arr.size != arr.size:
            raise ValueError(
                f"values and times must have the same length, got "
                f"{arr.size} and {times_arr.size}")
        valid = ~(np.isnan(arr) | np.isnan(times_arr))
        if not valid.any():
            raise ValueError(
                "all (values, times) pairs are NaN — cannot resample")
        arr = arr[valid]
        times_arr = times_arr[valid]
        if not np.all(np.diff(times_arr) >= 0):
            order = np.argsort(times_arr)
            arr = arr[order]
            times_arr = times_arr[order]
        if align == "interp":
            bin_centers = (np.arange(T) + 0.5) * self.dt
            out = np.interp(bin_centers, times_arr, arr)
        elif align == "bin":
            bin_idx = np.floor(times_arr / self.dt).astype(int)
            in_range = (bin_idx >= 0) & (bin_idx < T)
            bin_idx = bin_idx[in_range]
            arr_v = arr[in_range]
            sums = np.zeros(T)
            counts = np.zeros(T, dtype=int)
            np.add.at(sums, bin_idx, arr_v)
            np.add.at(counts, bin_idx, 1)
            out = np.zeros(T)
            nz = counts > 0
            out[nz] = sums[nz] / counts[nz]
        else:
            raise ValueError(
                f"unknown align {align!r}, must be 'interp' or 'bin'")
        return self._normalise_continuous(out, normalize)

    @staticmethod
    def _mark_outliers(arr: np.ndarray,
                        threshold: Optional[float],
                        max_passes: int = 5) -> np.ndarray:
        """Return a copy of arr with robust |z| > threshold set to NaN.

        Uses median + MAD (median absolute deviation) instead of mean + std,
        so the outliers themselves can't inflate the scale enough to hide
        each other — a known failure mode of iterated mean/std thresholding
        (e.g., 5% outliers at ±100 in N(0,1) give a std of ~22, making the
        outliers' z only ~4.5, so a threshold of 5 misses them entirely).

        Robust z = (x − median) / (1.4826 · MAD), where the 1.4826 makes
        MAD consistent with std under a normal distribution. Iterates up
        to `max_passes` for stability; with MAD a single pass is usually
        sufficient.
        """
        if threshold is None or threshold <= 0:
            return arr
        out = arr.astype(float, copy=True)
        for _ in range(max_passes):
            finite = np.isfinite(out)
            if int(finite.sum()) < 2:
                break
            finite_vals = out[finite]
            med = float(np.median(finite_vals))
            mad = float(np.median(np.abs(finite_vals - med)))
            if mad >= 1e-12:
                scale = 1.4826 * mad
            else:
                # Degenerate MAD (e.g., > half the values are equal) — fall
                # back to std so we still flag truly extreme points.
                sd = float(np.std(finite_vals))
                if sd < 1e-12:
                    break
                scale = sd
            z = np.abs((out - med) / scale)
            new_outliers = finite & (z > threshold)
            if not new_outliers.any():
                break
            out[new_outliers] = np.nan
        return out

    def _continuous_used_mask(self, values, T, *,
                                times=None, align: str = "interp",
                                outlier_zscore: Optional[float] = 5.0,
                                ) -> np.ndarray:
        """Return a (T,) bool mask: True at bins whose value derives from
        an actual original sample (not from outlier replacement, NaN fill,
        empty-bin zero, or out-of-range edge clamping).

        Useful for plotting: setting `series[~mask] = NaN` hides synthetic
        fill values so only "real" data is shown.
        """
        arr = np.asarray(values, dtype=float).ravel()
        arr = self._mark_outliers(arr, outlier_zscore)
        if times is None:
            if arr.size != T:
                return np.zeros(T, dtype=bool)
            return ~np.isnan(arr)
        times_arr = np.asarray(times, dtype=float).ravel()
        if times_arr.size != arr.size:
            return np.zeros(T, dtype=bool)
        valid = ~(np.isnan(arr) | np.isnan(times_arr))
        if not valid.any():
            return np.zeros(T, dtype=bool)
        valid_times = times_arr[valid]
        if align == "bin":
            bin_idx = np.floor(valid_times / self.dt).astype(int)
            in_range = (bin_idx >= 0) & (bin_idx < T)
            mask = np.zeros(T, dtype=bool)
            mask[bin_idx[in_range]] = True
            return mask
        # align == "interp": bins inside the range of valid samples derive
        # their value from real data; outside, np.interp clamps to the edge
        # which is synthetic.
        bin_centers = (np.arange(T) + 0.5) * self.dt
        t_lo, t_hi = float(valid_times.min()), float(valid_times.max())
        return (bin_centers >= t_lo) & (bin_centers <= t_hi)

    @staticmethod
    def _normalise_continuous(arr: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return arr
        if mode == "center":
            return arr - float(np.mean(arr))
        if mode == "zscore":
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            if sd < 1e-12:
                return arr - mu
            return (arr - mu) / sd
        raise ValueError(
            f"unknown normalize {mode!r}, must be 'none', 'center', or 'zscore'")

    def _input_series(self, T: int, overrides: Optional[dict] = None):
        """Return {pred_name: 1D series of length T} for every predictor.

        Continuous-predictor overrides may be either a length-T array
        (already binned — fed through the predictor's `normalize` step
        only; assumed not to need resampling) or a (values, times) tuple
        that gets resampled and normalised onto the bin grid using the
        predictor's `align` / `normalize` settings.
        """
        out = {}
        ov = overrides or {}
        for p in self.predictors:
            if p.name in ov:
                val = ov[p.name]
                if _is_event(p):
                    arr = np.asarray(val, dtype=float).ravel()
                    out[p.name] = _event_series(arr, self.dt, T)
                else:
                    if isinstance(val, tuple) and len(val) == 2:
                        values, times = val
                        out[p.name] = self._continuous_to_bins(
                            values, T, times=times,
                            align=getattr(p, "align", "interp"),
                            normalize=getattr(p, "normalize", "none"),
                            outlier_zscore=getattr(
                                p, "outlier_zscore", 5.0))
                    else:
                        arr = np.asarray(val, dtype=float).ravel()
                        if arr.size != T:
                            raise ValueError(
                                f"override for {p.name!r} has length "
                                f"{arr.size}, expected {T} (or pass a "
                                f"(values, times) tuple to resample)")
                        # Route through _continuous_to_bins to apply the
                        # predictor's outlier / normalize settings consistently.
                        out[p.name] = self._continuous_to_bins(
                            arr, T,
                            normalize=getattr(p, "normalize", "none"),
                            outlier_zscore=getattr(
                                p, "outlier_zscore", 5.0))
            elif _is_event(p):
                out[p.name] = _event_series(p.times, self.dt, T)
            else:
                out[p.name] = self._continuous_to_bins(
                    p.values, T,
                    times=getattr(p, "times", None),
                    align=getattr(p, "align", "interp"),
                    normalize=getattr(p, "normalize", "none"),
                    outlier_zscore=getattr(p, "outlier_zscore", 5.0))
        return out

    def _design_blocks(self, series: dict) -> dict:
        """{pred_name: (T, n_basis) block}."""
        return {p.name: _design_block(series[p.name],
                                       self._lags[p.name],
                                       self._basis[p.name])
                for p in self.predictors}

    def _with_history(self, overrides, y) -> Optional[dict]:
        """Inject the auto spike-history series into an overrides dict.

        No-op when `spike_history` wasn't requested or the user already
        supplied their own override for the history predictor.
        """
        if self._history_predictor_name is None:
            return overrides
        out = dict(overrides) if overrides else {}
        if self._history_predictor_name not in out:
            out[self._history_predictor_name] = y
        return out

    @staticmethod
    def _safe_under_history(fold_mask: np.ndarray,
                             gap_bins: int) -> np.ndarray:
        """Bool mask: bin t kept iff bins [t-gap_bins, ..., t] all True in fold_mask.

        Used to gap CV folds at history-kernel-length boundaries, so a fold's
        design only uses lagged y from within that fold (no AR leak across
        train/test split).
        """
        T = fold_mask.size
        if gap_bins <= 0:
            return fold_mask.copy()
        safe = fold_mask.copy()
        for lag in range(1, gap_bins + 1):
            if lag >= T:
                safe[:] = False
                return safe
            safe[:lag] = False
            safe[lag:] &= fold_mask[:-lag]
        return safe

    def _gain_per_t(self, T: int, trial_idx: np.ndarray,
                    gain_overrides: Optional[dict] = None) -> dict:
        """{gain_name: (T,) gain value at each time bin}."""
        ov = gain_overrides or {}
        out = {}
        ti = np.asarray(trial_idx, dtype=int)
        for g in self.gains:
            vals = np.asarray(ov.get(g.name, g.values), dtype=float).ravel()
            out[g.name] = vals[ti]
        return out

    # ----- helpers ----------------------------------------------------------

    def _build_kernel_design(self, blocks: dict, gain_t: dict,
                              gain_vec: np.ndarray) -> np.ndarray:
        """Stack (T, beta_size) design weighted by current per-predictor gains."""
        T = next(iter(blocks.values())).shape[0]
        X = np.zeros((T, self._beta_size))
        for k, p in enumerate(self.predictors):
            start, end = self._beta_off[k], self._beta_off[k + 1]
            if p.gain_modulated:
                # g_{p,n}(t) = offset + sum_v beta^g_{v,p} * V_v(t)
                w = np.full(T, gain_vec[self._gain_offset_idx[p.name]])
                for vname in self._gains_for[p.name]:
                    a = gain_vec[self._gain_var_idx[(vname, p.name)]]
                    w = w + a * gain_t[vname]
                X[:, start:end] = blocks[p.name] * w[:, None]
            else:
                X[:, start:end] = blocks[p.name]
        return X

    def _kernel_drives(self, blocks: dict, beta: np.ndarray) -> dict:
        """{pred_name: (T,) drive D_p(t) = blocks[p] @ beta_p}."""
        out = {}
        for k, p in enumerate(self.predictors):
            start, end = self._beta_off[k], self._beta_off[k + 1]
            out[p.name] = blocks[p.name] @ beta[start:end]
        return out

    def _normalise_kernels(self, beta: np.ndarray, gain_vec: np.ndarray):
        """Rescale beta_p so that the time-domain kernel K_p has unit L2 norm
        for every gain-modulated predictor; absorb the scale into its gain
        offset so the model is unchanged."""
        for k, p in enumerate(self.predictors):
            if not p.gain_modulated:
                continue
            start, end = self._beta_off[k], self._beta_off[k + 1]
            kernel = self._basis[p.name] @ beta[start:end]
            nrm = np.linalg.norm(kernel)
            if nrm < 1e-12:
                continue
            beta[start:end] /= nrm
            # Move the kernel's amplitude into the gain offset so
            # g * K stays the same.
            i = self._gain_offset_idx[p.name]
            gain_vec[i] *= nrm
            for vname in self._gains_for[p.name]:
                gain_vec[self._gain_var_idx[(vname, p.name)]] *= nrm

    def _build_gain_design(self, drives: dict, gain_t: dict) -> np.ndarray:
        """Stack (T, gain_size) feature matrix for the gain-fitting step.

        Each gain-modulated predictor p contributes:
            - column D_p(t)                       -> beta^g_{0,p}
            - column V_v(t) * D_p(t) for each v   -> beta^g_{v,p}
        """
        if self._gain_size == 0:
            return None
        T = next(iter(drives.values())).shape[0]
        Z = np.zeros((T, self._gain_size))
        for p in self._modulated:
            i = self._gain_offset_idx[p.name]
            Z[:, i] = drives[p.name]
            for vname in self._gains_for[p.name]:
                j = self._gain_var_idx[(vname, p.name)]
                Z[:, j] = gain_t[vname] * drives[p.name]
        return Z

    def _initial_gain_vec(self) -> np.ndarray:
        """Identity gain: every offset = 1, every value coef = 0."""
        gv = np.zeros(self._gain_size)
        for p in self._modulated:
            gv[self._gain_offset_idx[p.name]] = 1.0
        return gv

    def _gain_keep_mask(self, *,
                        remove_gains: Sequence[str] = (),
                        remove_predictors: Sequence[str] = ()) -> np.ndarray:
        """Gain-vector columns retained by a reduced model.

        Removing a gain drops only its trial-variable slopes; each predictor's
        baseline gain offset remains. Removing a predictor drops its offset and
        every slope attached to it because its entire design block is absent.
        """
        known_gains = {g.name for g in self.gains}
        unknown_gains = sorted(set(remove_gains) - known_gains)
        if unknown_gains:
            raise ValueError(f"unknown gains: {unknown_gains}")

        known_predictors = set(self._pred_idx)
        unknown_predictors = sorted(
            set(remove_predictors) - known_predictors)
        if unknown_predictors:
            raise ValueError(f"unknown predictors: {unknown_predictors}")

        keep = np.ones(self._gain_size, dtype=bool)
        for gain_name in remove_gains:
            for p in self._modulated:
                key = (gain_name, p.name)
                if key in self._gain_var_idx:
                    keep[self._gain_var_idx[key]] = False
        for pred_name in remove_predictors:
            if pred_name not in self._gain_offset_idx:
                continue
            keep[self._gain_offset_idx[pred_name]] = False
            for gain_name in self._gains_for[pred_name]:
                keep[self._gain_var_idx[(gain_name, pred_name)]] = False
        return keep

    def _fit_gain_coefficients(self, Z: Optional[np.ndarray],
                               target: np.ndarray, *,
                               keep_mask: Optional[np.ndarray] = None,
                               alpha: Optional[float] = None
                               ) -> tuple[np.ndarray, Optional[float]]:
        """Fit retained gain columns and expand back to the full gain vector."""
        gain_vec = np.zeros(self._gain_size)
        if Z is None or self._gain_size == 0:
            return gain_vec, None

        if keep_mask is None:
            keep = np.ones(self._gain_size, dtype=bool)
        else:
            keep = np.asarray(keep_mask, dtype=bool).ravel()
            if keep.size != self._gain_size:
                raise ValueError(
                    f"gain keep mask has length {keep.size}, expected "
                    f"{self._gain_size}")
        if not keep.any():
            return gain_vec, None

        Z_fit = Z[:, keep]
        if alpha is None:
            solver = RidgeCV(alphas=self.alphas, cv=self.cv_folds,
                             fit_intercept=False)
            solver.fit(Z_fit, target)
            selected_alpha = float(solver.alpha_)
        else:
            solver = Ridge(alpha=alpha, fit_intercept=False)
            solver.fit(Z_fit, target)
            selected_alpha = float(alpha)
        gain_vec[keep] = solver.coef_
        return gain_vec, selected_alpha

    def _refit_gains(self, y: np.ndarray, blocks: dict, gain_t: dict,
                     beta: np.ndarray, b0: float, *,
                     keep_mask: Optional[np.ndarray] = None,
                     alpha: Optional[float] = None
                     ) -> tuple[np.ndarray, Optional[float]]:
        """Refit gains with kernels and intercept fixed.

        This is the final step of a full fit and the reduced-model operation
        used when testing trial-by-trial gains. If ``alpha`` is None, ridge
        regularization is selected afresh for this particular gain model.
        """
        drives = self._kernel_drives(blocks, beta)
        offset_t = np.full(y.size, b0)
        for p in self.predictors:
            if not p.gain_modulated:
                offset_t = offset_t + drives[p.name]
        target = y - offset_t
        Z = self._build_gain_design(drives, gain_t)
        return self._fit_gain_coefficients(
            Z, target, keep_mask=keep_mask, alpha=alpha)

    @staticmethod
    def _r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0.0:
            return float("nan")
        return 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot

    # ----- fit / predict ----------------------------------------------------

    def _fit_als(self, y: np.ndarray, blocks: dict, gain_t: dict, *,
                 gain_keep_mask: Optional[np.ndarray] = None,
                 max_iter: int = 100, tol: float = 1e-3,
                 patience: int = 3, verbose: bool = False):
        """Core ALS loop operating on pre-built blocks and gain_t arrays.

        Returns (beta, gain_vec, b0, history). Callers slice blocks / gain_t
        to the desired time bins before passing them in.
        """
        T = y.size
        gain_vec = self._initial_gain_vec()
        beta = np.zeros(self._beta_size)
        b0 = 0.0
        history = []

        if gain_keep_mask is None:
            gain_keep = np.ones(self._gain_size, dtype=bool)
        else:
            gain_keep = np.asarray(gain_keep_mask, dtype=bool).ravel()
            if gain_keep.size != self._gain_size:
                raise ValueError(
                    f"gain keep mask has length {gain_keep.size}, expected "
                    f"{self._gain_size}")
            gain_vec[~gain_keep] = 0.0

        kernel_alpha = self.kernel_alpha
        gain_alpha = self.gain_alpha

        # Inner-loop solvers are hoisted out so we can reuse them across
        # iterations. For Lasso, `warm_start=True` lets coordinate descent
        # initialise from the previous iteration's beta — usually 2–3× faster
        # than a cold restart since the kernels change only modestly between
        # ALS sweeps. Ridge has a closed-form solve, so reuse is cosmetic.
        kernel_solver = None

        var_indices = [self._gain_var_idx[(v, p)]
                       for (v, p) in self._gain_var_idx
                       if gain_keep[self._gain_var_idx[(v, p)]]]
        prev_gain_vars = (gain_vec[var_indices].copy()
                          if var_indices else np.zeros(0))
        stable_iters = 0

        for it in range(max_iter):
            # ---- step 1: kernels | gains ------------------------------------
            X = self._build_kernel_design(blocks, gain_t, gain_vec)
            if self.kernel_regularizer == "lasso":
                if kernel_alpha is None:
                    cv = LassoCV(alphas=self.alphas, cv=self.cv_folds,
                                 fit_intercept=True, max_iter=10_000)
                    cv.fit(X, y)
                    kernel_alpha = float(cv.alpha_)
                    beta = cv.coef_.copy()
                    b0 = float(cv.intercept_)
                else:
                    if kernel_solver is None:
                        kernel_solver = Lasso(alpha=kernel_alpha,
                                               fit_intercept=True,
                                               max_iter=10_000,
                                               warm_start=True)
                        # Seed with the previous iter's beta so the first
                        # post-CV fit also benefits from a warm start.
                        kernel_solver.coef_ = beta.copy()
                        kernel_solver.intercept_ = float(b0)
                    kernel_solver.fit(X, y)
                    beta = kernel_solver.coef_.copy()
                    b0 = float(kernel_solver.intercept_)
            elif self.kernel_regularizer == "ridge":
                if kernel_alpha is None:
                    cv = RidgeCV(alphas=self.alphas, cv=self.cv_folds,
                                 fit_intercept=True)
                    cv.fit(X, y)
                    kernel_alpha = float(cv.alpha_)
                    beta = cv.coef_.copy()
                    b0 = float(cv.intercept_)
                else:
                    if kernel_solver is None:
                        kernel_solver = Ridge(alpha=kernel_alpha,
                                               fit_intercept=True)
                    kernel_solver.fit(X, y)
                    beta = kernel_solver.coef_.copy()
                    b0 = float(kernel_solver.intercept_)
            else:
                raise ValueError(f"unknown kernel_regularizer "
                                 f"{self.kernel_regularizer!r}")

            # ---- step 2: normalise kernels & rescale gain offsets ----------
            self._normalise_kernels(beta, gain_vec)
            # Keep the warm-start solver's internal coef in sync with the
            # normalised beta — otherwise the next iter restarts from the
            # un-normalised previous beta and pays back the rescaling.
            if kernel_solver is not None and hasattr(kernel_solver, "coef_"):
                kernel_solver.coef_ = beta.copy()

            # ---- step 3: gains | kernels -----------------------------------
            drives = self._kernel_drives(blocks, beta)
            offset_t = np.full(T, b0)
            for p in self.predictors:
                if not p.gain_modulated:
                    offset_t = offset_t + drives[p.name]

            target = y - offset_t
            Z = self._build_gain_design(drives, gain_t)

            if Z is not None and Z.shape[1] > 0:
                gain_vec, fitted_gain_alpha = self._fit_gain_coefficients(
                    Z, target, keep_mask=gain_keep, alpha=gain_alpha)
                if gain_alpha is None:
                    gain_alpha = fitted_gain_alpha

            # ---- bookkeeping -----------------------------------------------
            yhat = self._predict_from_state(blocks, gain_t, beta, gain_vec, b0)
            mse = float(np.mean((y - yhat) ** 2))
            history.append({"iter": it, "mse": mse,
                            "kernel_alpha": kernel_alpha,
                            "gain_alpha": gain_alpha})
            if verbose:
                gain_alpha_text = ("none" if gain_alpha is None
                                   else f"{gain_alpha:.3g}")
                print(f"iter {it:3d}  mse={mse:.6g}  "
                      f"kernel_alpha={kernel_alpha:.3g}  "
                      f"gain_alpha={gain_alpha_text}")

            if var_indices:
                cur = gain_vec[var_indices]
                delta = np.max(np.abs(cur - prev_gain_vars))
                prev_gain_vars = cur.copy()
                if delta <= tol:
                    stable_iters += 1
                    if stable_iters >= patience:
                        break
                else:
                    stable_iters = 0

        # The alpha selected during ALS belongs to an earlier kernel iterate.
        # Refit the full gain model against the final kernels, selecting its
        # regularization afresh unless the caller fixed ``gain_alpha``.
        gain_vec, final_gain_alpha = self._refit_gains(
            y, blocks, gain_t, beta, b0,
            keep_mask=gain_keep, alpha=self.gain_alpha)
        if history:
            final_yhat = self._predict_from_state(
                blocks, gain_t, beta, gain_vec, b0)
            history[-1]["mse"] = float(np.mean((y - final_yhat) ** 2))
            history[-1]["gain_alpha"] = final_gain_alpha

        return beta, gain_vec, b0, history

    def precompute_design(self, trial_idx: np.ndarray, *,
                            T: Optional[int] = None) -> dict:
        """Build the design blocks and per-bin gain values that are shared
        across cells fit on the same session.

        Returns a dict with:
          blocks : {pred_name: (T, n_basis) design block}, omitting the
                   spike-history predictor (which depends on each cell's y)
          gain_t : {gain_name: (T,) per-bin gain modulator values}
          T      : number of bins (for validation)

        Pass this dict to ``fit(... design=...)`` or
        ``fit_summary(... design=...)`` to skip the per-cell design build.
        Saves the (T·n_lag·basis) convolution work that's otherwise repeated
        per cell — most useful for Ridge fits where it's ~25% of per-cell
        time. For Lasso the inner solver dominates so caching saves little.

        If `spike_history` is configured, the history block is rebuilt per
        cell from that cell's y (since y is itself the history input).
        """
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if T is None:
            T = trial_idx.size
        elif trial_idx.size != T:
            raise ValueError(
                f"trial_idx length {trial_idx.size} != T={T}")

        # Build series for every predictor except the auto spike-history
        # (its values depend on the cell's y and are filled in per fit).
        series = {}
        for p in self.predictors:
            if p.name == self._history_predictor_name:
                continue
            if _is_event(p):
                series[p.name] = _event_series(p.times, self.dt, T)
            else:
                series[p.name] = self._continuous_to_bins(
                    p.values, T,
                    times=getattr(p, "times", None),
                    align=getattr(p, "align", "interp"),
                    normalize=getattr(p, "normalize", "none"),
                    outlier_zscore=getattr(p, "outlier_zscore", 5.0))

        blocks = {p.name: _design_block(series[p.name],
                                          self._lags[p.name],
                                          self._basis[p.name])
                  for p in self.predictors
                  if p.name != self._history_predictor_name}
        gain_t = self._gain_per_t(T, trial_idx)
        return {"blocks": blocks, "gain_t": gain_t, "T": T}

    def _resolve_design(self, y: np.ndarray, T: int,
                          trial_idx: np.ndarray,
                          design: Optional[dict]):
        """Return (blocks, gain_t) using the precomputed `design` if given,
        else building from scratch. If spike_history is on, its block is
        always built fresh from `y` (since y is its input)."""
        if design is None:
            series = self._input_series(
                T, overrides=self._with_history(None, y))
            return self._design_blocks(series), self._gain_per_t(T, trial_idx)
        if design.get("T") != T:
            raise ValueError(
                f"precomputed design has T={design.get('T')}, got T={T}")
        blocks = dict(design["blocks"])  # shallow copy so we can add history
        gain_t = design["gain_t"]
        if self._history_predictor_name is not None:
            h = self._history_predictor_name
            blocks[h] = _design_block(
                np.asarray(y, dtype=float).ravel(),
                self._lags[h], self._basis[h])
        return blocks, gain_t

    def fit(self, y: np.ndarray, trial_idx: np.ndarray, *,
            fit_mask: Optional[np.ndarray] = None,
            design: Optional[dict] = None,
            max_iter: int = 100, tol: float = 1e-3,
            patience: int = 3, verbose: bool = False):
        """Alternating-least-squares fit of the bilinear model.

        Parameters
        ----------
        y         : (T,) signal to model, sampled at `dt` (one scalar per
            time bin).
        trial_idx : (T,) integer array — `trial_idx[t]` is the trial number
            owning bin `t`. Used to look up per-trial `GainModulator.values`:
            the gain at bin `t` is `V[trial_idx[t]]`. Must satisfy
            `0 <= trial_idx[t] < len(gain.values)` for every supplied gain.
            Trials don't need to be equal-length or strictly contiguous,
            but every bin needs a (valid) trial label.
        fit_mask  : optional (T,) bool. If given, only those bins enter the
            ALS solver, while the convolutional design is still built over
            the full series — useful for fitting on peri-event windows
            without trial-edge kernel leakage. Stored as `self.fit_mask_`
            and reused as the default by `score`, `predict_*`,
            `cross_val_score`, `delta_r2`, and `pseudosession_test`.
        design    : optional dict returned by ``precompute_design`` — when
            fitting many cells on the same session, building the design
            once and passing it here skips the per-cell convolution step.

        Convergence: every value gain coef changes by <= `tol` for `patience`
        consecutive iterations, or `max_iter` reached.
        """
        y = np.asarray(y, dtype=float).ravel()
        T = y.size
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if trial_idx.size != T:
            raise ValueError("trial_idx must have the same length as y")

        blocks, gain_t = self._resolve_design(y, T, trial_idx, design)

        if fit_mask is not None:
            m = np.asarray(fit_mask, dtype=bool).ravel()
            if m.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {m.size}")
            self.fit_mask_ = m
            y_fit = y[m]
            blocks_fit = {k: v[m] for k, v in blocks.items()}
            gain_t_fit = {k: v[m] for k, v in gain_t.items()}
        else:
            self.fit_mask_ = None
            y_fit, blocks_fit, gain_t_fit = y, blocks, gain_t

        self.beta_, self.gain_, self.intercept_, self.history_ = self._fit_als(
            y_fit, blocks_fit, gain_t_fit,
            max_iter=max_iter, tol=tol, patience=patience, verbose=verbose,
        )
        return self

    def _cross_validate_prepared(
            self, y: np.ndarray, trial_idx: np.ndarray,
            blocks: dict, gain_t: dict, eval_mask: np.ndarray, *,
            remove_gains: Sequence[str] = (),
            remove_predictors: Sequence[str] = (),
            n_folds: int = 5, fold_seed: Optional[int] = None,
            gap_history=False,
            max_iter: int = 100, tol: float = 1e-3,
            patience: int = 3, verbose: bool = False,
            ) -> _CrossValidationResult:
        """Shared trial-held-out evaluator over an already-built design.

        Gain-only reduced models keep each fold's final kernels and intercept
        but refit the retained gain coefficients with their own ridge alpha.
        If predictors are removed, the reduced model receives a full ALS refit
        with those design blocks zeroed, which is equivalent to omitting them
        while preserving this model's coefficient layout.
        """
        T = y.size
        eval_mask = np.asarray(eval_mask, dtype=bool).ravel()
        if eval_mask.size != T:
            raise ValueError(
                f"evaluation mask must have length T={T}, "
                f"got {eval_mask.size}")

        if gap_history:
            if self._history_predictor_name is None:
                raise ValueError(
                    "gap_history requires spike_history to be configured "
                    "on the model")
            gap_bins = (int(self._lags[self._history_predictor_name].max())
                        if gap_history is True else int(gap_history))
        else:
            gap_bins = 0

        compute_delta = bool(remove_gains) or bool(remove_predictors)
        reduced_gain_keep = self._gain_keep_mask(
            remove_gains=remove_gains,
            remove_predictors=remove_predictors)

        trials = np.unique(trial_idx)
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if n_folds > trials.size:
            raise ValueError(
                f"n_folds={n_folds} exceeds the number of trials "
                f"({trials.size})")
        if fold_seed is not None:
            trials = np.random.default_rng(fold_seed).permutation(trials)
        folds = np.array_split(trials, n_folds)

        cv_r2 = []
        cv_delta = [] if compute_delta else None
        n_iter_per_fold = []

        removed_predictors = set(remove_predictors)
        for fold_i, test_trials in enumerate(folds):
            train_trials = np.setdiff1d(trials, test_trials)
            train_in_fold = np.isin(trial_idx, train_trials)
            test_in_fold = np.isin(trial_idx, test_trials)
            if gap_bins > 0:
                train_in_fold = self._safe_under_history(
                    train_in_fold, gap_bins)
                test_in_fold = self._safe_under_history(
                    test_in_fold, gap_bins)
            train_mask = train_in_fold & eval_mask
            test_mask = test_in_fold & eval_mask

            y_tr = y[train_mask]
            blocks_tr = {k: v[train_mask] for k, v in blocks.items()}
            gain_t_tr = {k: v[train_mask] for k, v in gain_t.items()}

            if verbose:
                print(f"\n=== CV fold {fold_i + 1}/{n_folds} "
                      f"({test_mask.sum()} test bins, "
                      f"{train_mask.sum()} train bins"
                      + (f", gap={gap_bins}" if gap_bins else "")
                      + ") ===")

            beta, gain_vec, b0, fold_history = self._fit_als(
                y_tr, blocks_tr, gain_t_tr,
                max_iter=max_iter, tol=tol, patience=patience,
                verbose=verbose)
            n_iter_per_fold.append(len(fold_history))

            y_te = y[test_mask]
            blocks_te = {k: v[test_mask] for k, v in blocks.items()}
            gain_t_te = {k: v[test_mask] for k, v in gain_t.items()}
            full_yhat = self._predict_from_state(
                blocks_te, gain_t_te, beta, gain_vec, b0)
            r2_full = self._r2_score(y_te, full_yhat)
            cv_r2.append(r2_full)

            if compute_delta:
                if removed_predictors:
                    reduced_blocks_tr = {
                        name: (np.zeros_like(block)
                               if name in removed_predictors else block)
                        for name, block in blocks_tr.items()
                    }
                    beta_red, gain_red, b0_red, _ = self._fit_als(
                        y_tr, reduced_blocks_tr, gain_t_tr,
                        gain_keep_mask=reduced_gain_keep,
                        max_iter=max_iter, tol=tol, patience=patience,
                        verbose=verbose)
                    reduced_blocks_te = {
                        name: (np.zeros_like(block)
                               if name in removed_predictors else block)
                        for name, block in blocks_te.items()
                    }
                else:
                    # Paper-style gain comparison: preserve response shapes
                    # from the full fold fit, but fit the reduced gain model.
                    gain_red, _ = self._refit_gains(
                        y_tr, blocks_tr, gain_t_tr, beta, b0,
                        keep_mask=reduced_gain_keep,
                        alpha=self.gain_alpha)
                    beta_red = beta
                    b0_red = b0
                    reduced_blocks_te = blocks_te

                reduced_yhat = self._predict_from_state(
                    reduced_blocks_te, gain_t_te,
                    beta_red, gain_red, b0_red)
                r2_reduced = self._r2_score(y_te, reduced_yhat)
                cv_delta.append(r2_full - r2_reduced)

            if verbose:
                message = f"  fold {fold_i + 1} R²={r2_full:.4f}"
                if compute_delta:
                    message += f"  ΔR²={cv_delta[-1]:+.4f}"
                print(message)

        return _CrossValidationResult(
            r2_per_fold=np.asarray(cv_r2, dtype=float),
            delta_r2_per_fold=(np.asarray(cv_delta, dtype=float)
                               if cv_delta is not None else None),
            n_iter_per_fold=np.asarray(n_iter_per_fold, dtype=int),
        )

    def cross_val_score(self, y: np.ndarray, trial_idx: np.ndarray, *,
                        fit_mask: Optional[np.ndarray] = None,
                        gap_history=False,
                        n_folds: int = 5, fold_seed: Optional[int] = None,
                        max_iter: int = 100,
                        tol: float = 1e-3, patience: int = 3,
                        verbose: bool = False) -> float:
        """Trial-held-out cross-validated R².

        `y` is the (T,) target; `trial_idx` is the (T,) bin-to-trial map used
        to define the folds (see `BilinearGLM` class docstring).

        Trials are partitioned into `n_folds` groups. By default the groups are
        contiguous blocks of trials (in trial-id order); pass `fold_seed=<int>`
        to instead assign whole trials to folds at random (reproducibly). Random
        folds reduce the drift-driven catastrophic-fold blowups but interpolate
        across slow drift — see the README note on CV schemes. Whole trials are
        always kept together regardless. For each fold the model is refit from
        scratch on the remaining trials; the full convolution is computed over
        all time bins so that kernel windows that span trial boundaries are
        handled correctly, but only rows belonging to training trials enter the
        ALS solver.

        If `fit_mask` is provided (or one was stored at fit time), both the
        training and test rows are restricted to its True bins — convolutions
        are still built over the full series, so the same peri-event windows
        used at fit time can be used for CV without trial-edge leakage.

        `gap_history` controls the autoregressive leak through the
        spike-history predictor. With the default `spike_history` on, a bin's
        history regressors look back into the lagged `y`; without gapping, a
        train bin near a fold boundary uses test-fold `y` (and vice versa).
        Set `gap_history=True` to drop every bin whose history window crosses
        the current fold boundary (gap size = history kernel's max lag in
        samples). Pass an int to override the gap size. Requires
        `spike_history` to be configured.

        Returns the mean R² across folds. Per-fold scores are stored in
        `self.cv_scores_` (shape (n_folds,)).

        Does NOT update `self.beta_` / `self.gain_` / `self.intercept_`.
        Call `fit()` separately to obtain a model trained on all data.
        """
        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if trial_idx.size != T:
            raise ValueError("trial_idx must have the same length as y")

        if fit_mask is None:
            fit_mask = self.fit_mask_
        if fit_mask is not None:
            eval_mask = np.asarray(fit_mask, dtype=bool).ravel()
            if eval_mask.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {eval_mask.size}")
        else:
            eval_mask = np.ones(T, dtype=bool)

        blocks, gain_t = self._resolve_design(
            y_arr, T, trial_idx, design=None)
        result = self._cross_validate_prepared(
            y_arr, trial_idx, blocks, gain_t, eval_mask,
            n_folds=n_folds, fold_seed=fold_seed,
            gap_history=gap_history,
            max_iter=max_iter, tol=tol, patience=patience,
            verbose=verbose)
        self.cv_scores_ = result.r2_per_fold
        return float(np.nanmean(self.cv_scores_))

    def fit_summary(self, y: np.ndarray, trial_idx: np.ndarray, *,
                     remove_gains: Sequence[str] = (),
                     remove_predictors: Sequence[str] = (),
                     fit_mask: Optional[np.ndarray] = None,
                     design: Optional[dict] = None,
                     n_folds: int = 5, fold_seed: Optional[int] = None,
                     gap_history=False,
                     max_iter: int = 100, tol: float = 1e-3,
                     patience: int = 3, verbose: bool = False) -> dict:
        """Fit + in-sample R² + CV R² + ΔR² in one shared pass.

        Equivalent to calling ``fit``, ``score``, ``cross_val_score``, and
        ``delta_r2`` separately, but shares one design build and one set of
        full-model fold fits. Gain-only reduced models add inexpensive ridge
        refits against those folds' final kernels. Whole-predictor reductions
        add one reduced ALS fit per fold.

        In every case, full and reduced predictions are scored on identical
        held-out bins.

        After this method returns, the model is fitted on the full training
        data (``self.beta_``, ``self.gain_``, ``self.intercept_`` are set),
        so further inspection / prediction works as usual.

        Parameters
        ----------
        y, trial_idx : as in ``fit``.
        remove_gains, remove_predictors : as in ``delta_r2``. Gain-only
            reductions refit gains with the full model's final kernels;
            predictor reductions refit the reduced ALS model. If both are
            empty, the ΔR² fields of the returned dict are None.
        fit_mask, n_folds, fold_seed, gap_history : as in ``cross_val_score``.
        design : optional precomputed design dict (see
            ``precompute_design``). When fitting many cells against the
            same session, build it once and pass it here to skip the
            per-cell design construction.
        max_iter, tol, patience, verbose : as in ``fit``.

        Returns
        -------
        dict with keys:
          train_r2          : in-sample R² of the full model.
          cv_r2             : mean held-out R² across folds.
          cv_r2_per_fold    : (n_folds,) array of per-fold R².
          delta_r2          : mean held-out ΔR² (or None).
          delta_r2_per_fold : (n_folds,) array (or None).
          kernels           : {pred_name: time-domain kernel array}.
          gain_table        : as ``gain_table()``.
          intercept         : scalar intercept b0.
          n_iter            : ALS iterations used by the full-data fit
                              (== max_iter means convergence wasn't reached).
          n_iter_per_fold   : (n_folds,) array of ALS iterations per fold.
          converged         : True iff the full-data fit converged before
                              hitting max_iter.
        """
        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if trial_idx.size != T:
            raise ValueError("trial_idx must have the same length as y")

        if fit_mask is not None:
            mask = np.asarray(fit_mask, dtype=bool).ravel()
            if mask.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {mask.size}")
        else:
            mask = None
        eval_mask = mask if mask is not None else np.ones(T, dtype=bool)

        # Build design ONCE — shared across full fit, in-sample R², all folds.
        # If `design` was supplied (precompute_design output), use it and only
        # rebuild the history block from this cell's y.
        blocks, gain_t = self._resolve_design(y_arr, T, trial_idx, design)

        # --- full-data fit (replaces fit()) ----------------------------------
        if mask is not None:
            y_fit = y_arr[mask]
            blocks_fit = {k: v[mask] for k, v in blocks.items()}
            gain_t_fit = {k: v[mask] for k, v in gain_t.items()}
        else:
            y_fit, blocks_fit, gain_t_fit = y_arr, blocks, gain_t
        self.fit_mask_ = mask
        (self.beta_, self.gain_, self.intercept_,
         self.history_) = self._fit_als(
            y_fit, blocks_fit, gain_t_fit,
            max_iter=max_iter, tol=tol, patience=patience, verbose=verbose)

        # --- in-sample R² (replaces score()) — reuses the already-built design
        yhat = self._predict_from_state(
            blocks, gain_t, self.beta_, self.gain_, self.intercept_)
        y_in = y_arr[eval_mask]
        yhat_in = yhat[eval_mask]
        ss_tot_in = float(np.sum((y_in - y_in.mean()) ** 2))
        train_r2 = (1.0 - float(np.sum((y_in - yhat_in) ** 2)) / ss_tot_in
                    if ss_tot_in > 0 else float("nan"))

        # --- CV pass: full R² and optional refit reduced-model ΔR² ----------
        cv_result = self._cross_validate_prepared(
            y_arr, trial_idx, blocks, gain_t, eval_mask,
            remove_gains=remove_gains,
            remove_predictors=remove_predictors,
            n_folds=n_folds, fold_seed=fold_seed,
            gap_history=gap_history,
            max_iter=max_iter, tol=tol, patience=patience,
            verbose=verbose)
        cv_r2_arr = cv_result.r2_per_fold
        cv_delta_arr = cv_result.delta_r2_per_fold
        self.cv_scores_ = cv_r2_arr
        self.cv_delta_r2_ = cv_delta_arr

        return {
            "train_r2": train_r2,
            "cv_r2": float(np.nanmean(cv_r2_arr)),
            "cv_r2_per_fold": cv_r2_arr,
            "delta_r2": (float(np.nanmean(cv_delta_arr))
                         if cv_delta_arr is not None else None),
            "delta_r2_per_fold": cv_delta_arr,
            "kernels": {p.name: self.kernel(p.name)
                        for p in self.predictors},
            "gain_table": self.gain_table(),
            "intercept": float(self.intercept_),
            "n_iter": len(self.history_),
            "n_iter_per_fold": cv_result.n_iter_per_fold,
            "converged": len(self.history_) < max_iter,
        }

    def _predict_from_state(self, blocks, gain_t, beta, gain_vec, b0):
        T = next(iter(blocks.values())).shape[0]
        drives = self._kernel_drives(blocks, beta)
        out = np.full(T, b0)
        for p in self.predictors:
            if p.gain_modulated:
                w = np.full(T, gain_vec[self._gain_offset_idx[p.name]])
                for vname in self._gains_for[p.name]:
                    a = gain_vec[self._gain_var_idx[(vname, p.name)]]
                    w = w + a * gain_t[vname]
                out = out + w * drives[p.name]
            else:
                out = out + drives[p.name]
        return out

    def _check_predictor_compat(self, p_new) -> None:
        """Validate that a new predictor matches the fitted model's structure."""
        if p_new.name not in self._pred_idx:
            raise ValueError(f"unknown predictor {p_new.name!r}")
        p_orig = self.predictors[self._pred_idx[p_new.name]]
        if type(p_new).__name__ != type(p_orig).__name__:
            raise ValueError(
                f"predictor {p_new.name!r}: type mismatch "
                f"({type(p_new).__name__} vs fitted "
                f"{type(p_orig).__name__})")
        mismatches = []
        for f in ("window", "n_basis", "basis", "gain_modulated"):
            if getattr(p_new, f) != getattr(p_orig, f):
                mismatches.append(
                    f"{f}={getattr(p_new, f)!r} vs fitted "
                    f"{getattr(p_orig, f)!r}")
        if mismatches:
            raise ValueError(
                f"predictor {p_new.name!r} doesn't match fitted model: "
                + "; ".join(mismatches))

    def _check_gain_compat(self, g_new) -> None:
        """Validate that a new gain modulator matches the fitted model's gain."""
        g_lookup = {g.name: g for g in self.gains}
        if g_new.name not in g_lookup:
            raise ValueError(f"unknown gain {g_new.name!r}")
        g_orig = g_lookup[g_new.name]
        if g_new.modulates != g_orig.modulates:
            raise ValueError(
                f"gain {g_new.name!r}: modulates={g_new.modulates!r} "
                f"doesn't match fitted {g_orig.modulates!r}")

    def predict_fitted(self, y: np.ndarray, trial_idx: np.ndarray, *,
                       event_overrides: Optional[dict] = None,
                       continuous_overrides: Optional[dict] = None,
                       gain_overrides: Optional[dict] = None,
                       fit_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict y using the predictors/gains supplied at fit time, with
        optional targeted substitutions.

        `y` is the (T,) target (used only for `T` and as the history input
        when `spike_history` is on); `trial_idx` is the (T,) bin-to-trial
        map. See the class docstring for the data layout.

        Defaults to the in-sample prediction (uses `y` as the history series
        if `spike_history` is on, and the fit-time event times / continuous
        values / gain values for everything else). Pass `*_overrides` dicts
        to swap specific entries — useful for spike-history simulation,
        pseudosession-style nulls, or "what-if" comparisons that share most
        of the original session's data.

        For predicting on a **new session** with structurally identical
        predictors but different data, use `predict_with` instead.

        If `fit_mask` is provided, returns predictions only at masked bins.
        Convolutional design is still built over the full T-length series, so
        edge bins of the mask see the correct surrounding context.
        """
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        y = np.asarray(y, dtype=float).ravel()
        T = y.size
        overrides = {}
        if event_overrides:
            overrides.update(event_overrides)
        if continuous_overrides:
            overrides.update(continuous_overrides)
        overrides = self._with_history(overrides, y)
        series = self._input_series(T, overrides=overrides)
        blocks = self._design_blocks(series)
        gain_t = self._gain_per_t(T, trial_idx, gain_overrides=gain_overrides)
        yhat = self._predict_from_state(blocks, gain_t, self.beta_,
                                         self.gain_, self.intercept_)
        if fit_mask is not None:
            m = np.asarray(fit_mask, dtype=bool).ravel()
            if m.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {m.size}")
            return yhat[m]
        return yhat

    def predict_with(self,
                     predictors: Sequence = (),
                     gains: Sequence = (),
                     *,
                     trial_idx: np.ndarray,
                     history: Optional[np.ndarray] = None,
                     T: Optional[int] = None,
                     fit_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict y for a new session by passing fresh predictor and gain
        objects.

        Mirrors the construction-time API: each `EventPredictor` /
        `ContinuousPredictor` / `GainModulator` carries its own data (event
        times, continuous values, per-trial gain values). The structural
        fields (window, n_basis, basis, gain_modulated for predictors;
        modulates for gains) must match the corresponding fitted entry by
        name. Any predictor or gain you don't include falls back to the
        data supplied at construction time.

        If the model was constructed with `spike_history`, pass the new
        session's signal as `history=` (since there's no `y` argument).

        Parameters
        ----------
        predictors  : sequence of EventPredictor / ContinuousPredictor
        gains       : sequence of GainModulator
        trial_idx   : (T,) bin-to-trial map for the new session
        history     : optional (T,) array; required if `spike_history` is on
        T           : optional; defaults to `len(trial_idx)`
        fit_mask    : optional bool mask to restrict the returned bins

        Returns
        -------
        yhat : (T,) prediction (or shorter if `fit_mask` is given).
        """
        if self.beta_ is None:
            raise RuntimeError("call fit() first")

        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if T is None:
            T = trial_idx.size
        elif trial_idx.size != T:
            raise ValueError(
                f"trial_idx has length {trial_idx.size}, expected T={T}")

        overrides: dict = {}
        for p_new in predictors:
            self._check_predictor_compat(p_new)
            if _is_event(p_new):
                overrides[p_new.name] = p_new.times
            elif getattr(p_new, "times", None) is not None:
                overrides[p_new.name] = (p_new.values, p_new.times)
            else:
                overrides[p_new.name] = p_new.values

        gain_ov: dict = {}
        for g_new in gains:
            self._check_gain_compat(g_new)
            gain_ov[g_new.name] = g_new.values

        if self._history_predictor_name is not None:
            if history is not None:
                overrides[self._history_predictor_name] = history
            elif self._history_predictor_name not in overrides:
                raise ValueError(
                    "model was constructed with spike_history but no history "
                    "series was supplied — pass `history=` (the new session's "
                    "signal) or include the history predictor in `predictors`")

        series = self._input_series(T, overrides=overrides)
        blocks = self._design_blocks(series)
        gain_t = self._gain_per_t(T, trial_idx, gain_overrides=gain_ov)
        yhat = self._predict_from_state(blocks, gain_t, self.beta_,
                                         self.gain_, self.intercept_)
        if fit_mask is not None:
            m = np.asarray(fit_mask, dtype=bool).ravel()
            if m.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {m.size}")
            return yhat[m]
        return yhat

    def score(self, y: np.ndarray, trial_idx: np.ndarray, *,
              fit_mask: Optional[np.ndarray] = None, **kw) -> float:
        """R^2 of the fitted model on (y, trial_idx).

        If `fit_mask` is given (or one was stored at fit time), R^2 is
        computed only over masked bins. Pass ``fit_mask=np.ones(T, bool)``
        to force scoring over the full session even when a fit-time mask
        was stored.
        """
        if fit_mask is None:
            fit_mask = self.fit_mask_
        y_arr = np.asarray(y, dtype=float).ravel()
        yhat = self.predict_fitted(y_arr, trial_idx, **kw)
        if fit_mask is not None:
            m = np.asarray(fit_mask, dtype=bool).ravel()
            if m.size != y_arr.size:
                raise ValueError(
                    f"fit_mask must have length T={y_arr.size}, got {m.size}")
            y_arr = y_arr[m]
            yhat = yhat[m]
        ss_res = float(np.sum((y_arr - yhat) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ----- accessors --------------------------------------------------------

    def kernel(self, name: str) -> np.ndarray:
        """Time-domain kernel K_p(tau) for predictor p, indexed by lag samples."""
        k = self._pred_idx[name]
        start, end = self._beta_off[k], self._beta_off[k + 1]
        return self._basis[name] @ self.beta_[start:end]

    def lags(self, name: str) -> np.ndarray:
        """Lag axis (in samples) for the kernel of `name`."""
        return self._lags[name].copy()

    def lags_seconds(self, name: str) -> np.ndarray:
        return self._lags[name] * self.dt

    def gain_offset(self, name: str) -> float:
        return float(self.gain_[self._gain_offset_idx[name]])

    def gain_coefficient(self, gain_name: str, pred_name: str) -> float:
        return float(self.gain_[self._gain_var_idx[(gain_name, pred_name)]])

    def gain_table(self) -> dict:
        """{pred_name: {"offset": ..., "<gain_var>": ...}} for inspection."""
        out = {}
        for p in self._modulated:
            row = {"offset": self.gain_offset(p.name)}
            for v in self._gains_for[p.name]:
                row[v] = self.gain_coefficient(v, p.name)
            out[p.name] = row
        return out

    def psth(self, name: str, y: np.ndarray) -> Optional[np.ndarray]:
        """Mean of `y` aligned to event times of predictor `name`.

        Returns a 1D array on the same lag grid as `lags_seconds(name)`.
        Events whose lag window falls outside `y`'s extent contribute NaN
        at the out-of-range lags; the average is taken with `nanmean`.
        Returns None if `name` isn't an EventPredictor.
        """
        p = self.predictors[self._pred_idx[name]]
        if not _is_event(p):
            return None
        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        lags = self._lags[name]
        if p.times is None or len(p.times) == 0:
            return np.full(lags.size, np.nan)
        bins = np.floor(np.asarray(p.times, dtype=float) / self.dt).astype(int)
        idx = bins[:, None] + lags[None, :]
        valid = (idx >= 0) & (idx < T)
        snippets = np.where(valid, y_arr[np.clip(idx, 0, T - 1)], np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(snippets, axis=0)

    def print_parameter_table(self) -> None:
        """Print intercept, gain offsets/slopes, and kernel norms to stdout."""
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        print(f"intercept: {self.intercept_:+.4f}")
        if self._modulated:
            print("\nGain-modulated predictors:")
            all_vars = sorted({v for p in self._modulated
                                 for v in self._gains_for[p.name]})
            header = f"  {'predictor':<18}  {'offset':>10}"
            for v in all_vars:
                header += f"  {v:>10}"
            header += f"  {'|K|2':>8}"
            print(header)
            for p in self._modulated:
                row = (f"  {p.name:<18}  "
                       f"{self.gain_offset(p.name):>+10.4f}")
                for v in all_vars:
                    if v in self._gains_for[p.name]:
                        row += f"  {self.gain_coefficient(v, p.name):>+10.4f}"
                    else:
                        row += f"  {'—':>10}"
                row += f"  {np.linalg.norm(self.kernel(p.name)):>8.4f}"
                print(row)
        non_mod = [p for p in self.predictors if not p.gain_modulated]
        if non_mod:
            print("\nNon-modulated predictors:")
            print(f"  {'predictor':<18}  {'|K|2':>8}  {'max|K|':>8}")
            for p in non_mod:
                k = self.kernel(p.name)
                print(f"  {p.name:<18}  {np.linalg.norm(k):>8.4f}  "
                      f"{np.max(np.abs(k)):>8.4f}")
        if self.history_:
            last = self.history_[-1]
            print(f"\nFinal regularization: "
                  f"kernel_alpha={last['kernel_alpha']:.3g}  "
                  f"gain_alpha={last['gain_alpha']:.3g}  "
                  f"({len(self.history_)} ALS iterations)")

    def summary_plot(self, y: Optional[np.ndarray] = None, *,
                     axes=None, figsize=None,
                     print_table: bool = True):
        """Plot fitted kernels; overlay PSTHs of `y` on event-predictor axes.

        One subplot per predictor. The kernel is drawn on the primary y-axis;
        for EventPredictors, if `y` is given, the PSTH of `y` aligned to that
        predictor's event times is drawn on a twin y-axis (red).

        Parameters
        ----------
        y           : optional (T,) signal used to compute event-aligned PSTHs
        axes        : optional flat sequence of matplotlib axes
                      (one per predictor); if None, a new figure is created
        figsize     : figsize for the new figure (auto if None)
        print_table : if True, also prints the parameter-fits table

        Returns the matplotlib Figure.
        """
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        import matplotlib.pyplot as plt

        n_pred = len(self.predictors)
        if axes is None:
            ncols = min(n_pred, 3)
            nrows = int(np.ceil(n_pred / ncols))
            if figsize is None:
                figsize = (4.2 * ncols, 3.0 * nrows)
            fig, ax_arr = plt.subplots(nrows, ncols, figsize=figsize,
                                        squeeze=False)
            ax_list = ax_arr.ravel().tolist()
        else:
            ax_list = list(np.atleast_1d(axes).ravel())
            if len(ax_list) < n_pred:
                raise ValueError(
                    f"need >= {n_pred} axes, got {len(ax_list)}")
            fig = ax_list[0].figure

        y_arr = (np.asarray(y, dtype=float).ravel()
                 if y is not None else None)

        for i, p in enumerate(self.predictors):
            ax = ax_list[i]
            lags_sec = self.lags_seconds(p.name)
            kernel = self.kernel(p.name)
            ax.plot(lags_sec, kernel, color="C0", lw=1.5, label="kernel")
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.axvline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_xlabel("lag (s)")
            ax.set_ylabel("kernel", color="C0")
            ax.tick_params(axis="y", labelcolor="C0")
            tag = " (gain)" if p.gain_modulated else ""
            ax.set_title(f"{p.name}{tag}")

            if _is_event(p) and y_arr is not None:
                psth = self.psth(p.name, y_arr)
                if psth is not None and np.isfinite(psth).any():
                    ax2 = ax.twinx()
                    ax2.plot(lags_sec, psth, color="C3", lw=1.2,
                             alpha=0.8, label="PSTH")
                    ax2.set_ylabel("y (PSTH)", color="C3")
                    ax2.tick_params(axis="y", labelcolor="C3")

        for ax in ax_list[n_pred:]:
            ax.set_visible(False)

        fig.tight_layout()
        if print_table:
            self.print_parameter_table()
        return fig

    def plot_fit(self, y: np.ndarray, trial_idx: np.ndarray, *,
                 fraction: float = 1.0 / 3.0, start: int = 0,
                 show_events: bool = True, show_gains: bool = True,
                 show_continuous: bool = True,
                 ax=None, figsize=None):
        """Plot a snippet of `y` with the full-model prediction overlaid.

        Parameters
        ----------
        y, trial_idx : as in `fit()`. The prediction uses the in-sample
                       `predict_fitted` (i.e. fit-time predictors and gains).
        fraction     : portion of T to show (default 1/3). Ignored if the
                       resulting window has zero length.
        start        : starting bin index (default 0).
        show_events  : if True, mark EventPredictor times falling within the
                       snippet as raster ticks at the top of the y/model panel.
        show_gains   : if True (and the model has gain-modulated predictors
                       and `ax` is not supplied), add a panel below the
                       y/model trace showing g_p(t) = offset + Σ slope·V(t)
                       for each gain-modulated predictor, with a dashed line
                       at the offset.
        show_continuous : if True (and there are ContinuousPredictors other
                       than the auto spike-history series, and `ax` is not
                       supplied), add a final panel showing each continuous
                       predictor's input series over the snippet.
        ax           : optional matplotlib axes; if None, a new figure is made.
                       If supplied, only the y/model panel is drawn.
        figsize      : figsize for the new figure (auto if None).

        Returns the matplotlib Figure.
        """
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        import matplotlib.pyplot as plt

        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if not 0 <= start < T:
            raise ValueError(f"start must be in [0, T={T}), got {start}")
        stop = min(T, start + max(1, int(np.floor(fraction * T))))

        yhat = self.predict_fitted(y_arr, trial_idx)
        t_axis = np.arange(start, stop) * self.dt

        continuous_preds = [p for p in self.predictors
                            if _is_continuous(p)
                            and p.name != self._history_predictor_name]
        want_gain_panel = (ax is None and show_gains
                           and len(self._modulated) > 0
                           and len(self.gains) > 0)
        want_cont_panel = (ax is None and show_continuous
                           and len(continuous_preds) > 0)

        ax_g = None
        ax_c = None
        if ax is None:
            heights = [3.0]
            roles = ["main"]
            if want_gain_panel:
                heights.append(1.5)
                roles.append("gain")
            if want_cont_panel:
                heights.append(1.5)
                roles.append("cont")
            if figsize is None:
                figsize = (12, 3.5 + 1.5 * (len(roles) - 1))
            fig, ax_arr = plt.subplots(
                len(roles), 1, figsize=figsize, sharex=True,
                gridspec_kw={"height_ratios": heights}, squeeze=False)
            ax_arr = ax_arr.ravel()
            ax = ax_arr[0]
            if "gain" in roles:
                ax_g = ax_arr[roles.index("gain")]
            if "cont" in roles:
                ax_c = ax_arr[roles.index("cont")]
            bottom_ax = ax_arr[-1]
        else:
            fig = ax.figure
            bottom_ax = ax

        ax.plot(t_axis, y_arr[start:stop], color="k", lw=0.8,
                alpha=0.6, label="y")
        ax.plot(t_axis, yhat[start:stop], color="C3", lw=1.2,
                label="model")
        ax.set_ylabel("y")
        ax.set_title(f"fit snippet (bins {start}:{stop} of {T})")

        if show_events:
            t_lo, t_hi = start * self.dt, stop * self.dt
            event_preds = [p for p in self.predictors
                           if _is_event(p)
                           and p.times is not None]
            n_evt = len(event_preds)
            # one raster row per event predictor, stacked just above the data
            for row, p in enumerate(event_preds):
                times = np.asarray(p.times, dtype=float)
                times = times[(times >= t_lo) & (times < t_hi)]
                if times.size == 0:
                    continue
                color = f"C{(self.predictors.index(p) % 9) + 1}"
                # ymin/ymax in axis coords: top ~8% of the plot, split into rows
                band_lo = 1.0 - 0.08 * (row + 1) / max(n_evt, 1)
                band_hi = 1.0 - 0.08 * row / max(n_evt, 1)
                for ti, t in enumerate(times):
                    ax.axvline(t, ymin=band_lo, ymax=band_hi,
                                color=color, lw=0.8, alpha=0.9,
                                label=p.name if ti == 0 else None)

        ax.legend(loc="best", fontsize=9, framealpha=0.9)

        if ax_g is not None:
            gain_t_full = self._gain_per_t(T, trial_idx)
            for p in self._modulated:
                offset = self.gain_offset(p.name)
                g_t = np.full(stop - start, offset)
                for vname in self._gains_for[p.name]:
                    slope = self.gain_coefficient(vname, p.name)
                    g_t = g_t + slope * gain_t_full[vname][start:stop]
                color = f"C{(self.predictors.index(p) % 9) + 1}"
                ax_g.plot(t_axis, g_t, color=color, lw=1.0, label=p.name)
                ax_g.axhline(offset, color=color, lw=0.6, ls="--",
                              alpha=0.5)
            ax_g.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax_g.set_ylabel("gain $g_p(t)$")
            ax_g.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)

        if ax_c is not None:
            for p in continuous_preds:
                k = self.predictors.index(p)
                color = f"C{(k % 9) + 1}"
                series = self._continuous_to_bins(
                    p.values, T,
                    times=getattr(p, "times", None),
                    align=getattr(p, "align", "interp"),
                    normalize=getattr(p, "normalize", "none"),
                    outlier_zscore=getattr(p, "outlier_zscore", 5.0))
                used = self._continuous_used_mask(
                    p.values, T,
                    times=getattr(p, "times", None),
                    align=getattr(p, "align", "interp"),
                    outlier_zscore=getattr(p, "outlier_zscore", 5.0))
                series_plot = np.where(used, series, np.nan)
                ax_c.plot(t_axis, series_plot[start:stop], color=color,
                           lw=0.9, label=p.name)
            ax_c.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax_c.set_ylabel("continuous")
            ax_c.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)

        bottom_ax.set_xlabel("time (s)")

        fig.tight_layout()
        return fig

    # ----- nested model comparison -----------------------------------------

    def delta_r2(self, y: np.ndarray, trial_idx: np.ndarray, *,
                 remove_gains: Sequence[str] = (),
                 remove_predictors: Sequence[str] = (),
                 fit_mask: Optional[np.ndarray] = None,
                 design: Optional[dict] = None,
                 n_folds: int = 5, fold_seed: Optional[int] = None,
                 gap_history=False,
                 max_iter: int = 100, tol: float = 1e-3,
                 patience: int = 3, verbose: bool = False) -> float:
        """Cross-validated ΔR² = R²(full) - R²(reduced) on held-out test trials.

        For gain-only comparisons, each fold's full bilinear model supplies
        the final temporal kernels and intercept. Full gains are refit against
        those final kernels, and the reduced gains are independently refit
        after removing the requested trial-variable columns. This matches the
        paper's fixed-kernel comparison while allowing retained gain terms to
        adjust to the reduced model.

        Removing whole predictors instead triggers a full reduced ALS fit in
        each training fold, with those predictor blocks omitted. If gains and
        predictors are both removed, both restrictions apply to that reduced
        fit.

        If `fit_mask` is given (or one was stored at fit time), both training
        and test rows are restricted to its True bins. `design`, `fold_seed`,
        and `gap_history` work as in `fit_summary` / `cross_val_score`.

        Returns the mean ΔR² across folds; per-fold values are stored in
        `self.cv_delta_r2_`.

        Does NOT update `self.beta_` / `self.gain_` / `self.intercept_`.
        """
        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if trial_idx.size != T:
            raise ValueError("trial_idx must have the same length as y")

        if fit_mask is None:
            fit_mask = self.fit_mask_
        if fit_mask is not None:
            eval_mask = np.asarray(fit_mask, dtype=bool).ravel()
            if eval_mask.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {eval_mask.size}")
        else:
            eval_mask = np.ones(T, dtype=bool)

        blocks, gain_t = self._resolve_design(
            y_arr, T, trial_idx, design)
        result = self._cross_validate_prepared(
            y_arr, trial_idx, blocks, gain_t, eval_mask,
            remove_gains=remove_gains,
            remove_predictors=remove_predictors,
            n_folds=n_folds, fold_seed=fold_seed,
            gap_history=gap_history,
            max_iter=max_iter, tol=tol, patience=patience,
            verbose=verbose)
        self.cv_scores_ = result.r2_per_fold
        self.cv_delta_r2_ = (
            result.delta_r2_per_fold
            if result.delta_r2_per_fold is not None
            else np.zeros_like(result.r2_per_fold))
        return float(np.nanmean(self.cv_delta_r2_))

    # ----- pseudosession permutation test -----------------------------------

    def pseudosession_test(self, y: np.ndarray, trial_idx: np.ndarray,
                           gain_name: str, *,
                           fit_mask: Optional[np.ndarray] = None,
                           n_perm: int = 200,
                           rng: Optional[np.random.Generator] = None) -> dict:
        """Permutation-based significance test for one trial-by-trial gain.

        Holds kernels and intercept fixed at their fitted values; for each of
        `n_perm` shuffles, draws a circularly shifted version of the gain's
        trial values, refits *only* the gain coefficients (ridge), and records
        ΔR². Full and reduced gain models select regularization independently
        unless ``gain_alpha`` was fixed at construction. P-value is the
        fraction of nulls with ΔR² >= observed.

        If `fit_mask` is given (or one was stored at fit time), both the gain
        refits and the ΔR² are restricted to masked bins — same window the
        original `fit()` used.
        """
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        if rng is None:
            rng = np.random.default_rng(0)
        if fit_mask is None:
            fit_mask = self.fit_mask_

        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        series = self._input_series(T,
                                    overrides=self._with_history(None, y_arr))
        blocks = self._design_blocks(series)
        gain_t_real = self._gain_per_t(T, trial_idx)
        drives = self._kernel_drives(blocks, self.beta_)
        offset_t = np.full(T, self.intercept_)
        for p in self.predictors:
            if not p.gain_modulated:
                offset_t = offset_t + drives[p.name]

        Z_full = self._build_gain_design(drives, gain_t_real)
        target = y_arr - offset_t

        if fit_mask is not None:
            mask = np.asarray(fit_mask, dtype=bool).ravel()
            if mask.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {mask.size}")
        else:
            mask = np.ones(T, dtype=bool)

        y_fit       = y_arr[mask]
        offset_fit  = offset_t[mask]
        target_fit  = target[mask]
        Z_full_fit  = Z_full[mask]

        observed = self._delta_r2_remove(
            y_fit, offset_fit, Z_full_fit, target_fit, gain_name)

        # null distribution: circularly shift this gain's trial values
        gv = next(g for g in self.gains if g.name == gain_name)
        n_trials = np.asarray(gv.values).size
        nulls = np.zeros(n_perm)
        for i in range(n_perm):
            shift = int(rng.integers(1, n_trials))
            shuffled = np.roll(np.asarray(gv.values, dtype=float), shift)
            gain_t_null = dict(gain_t_real)
            gain_t_null[gain_name] = shuffled[trial_idx]
            Z_null = self._build_gain_design(drives, gain_t_null)
            Z_null_fit = Z_null[mask]
            nulls[i] = self._delta_r2_remove(
                y_fit, offset_fit, Z_null_fit, target_fit, gain_name)
        p = float(np.mean(nulls >= observed))
        return {"observed": float(observed), "null": nulls, "p": p}

    def _delta_r2_remove(self, y, offset, Z, target, gain_name):
        """Gain-only ΔR² with independently refit full and reduced ridge."""
        full_coef, _ = self._fit_gain_coefficients(
            Z, target, alpha=self.gain_alpha)
        keep = self._gain_keep_mask(remove_gains=[gain_name])
        reduced_coef, _ = self._fit_gain_coefficients(
            Z, target, keep_mask=keep, alpha=self.gain_alpha)
        full_yhat = offset + Z @ full_coef
        reduced_yhat = offset + Z @ reduced_coef
        return (self._r2_score(y, full_yhat)
                - self._r2_score(y, reduced_yhat))
