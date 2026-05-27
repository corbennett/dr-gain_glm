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

from dataclasses import dataclass, field
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
    """Dense regressor: arbitrary 1D signal sampled at the y bin grid, convolved
    with a temporal kernel. To use as a spike-history term, pass values=y and
    window=(dt, T_history)."""
    name: str
    values: np.ndarray           # length-T series at the same dt as y
    window: tuple
    n_basis: int = 8
    basis: str = "cosine"
    gain_modulated: bool = False


@dataclass
class GainModulator:
    """Trial-by-trial scalar that multiplies one or more gain-modulated kernels.

    `modulates=None` means: every gain-modulated predictor.
    """
    name: str
    values: np.ndarray           # length-n_trials trial-by-trial values
    modulates: Optional[Sequence[str]] = None


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
            ))
            self._history_predictor_name = h_name

        self.gains = list(gains)
        self.kernel_regularizer = kernel_regularizer
        self.kernel_alpha = kernel_alpha
        self.gain_alpha = gain_alpha
        self.alphas = (np.logspace(-5, 5, 51) if alphas is None
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

    def _input_series(self, T: int, overrides: Optional[dict] = None):
        """Return {pred_name: 1D series of length T} for every predictor."""
        out = {}
        ov = overrides or {}
        for p in self.predictors:
            if p.name in ov:
                arr = np.asarray(ov[p.name], dtype=float).ravel()
                if isinstance(p, EventPredictor):
                    out[p.name] = _event_series(arr, self.dt, T)
                else:
                    if arr.size != T:
                        raise ValueError(f"override for {p.name!r} has length "
                                         f"{arr.size}, expected {T}")
                    out[p.name] = arr
            elif isinstance(p, EventPredictor):
                out[p.name] = _event_series(p.times, self.dt, T)
            else:
                arr = np.asarray(p.values, dtype=float).ravel()
                if arr.size != T:
                    raise ValueError(f"continuous predictor {p.name!r} has "
                                     f"length {arr.size}, expected T={T}")
                out[p.name] = arr
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

    # ----- fit / predict ----------------------------------------------------

    def _fit_als(self, y: np.ndarray, blocks: dict, gain_t: dict, *,
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

        kernel_alpha = self.kernel_alpha
        gain_alpha = self.gain_alpha

        var_indices = [self._gain_var_idx[(v, p)]
                       for (v, p) in self._gain_var_idx]
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
                    m = Lasso(alpha=kernel_alpha, fit_intercept=True,
                              max_iter=10_000)
                    m.fit(X, y)
                    beta = m.coef_.copy()
                    b0 = float(m.intercept_)
            elif self.kernel_regularizer == "ridge":
                if kernel_alpha is None:
                    cv = RidgeCV(alphas=self.alphas, cv=self.cv_folds,
                                 fit_intercept=True)
                    cv.fit(X, y)
                    kernel_alpha = float(cv.alpha_)
                    beta = cv.coef_.copy()
                    b0 = float(cv.intercept_)
                else:
                    m = Ridge(alpha=kernel_alpha, fit_intercept=True)
                    m.fit(X, y)
                    beta = m.coef_.copy()
                    b0 = float(m.intercept_)
            else:
                raise ValueError(f"unknown kernel_regularizer "
                                 f"{self.kernel_regularizer!r}")

            # ---- step 2: normalise kernels & rescale gain offsets ----------
            self._normalise_kernels(beta, gain_vec)

            # ---- step 3: gains | kernels -----------------------------------
            drives = self._kernel_drives(blocks, beta)
            offset_t = np.full(T, b0)
            for p in self.predictors:
                if not p.gain_modulated:
                    offset_t = offset_t + drives[p.name]

            target = y - offset_t
            Z = self._build_gain_design(drives, gain_t)

            if Z is not None and Z.shape[1] > 0:
                if gain_alpha is None:
                    cv = RidgeCV(alphas=self.alphas, cv=self.cv_folds,
                                 fit_intercept=False)
                    cv.fit(Z, target)
                    gain_alpha = float(cv.alpha_)
                    gain_vec = cv.coef_.copy()
                else:
                    m = Ridge(alpha=gain_alpha, fit_intercept=False)
                    m.fit(Z, target)
                    gain_vec = m.coef_.copy()

            # ---- bookkeeping -----------------------------------------------
            yhat = self._predict_from_state(blocks, gain_t, beta, gain_vec, b0)
            mse = float(np.mean((y - yhat) ** 2))
            history.append({"iter": it, "mse": mse,
                            "kernel_alpha": kernel_alpha,
                            "gain_alpha": gain_alpha})
            if verbose:
                print(f"iter {it:3d}  mse={mse:.6g}  "
                      f"kernel_alpha={kernel_alpha:.3g}  "
                      f"gain_alpha={gain_alpha:.3g}")

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

        return beta, gain_vec, b0, history

    def fit(self, y: np.ndarray, trial_idx: np.ndarray, *,
            fit_mask: Optional[np.ndarray] = None,
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

        Convergence: every value gain coef changes by <= `tol` for `patience`
        consecutive iterations, or `max_iter` reached.
        """
        y = np.asarray(y, dtype=float).ravel()
        T = y.size
        trial_idx = np.asarray(trial_idx, dtype=int).ravel()
        if trial_idx.size != T:
            raise ValueError("trial_idx must have the same length as y")

        series = self._input_series(T, overrides=self._with_history(None, y))
        blocks = self._design_blocks(series)
        gain_t = self._gain_per_t(T, trial_idx)

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

    def cross_val_score(self, y: np.ndarray, trial_idx: np.ndarray, *,
                        fit_mask: Optional[np.ndarray] = None,
                        gap_history=False,
                        n_folds: int = 5, max_iter: int = 100,
                        tol: float = 1e-3, patience: int = 3,
                        verbose: bool = False) -> float:
        """Trial-held-out cross-validated R².

        `y` is the (T,) target; `trial_idx` is the (T,) bin-to-trial map used
        to define the folds (see `BilinearGLM` class docstring).

        Trials are partitioned into `n_folds` contiguous groups. For each fold
        the model is refit from scratch on the remaining trials; the full
        convolution is computed over all time bins so that kernel windows that
        span trial boundaries are handled correctly, but only rows belonging to
        training trials enter the ALS solver.

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
        y = np.asarray(y, dtype=float).ravel()
        T = y.size
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

        if gap_history:
            if self._history_predictor_name is None:
                raise ValueError(
                    "gap_history requires spike_history to be configured "
                    "on the model")
            if gap_history is True:
                gap_bins = int(self._lags[self._history_predictor_name].max())
            else:
                gap_bins = int(gap_history)
        else:
            gap_bins = 0

        # build design blocks once using all time bins (convolution is global)
        series = self._input_series(T, overrides=self._with_history(None, y))
        blocks = self._design_blocks(series)
        gain_t = self._gain_per_t(T, trial_idx)

        trials = np.unique(trial_idx)
        folds = np.array_split(trials, n_folds)

        cv_scores = []
        for fold_i, test_trials in enumerate(folds):
            train_trials = np.setdiff1d(trials, test_trials)
            train_in_fold = np.isin(trial_idx, train_trials)
            test_in_fold = np.isin(trial_idx, test_trials)
            if gap_bins > 0:
                train_in_fold = self._safe_under_history(train_in_fold,
                                                         gap_bins)
                test_in_fold = self._safe_under_history(test_in_fold,
                                                        gap_bins)
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

            beta, gain_vec, b0, _ = self._fit_als(
                y_tr, blocks_tr, gain_t_tr,
                max_iter=max_iter, tol=tol, patience=patience, verbose=verbose,
            )

            y_te = y[test_mask]
            blocks_te = {k: v[test_mask] for k, v in blocks.items()}
            gain_t_te = {k: v[test_mask] for k, v in gain_t.items()}
            yhat_te = self._predict_from_state(
                blocks_te, gain_t_te, beta, gain_vec, b0)

            ss_res = float(np.sum((y_te - yhat_te) ** 2))
            ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            cv_scores.append(r2)
            if verbose:
                print(f"  fold {fold_i + 1} R²={r2:.4f}")

        self.cv_scores_ = np.array(cv_scores)
        return float(np.nanmean(self.cv_scores_))

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
        if type(p_new) is not type(p_orig):
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
            if isinstance(p_new, EventPredictor):
                overrides[p_new.name] = p_new.times
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

    # ----- nested model comparison -----------------------------------------

    def delta_r2(self, y: np.ndarray, trial_idx: np.ndarray, *,
                 remove_gains: Sequence[str] = (),
                 remove_predictors: Sequence[str] = (),
                 fit_mask: Optional[np.ndarray] = None) -> float:
        """ΔR² = R²(full) - R²(reduced).

        Reduced model zeros out specified value-gain coefficients (for every
        predictor they modulate) and/or whole predictor blocks. Kernels and
        gain offsets are not refit; this matches the paper's "kernels held at
        the final iteration; only the targeted gains are removed" comparison
        when used after fit().

        If `fit_mask` is given (or one was stored at fit time), R² is computed
        only over masked bins.
        """
        if self.beta_ is None:
            raise RuntimeError("call fit() first")
        if fit_mask is None:
            fit_mask = self.fit_mask_

        y_arr = np.asarray(y, dtype=float).ravel()
        T = y_arr.size
        full_yhat = self.predict_fitted(y_arr, trial_idx)
        beta = self.beta_.copy()
        gain = self.gain_.copy()
        for vname in remove_gains:
            for p in self._modulated:
                key = (vname, p.name)
                if key in self._gain_var_idx:
                    gain[self._gain_var_idx[key]] = 0.0
        for pname in remove_predictors:
            k = self._pred_idx[pname]
            beta[self._beta_off[k]:self._beta_off[k + 1]] = 0.0
        # build reduced prediction
        series = self._input_series(T,
                                    overrides=self._with_history(None, y_arr))
        blocks = self._design_blocks(series)
        gain_t = self._gain_per_t(T, trial_idx)
        red_yhat = self._predict_from_state(blocks, gain_t, beta, gain,
                                             self.intercept_)
        if fit_mask is not None:
            m = np.asarray(fit_mask, dtype=bool).ravel()
            if m.size != T:
                raise ValueError(
                    f"fit_mask must have length T={T}, got {m.size}")
            y_arr = y_arr[m]
            full_yhat = full_yhat[m]
            red_yhat = red_yhat[m]
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        r2_full = 1.0 - float(np.sum((y_arr - full_yhat) ** 2)) / ss_tot
        r2_red = 1.0 - float(np.sum((y_arr - red_yhat) ** 2)) / ss_tot
        return r2_full - r2_red

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
        ΔR². P-value is the fraction of nulls with ΔR² >= observed.

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

        gain_alpha = (self.history_[-1]["gain_alpha"]
                      if self.history_ else 1.0)
        rm = Ridge(alpha=gain_alpha, fit_intercept=False)
        rm.fit(Z_full_fit, target_fit)
        full_pred_fit = offset_fit + Z_full_fit @ rm.coef_
        observed = self._delta_r2_remove(
            y_fit, full_pred_fit, Z_full_fit, rm.coef_,
            target_fit, gain_alpha, gain_name)

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
            m2 = Ridge(alpha=gain_alpha, fit_intercept=False)
            m2.fit(Z_null_fit, target_fit)
            nulls[i] = self._delta_r2_remove(
                y_fit, offset_fit + Z_null_fit @ m2.coef_,
                Z_null_fit, m2.coef_, target_fit, gain_alpha, gain_name)
        p = float(np.mean(nulls >= observed))
        return {"observed": float(observed), "null": nulls, "p": p}

    def _delta_r2_remove(self, y, full_yhat, Z, coef, target, alpha,
                          gain_name):
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0:
            return float("nan")
        r2_full = 1.0 - float(np.sum((y - full_yhat) ** 2)) / ss_tot
        # reduced: zero out every column corresponding to gain_name
        keep = np.ones(Z.shape[1], dtype=bool)
        for p in self._modulated:
            key = (gain_name, p.name)
            if key in self._gain_var_idx:
                keep[self._gain_var_idx[key]] = False
        if keep.all():
            return 0.0
        Z_red = Z[:, keep]
        m = Ridge(alpha=alpha, fit_intercept=False)
        m.fit(Z_red, target)
        red_yhat = (y - target) + Z_red @ m.coef_
        r2_red = 1.0 - float(np.sum((y - red_yhat) ** 2)) / ss_tot
        return r2_full - r2_red
