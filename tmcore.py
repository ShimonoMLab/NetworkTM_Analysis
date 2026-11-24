# -*- coding: utf-8 -*-
"""Core procedures shared by analysis.py (traffic matrices) and
analysis_spikes.py (neuronal spike trains).

Everything here works on a plain time x channel matrix, or on lists of
event times, so the same procedures apply to OD traffic series and to
multi-electrode recordings.
"""
import numpy as np
from math import erf, sqrt


# ------------------------------------------------------------ spectrum
def spectral_structure(M):
    s = _svd(M - M.mean(0))[1]
    e = np.cumsum(s**2) / np.sum(s**2)
    return {'sv': s, 'k90': int(np.searchsorted(e, 0.90)) + 1,
            'k95': int(np.searchsorted(e, 0.95)) + 1, 'top1_energy': float(e[0])}


# ------------------------------------------------------------ completion
def _svd(Z):
    try:
        return np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:          # LAPACK gesdd occasionally fails; fall back to gesvd
        import scipy.linalg
        return scipy.linalg.svd(Z, full_matrices=False, lapack_driver='gesvd')


def soft_impute(L, mask, lam_frac=0.01, iters=40):
    """Singular-value soft-thresholding (Mazumder, Hastie & Tibshirani 2010).
    Missing entries start from the per-channel mean; the threshold decreases
    geometrically from 0.3*s_max to lam_frac*s_max (warm starts)."""
    cm = np.nanmean(np.where(mask, L, np.nan), 0)
    Z = np.where(mask, L, np.nan_to_num(cm)[None, :])
    s0 = _svd(Z)[1][0]
    for lam in s0 * np.geomspace(0.3, lam_frac, iters):
        U, s, Vt = _svd(Z)
        Zl = (U * np.maximum(s - lam, 0)) @ Vt
        Z = np.where(mask, L, Zl)
    return Zl


def channel_mean(L, mask):
    cm = np.nanmean(np.where(mask, L, np.nan), 0)
    return np.broadcast_to(np.nan_to_num(cm), L.shape).copy()


def time_interp(L, mask):
    T, F = L.shape
    out = np.empty_like(L); t = np.arange(T)
    for f in range(F):
        m = mask[:, f]
        if m.sum() >= 2:
            out[:, f] = np.interp(t, t[m], L[m, f])
        else:
            out[:, f] = L[m, f].mean() if m.any() else 0.0
    return out


def periodic_profile(L, mask, period):
    out = np.empty_like(L)
    Lm = np.where(mask, L, np.nan)
    for k in range(period):
        prof = np.nanmedian(Lm[k::period], 0)
        out[k::period] = np.nan_to_num(prof, nan=np.nanmean(Lm))
    return out


def nmae(true, est):
    return float(np.sum(np.abs(true - est)) / np.sum(np.abs(true)))


def block_mask(rng, shape, p, blk=48):
    """Hide contiguous runs of `blk` time steps in randomly chosen channels
    until a fraction p of entries is hidden (True = observed)."""
    T, F = shape
    m = np.ones(shape, bool); need = int(p * T * F); hid = 0
    while hid < need:
        f = rng.integers(F); t0 = rng.integers(0, T - blk)
        hid += int(m[t0:t0 + blk, f].sum()); m[t0:t0 + blk, f] = False
    return m


def make_mask(rng, shape, p, scheme, blk=48):
    return (rng.random(shape) > p) if scheme == 'random' else block_mask(rng, shape, p, blk)


def choose_lambda(L, mask, rng, grid=(0.0003, 0.001, 0.003, 0.01)):
    """Pick the soft-impute threshold on a validation split drawn from the
    OBSERVED entries only (10%), so the hidden test entries are never used."""
    val = mask & (rng.random(L.shape) < 0.10)
    train = mask & ~val
    true = np.expm1(L[val])
    errs = {lam: nmae(true, np.expm1(np.maximum(soft_impute(L, train, lam)[val], 0))) for lam in grid}
    return min(errs, key=errs.get), errs


def completion_experiment(L, period=None, rates=(0.3, 0.5, 0.7, 0.9),
                          schemes=('random', 'block'), seeds=10, blk=48, log=print):
    """L: time x channel matrix on log1p scale. Hidden entries are scored
    after expm1, by NMAE. `period` (in samples) enables the periodic-profile
    baseline; None skips it (e.g. neural data without a daily cycle)."""
    rows = []
    for scheme in schemes:
        for p in rates:
            rng0 = np.random.default_rng(99)
            lam, _ = choose_lambda(L, make_mask(rng0, L.shape, p, scheme, blk), rng0)
            methods = {'soft_impute': lambda L, m: soft_impute(L, m, lam),
                       'time_interp': time_interp}
            if period:
                methods['tod_profile'] = lambda L, m: periodic_profile(L, m, period)
            methods['flow_mean'] = channel_mean
            per = {k: [] for k in methods}
            for sd in range(seeds):
                rng = np.random.default_rng(1000 + sd)
                mask = make_mask(rng, L.shape, p, scheme, blk)
                true = np.expm1(L[~mask])
                for k, fn in methods.items():
                    per[k].append(nmae(true, np.expm1(np.maximum(fn(L, mask)[~mask], 0))))
            row = {'scheme': scheme, 'missing': p, 'n_masks': seeds, 'soft_impute_lambda_frac': lam}
            for k, v in per.items():
                row[k] = {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1))}
            rows.append(row)
            log(f"  {scheme:6s} {p:.1f} (lam {lam}): " +
                ", ".join(f"{k} {row[k]['mean']:.3f}±{row[k]['sd']:.3f}" for k in methods))
    return rows


# ------------------------------------------------------------ statistics
def bh_fdr(p, q=0.05):
    p = np.asarray(p); m = p.size
    order = np.argsort(p); ranked = p[order]
    below = np.nonzero(ranked <= q * np.arange(1, m + 1) / m)[0]
    sig = np.zeros(m, bool)
    if below.size:
        sig[order[:below[-1] + 1]] = True
    return sig


def z_pvalue(obs, null):
    z = (obs - null.mean()) / (null.std(ddof=1) + 1e-12)
    return z, 0.5 * (1 - erf(z / sqrt(2)))


# ------------------------------------------------------------ TE, continuous series
def digitize(v, bins=4):
    q = np.quantile(v, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(q, v, 'right') - 1, 0, bins - 1)


def te_from_digits(dx, dy, bins=4):
    """TE x->y, history length 1, from pre-binned series."""
    y1, y0, x0 = dy[1:], dy[:-1], dx[:-1]
    def H(*a):
        idx = np.ravel_multi_index(a, [bins] * len(a))
        c = np.bincount(idx, minlength=bins ** len(a)).astype(float)
        p = c[c > 0] / c.sum()
        return -np.sum(p * np.log(p))
    return H(y1, y0) + H(y0, x0) - H(y1, y0, x0) - H(y0)


def te_network_binned(S, remove_period=None, min_shift=1, n_surr=200, q=0.05, seed=7, bins=4):
    """Directed dependence between the columns of S (time x channel).
    Optional removal of a periodic profile, quantile binning, TE with
    history 1, per-pair circular-shift surrogates (shift >= min_shift),
    one-sided z-test, BH-FDR."""
    rng = np.random.default_rng(seed)
    Tq, n = S.shape
    R = S.copy()
    if remove_period:
        for k in range(remove_period):
            R[k::remove_period] -= np.median(S[k::remove_period], 0)
    D = np.stack([digitize(R[:, i], bins) for i in range(n)], 1)
    lo = min_shift
    TE = np.zeros((n, n)); Z = np.zeros_like(TE); P_emp = np.ones_like(TE); P_z = np.ones_like(TE)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    for i, j in pairs:
        obs = te_from_digits(D[:, i], D[:, j], bins)
        shifts = rng.integers(lo, Tq - lo, size=n_surr)
        null = np.array([te_from_digits(np.roll(D[:, i], s), D[:, j], bins) for s in shifts])
        TE[i, j] = obs
        Z[i, j], P_z[i, j] = z_pvalue(obs, null)
        P_emp[i, j] = (1 + np.sum(null >= obs)) / (n_surr + 1)
    idx = tuple(np.array(pairs).T)
    sig = np.zeros_like(TE, bool); sig[idx] = bh_fdr(P_z[idx], q)
    pooled = []
    for _ in range(n_surr):                 # earlier pooled-threshold rule, for comparison only
        i, j = rng.integers(n, size=2)
        if i != j:
            pooled.append(te_from_digits(np.roll(D[:, i], rng.integers(lo, Tq - lo)), D[:, j], bins))
    pooled_thr = float(np.quantile(pooled, 0.99))
    return {'TE': TE, 'Z': Z, 'P_emp': P_emp, 'P_z': P_z, 'sig': sig,
            'n_pairs': len(pairs), 'n_sig_fdr': int(sig.sum()),
            'n_emp_min_p': int(np.sum(P_emp[idx] <= 1 / (n_surr + 1))),
            'n_above_pooled_thr': int(np.sum(TE[idx] > pooled_thr)), 'pooled_thr': pooled_thr}


# ------------------------------------------------------------ TE, spike trains
def _lag_counts(Y, X, maxlag, weight):
    """For every spike s in Y, count X spikes at s-L, L = 0..maxlag.
    Returns (counts per lag, counts weighted by `weight` per Y spike)."""
    lo = np.searchsorted(X, Y - maxlag, 'left')
    hi = np.searchsorted(X, Y, 'right')
    c = np.zeros(maxlag + 1); cw = np.zeros(maxlag + 1)
    width = int((hi - lo).max()) if Y.size else 0
    for k in range(width):
        idx = lo + k
        ok = idx < hi
        lag = Y[ok] - X[idx[ok]]
        c += np.bincount(lag, minlength=maxlag + 1)[:maxlag + 1]
        cw += np.bincount(lag, weights=weight[ok], minlength=maxlag + 1)[:maxlag + 1]
    return c, cw


def te_spike_delays(Y, Yprev, X, T, delays):
    """Binary transfer entropy X->Y (history 1 for both, 1-bin resolution)
    for each delay d: TE(d) uses y_{t+1}, y_t and x_{t+1-d}, as in the
    delayed TE of Ito et al. (2011). Y, X: sorted spike bins; Yprev:
    boolean, whether Y[k]-1 is also a spike of Y. Returns TE in bits."""
    maxd = int(max(delays))
    nY = Y.size; nX = X.size
    c_ac, c_abc = _lag_counts(Y, X, maxd, Yprev.astype(float))   # y_{t+1}=1 & x_{t+1-d}=1 (lag d); and with y_t=1
    n_ab = float(Yprev.sum())                      # y_{t+1}=1 & y_t=1
    te = np.zeros(len(delays))
    for k, d in enumerate(delays):
        N = T - d
        a, b, c = nY, nY, nX                       # marginal counts (edge effects ignored)
        ab, ac, bc, abc = n_ab, c_ac[d], c_ac[d - 1], c_abc[d]
        # joint counts of (y_{t+1}, y_t, x_{t+1-d}) by inclusion-exclusion
        n111 = abc; n110 = ab - abc; n101 = ac - abc; n011 = bc - abc
        n100 = a - n110 - n101 - n111; n010 = b - n110 - n011 - n111
        n001 = c - n101 - n011 - n111
        n000 = N - (n111 + n110 + n101 + n011 + n100 + n010 + n001)
        cnt = np.array([[[n000, n001], [n010, n011]], [[n100, n101], [n110, n111]]], float)
        cnt = np.maximum(cnt, 0)
        p = cnt / cnt.sum()                        # p[y1, y0, x0]
        p_y0x0 = p.sum(0, keepdims=True); p_y1y0 = p.sum(2, keepdims=True); p_y0 = p.sum((0, 2), keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = (p * p_y0) / (p_y0x0 * p_y1y0)
            te[k] = np.nansum(np.where(p > 0, p * np.log2(r), 0.0))
    return te


def te_network_spikes(spikes, T, delays=range(1, 31), n_surr=100, min_shift=1000, q=0.05, seed=7, log=print):
    """Peak-over-delay binary TE between all ordered pairs of spike trains,
    with per-pair circular-shift surrogates of the source (shift >= min_shift
    bins). The test statistic is the peak over delays for both the observed
    data and every surrogate, so scanning delays does not inflate the test."""
    rng = np.random.default_rng(seed)
    delays = list(delays)
    n = len(spikes)
    S = [np.unique(np.asarray(s, np.int64)) for s in spikes]
    prev = [np.isin(s - 1, s, assume_unique=True) for s in S]
    TEpk = np.zeros((n, n)); Dpk = np.zeros((n, n), int); Z = np.zeros((n, n)); P = np.ones((n, n))
    NULL = np.zeros((n, n, n_surr))
    surr = {}
    for i in range(n):                              # shifted copies of each source, reused for all targets
        sh = rng.integers(min_shift, T - min_shift, size=n_surr)
        surr[i] = [np.sort((S[i] + s) % T) for s in sh]
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            te = te_spike_delays(S[j], prev[j], S[i], T, delays)
            k = int(np.argmax(te)); TEpk[i, j] = te[k]; Dpk[i, j] = delays[k]
            null = np.array([te_spike_delays(S[j], prev[j], Xs, T, delays).max() for Xs in surr[i]])
            Z[i, j], P[i, j] = z_pvalue(te[k], null); NULL[i, j] = null
        log(f"    target {j + 1}/{n} done")
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = tuple(np.array(pairs).T)
    sig = np.zeros((n, n), bool); sig[idx] = bh_fdr(P[idx], q)
    return {'TE_peak': TEpk, 'delay_peak': Dpk, 'Z': Z, 'P_z': P, 'sig': sig, 'NULL': NULL,
            'n_pairs': len(pairs), 'n_sig_fdr': int(sig.sum())}


def pooled_null_p(TE, NULL, mask=None):
    """p-values from a pooled null.  Each pair's observed peak TE is
    standardized by its own surrogates; every surrogate is standardized by the
    other surrogates of the same pair (leave-one-out).  The pooled standardized
    surrogates of all pairs form one null distribution, which keeps its skewed
    tail instead of assuming a normal one.  mask: pairs to use (default: off-diagonal)."""
    n = TE.shape[0]; K = NULL.shape[-1]
    if mask is None: mask = ~np.eye(n, dtype=bool)
    m = NULL.mean(-1); sd = NULL.std(-1, ddof=1) + 1e-12
    z_obs = (TE - m) / sd
    tot = NULL.sum(-1, keepdims=True); tot2 = (NULL ** 2).sum(-1, keepdims=True)
    m_l = (tot - NULL) / (K - 1)
    v_l = ((tot2 - NULL ** 2) - (K - 1) * m_l ** 2) / (K - 2)
    z_null = (NULL - m_l) / (np.sqrt(np.maximum(v_l, 0)) + 1e-12)
    pool = np.sort(z_null[mask].ravel())
    P = np.ones(TE.shape)
    P[mask] = (1 + pool.size - np.searchsorted(pool, z_obs[mask], 'left')) / (1 + pool.size)
    return P, z_obs


def sig_from_p(P, q=0.05, mask=None):
    n = P.shape[0]
    if mask is None: mask = ~np.eye(n, dtype=bool)
    sig = np.zeros((n, n), bool); sig[mask] = bh_fdr(P[mask], q)
    return sig


# ------------------------------------------------------------ anomaly candidates
def anomaly_candidates(L, period=None, z_thr=6.0):
    """Residuals after removing a periodic profile (or the channel median
    when period is None); returns (time, channel) hits sorted by z."""
    med = np.zeros_like(L)
    if period:
        for k in range(period):
            med[k::period] = np.median(L[k::period], 0)
    else:
        med[:] = np.median(L, 0)
    R = L - med
    z = (R - R.mean(0)) / (R.std(0) + 1e-9)
    hits = np.argwhere(z > z_thr)
    order = np.argsort(-z[hits[:, 0], hits[:, 1]])
    return hits[order], z
