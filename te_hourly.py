# -*- coding: utf-8 -*-
"""Timestamp-aware transfer-entropy test for hourly traffic series.

The GEANT file has gaps (7 h, 4.8 days, 33 h).  Taking every 4th 15-min row
and removing an hour-of-week profile by row index puts the weekly phase out
of step after each gap.  Here the series are resampled on the clock, missing
hours stay missing, the hour-of-week profile is taken from the timestamps,
and only transitions between consecutive observed hours enter the estimate.
"""
import numpy as np
import pandas as pd
import tmcore as tc


def hourly_node_series(ts, X, n):
    """15-min OD matrix -> hourly node outgoing traffic on the clock (log1p).
    Hours without any 15-min sample are NaN."""
    out = X.reshape(len(ts), n, n).sum(2)
    df = pd.DataFrame(out, index=pd.DatetimeIndex(ts))
    H = df.resample('1h').mean() * 4
    return H.index, np.log1p(H.to_numpy())


def residuals(idx, S, how=('week',), trend_hours=168):
    """Remove the hour-of-week median (from timestamps) and, optionally,
    a centred moving average of `trend_hours` (slow common drift)."""
    R = S.copy()
    if 'week' in how:
        how_key = idx.dayofweek * 24 + idx.hour
        for k in np.unique(how_key):
            m = how_key == k
            R[m] -= np.nanmedian(S[m], 0)
    if 'trend' in how:
        df = pd.DataFrame(R)
        R = R - df.rolling(trend_hours, center=True, min_periods=trend_hours // 2).mean().to_numpy()
    return R


def digitize_nan(v, bins):
    d = np.full(v.shape, -1)
    ok = np.isfinite(v)
    d[ok] = tc.digitize(v[ok], bins)
    return d


def _H(cols, bins):
    idx = np.ravel_multi_index(cols, [b for b in bins])
    c = np.bincount(idx, minlength=int(np.prod(bins))).astype(float)
    p = c[c > 0] / c.sum()
    return -np.sum(p * np.log(p))


def te_masked(dx, dy, bins, dz=None, zbins=None):
    """TE x->y (history 1), optionally conditioned on z at the same lag.
    Uses only t with x_t, y_t, y_{t+1} (and z_t) observed."""
    ok = (dx[:-1] >= 0) & (dy[:-1] >= 0) & (dy[1:] >= 0)
    if dz is not None:
        ok &= dz[:-1] >= 0
    y1, y0, x0 = dy[1:][ok], dy[:-1][ok], dx[:-1][ok]
    if dz is None:
        b = (bins, bins)
        return _H((y1, y0), b) + _H((y0, x0), b) - _H((y1, y0, x0), (bins,) * 3) - _H((y0,), (bins,))
    z0 = dz[:-1][ok]
    return (_H((y1, y0, z0), (bins, bins, zbins)) + _H((y0, x0, z0), (bins, bins, zbins))
            - _H((y1, y0, x0, z0), (bins, bins, bins, zbins)) - _H((y0, z0), (bins, zbins)))


def te_test(R, bins=4, n_surr=200, min_shift=168, q=0.05, pval='z', condition_global=False,
            zbins=3, seed=7):
    """All ordered pairs; per-pair circular-shift surrogates of the source;
    BH-FDR.  If condition_global, condition on the mean of the other nodes."""
    rng = np.random.default_rng(seed)
    T, n = R.shape
    D = np.stack([digitize_nan(R[:, i], bins) for i in range(n)], 1)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    p = np.ones(len(pairs))
    for k, (i, j) in enumerate(pairs):
        dz = None
        if condition_global:
            others = [c for c in range(n) if c not in (i, j)]
            dz = digitize_nan(np.nanmean(R[:, others], 1), zbins)
        obs = te_masked(D[:, i], D[:, j], bins, dz, zbins)
        sh = rng.integers(min_shift, T - min_shift, size=n_surr)
        null = np.array([te_masked(np.roll(D[:, i], s), D[:, j], bins, dz, zbins) for s in sh])
        p[k] = tc.z_pvalue(obs, null)[1] if pval == 'z' else (1 + np.sum(null >= obs)) / (n_surr + 1)
    sig = tc.bh_fdr(p, q)
    return {pr for pr, s in zip(pairs, sig) if s}, len(pairs)


def te_network_hourly(R, bins=4, n_surr=200, min_shift=168, q=0.05, seed=7):
    """Same test as te_test, returning z-score and significance matrices."""
    rng = np.random.default_rng(seed)
    T, n = R.shape
    D = np.stack([digitize_nan(R[:, i], bins) for i in range(n)], 1)
    Z = np.zeros((n, n)); P = np.ones((n, n))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    for i, j in pairs:
        obs = te_masked(D[:, i], D[:, j], bins)
        sh = rng.integers(min_shift, T - min_shift, size=n_surr)
        null = np.array([te_masked(np.roll(D[:, i], s), D[:, j], bins) for s in sh])
        Z[i, j], P[i, j] = tc.z_pvalue(obs, null)
    idx = tuple(np.array(pairs).T)
    sig = np.zeros((n, n), bool); sig[idx] = tc.bh_fdr(P[idx], q)
    return {'Z': Z, 'sig': sig, 'n_pairs': len(pairs), 'n_sig_fdr': int(sig.sum())}
