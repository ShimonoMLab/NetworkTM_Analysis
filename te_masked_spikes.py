# -*- coding: utf-8 -*-
"""Binary transfer entropy for spike trains when some time bins were not
received (packet loss), with the loss mask known to the estimator.

The transport layer knows which packets were lost (sequence numbers).  If
the estimator is told, it can drop every time step whose inputs fall in a
lost packet, instead of reading the gap as "no spike".  For TE(X->Y, d)
with history 1, the step t uses y_{t+1}, y_t and x_{t+1-d}; it is kept
only if bins t+1, t and t+1-d were all received.

Counts are formed from spike times as in tmcore.te_spike_delays, with each
count restricted to valid steps.  A spike can only exist in a received bin,
so validity has to be checked only for the other bins involved.  With no
loss the result equals tmcore.te_spike_delays (checked in selftest()).
"""
import numpy as np
import tmcore as tc


def _lag_pairs(Y, X, lag):
    """For each spike time y in Y, count X spikes at y - lag."""
    return np.isin(Y - lag, X, assume_unique=False)


def valid_counts(V, delays):
    """Number of valid steps for each delay (bins t+1, t, t+1-d received)."""
    T = V.size
    return {d: int((V[d:T] & V[d - 1:T - 1] & V[0:T - d]).sum()) for d in delays}


def _lag_counts3(Y, X, maxlag, w1, w2, w3):
    """Like tmcore._lag_counts, with three weight vectors at once."""
    lo = np.searchsorted(X, Y - maxlag, 'left'); hi = np.searchsorted(X, Y, 'right')
    c1 = np.zeros(maxlag + 1); c2 = np.zeros(maxlag + 1); c3 = np.zeros(maxlag + 1)
    width = int((hi - lo).max()) if Y.size else 0
    for k in range(width):
        idx = lo + k; ok = idx < hi
        lag = Y[ok] - X[idx[ok]]
        c1 += np.bincount(lag, weights=w1[ok], minlength=maxlag + 1)[:maxlag + 1]
        c2 += np.bincount(lag, weights=w2[ok], minlength=maxlag + 1)[:maxlag + 1]
        c3 += np.bincount(lag, weights=w3[ok], minlength=maxlag + 1)[:maxlag + 1]
    return c1, c2, c3


def _vat(V, pos):
    """V at positions pos, False outside the array."""
    out = np.zeros(pos.shape, bool)
    m = (pos >= 0) & (pos < V.size)
    out[m] = V[pos[m]]
    return out


def target_terms(Y, V, delays):
    """Quantities that depend only on the target (reused for all sources)."""
    d = np.asarray(delays)
    yprev = np.isin(Y - 1, Y)
    vY1 = _vat(V, Y - 1) & (Y >= 1)                 # step t = Y-1 needs bin t received
    a = (vY1[:, None] & _vat(V, Y[:, None] - d[None, :]) & (Y[:, None] - d[None, :] >= 0)).sum(0)
    vYp1 = _vat(V, Y + 1)                             # step t = Y needs bin t+1 received
    b = (vYp1[:, None] & _vat(V, Y[:, None] + 1 - d[None, :]) & (Y[:, None] + 1 - d[None, :] >= 0)).sum(0)
    ab = (yprev & _vat(V, Y[:, None] - d[None, :]).T).sum(1) if False else (yprev[:, None] & _vat(V, Y[:, None] - d[None, :])).sum(0)
    return {'yprev': yprev, 'vY1': vY1, 'vYp1': vYp1, 'a': a, 'b': b, 'ab': ab,
            'w1': vY1.astype(float), 'w2': yprev.astype(float), 'w3': vYp1.astype(float)}


def te_spike_delays_masked(Y, X, V, delays, Ncache=None, tt=None):
    """Y, X: sorted unique spike bins of target and source (only in received
    bins).  V: boolean array over bins, True = received."""
    delays = list(delays); d = np.asarray(delays); maxd = int(d.max())
    if Ncache is None: Ncache = valid_counts(V, delays)
    if tt is None: tt = target_terms(Y, V, delays)
    c = (_vat(V, X[:, None] + d[None, :]) & _vat(V, X[:, None] + d[None, :] - 1)).sum(0)
    ac_l, abc_l, bc_l = _lag_counts3(Y, X, maxd, tt['w1'], tt['w2'], tt['w3'])
    te = np.zeros(len(delays))
    for k, dd in enumerate(delays):
        N = Ncache[dd]
        a, b, ab = tt['a'][k], tt['b'][k], tt['ab'][k]
        ac, abc, bc = ac_l[dd], abc_l[dd], bc_l[dd - 1]
        n111 = abc; n110 = ab - abc; n101 = ac - abc; n011 = bc - abc
        n100 = a - n110 - n101 - n111; n010 = b - n110 - n011 - n111
        n001 = c[k] - n101 - n011 - n111
        n000 = N - (n111 + n110 + n101 + n011 + n100 + n010 + n001)
        cnt = np.maximum(np.array([[[n000, n001], [n010, n011]], [[n100, n101], [n110, n111]]], float), 0)
        if cnt.sum() == 0: continue
        p = cnt / cnt.sum()
        p_y0x0 = p.sum(0, keepdims=True); p_y1y0 = p.sum(2, keepdims=True); p_y0 = p.sum((0, 2), keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = (p * p_y0) / (p_y0x0 * p_y1y0)
            te[k] = np.nansum(np.where(p > 0, p * np.log2(r), 0.0))
    return te


def te_network_masked(spikes, V, delays=range(1, 31), n_surr=50, min_shift=1000, q=0.05, seed=7):
    """As tmcore.te_network_spikes, but every count is restricted to received
    bins.  Surrogate sources are circularly shifted and then thinned to the
    received bins, as they would have been observed."""
    rng = np.random.default_rng(seed)
    delays = list(delays); T = V.size; n = len(spikes)
    S = [np.unique(np.asarray(s, np.int64)) for s in spikes]
    S = [s[V[s]] for s in S]
    TEpk = np.zeros((n, n)); Z = np.zeros((n, n)); P = np.ones((n, n)); NULL = np.zeros((n, n, n_surr))
    Nc = valid_counts(V, delays)
    surr = {}
    for i in range(n):
        sh = rng.integers(min_shift, T - min_shift, size=n_surr)
        surr[i] = [np.sort(((S[i] + s) % T)[V[(S[i] + s) % T]]) for s in sh]
    for j in range(n):
        tt = target_terms(S[j], V, delays)
        for i in range(n):
            if i == j: continue
            te = te_spike_delays_masked(S[j], S[i], V, delays, Nc, tt)
            k = int(np.argmax(te)); TEpk[i, j] = te[k]
            null = np.array([te_spike_delays_masked(S[j], Xs, V, delays, Nc, tt).max() for Xs in surr[i]])
            Z[i, j], P[i, j] = tc.z_pvalue(te[k], null); NULL[i, j] = null
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = tuple(np.array(pairs).T)
    sig = np.zeros((n, n), bool); sig[idx] = tc.bh_fdr(P[idx], q)
    return {'TE_peak': TEpk, 'Z': Z, 'P_z': P, 'sig': sig, 'NULL': NULL, 'n_pairs': len(pairs), 'n_sig_fdr': int(sig.sum())}


def selftest(seed=0):
    """Masked TE with nothing lost equals tmcore.te_spike_delays; and with a
    random mask it equals a dense brute-force count."""
    rng = np.random.default_rng(seed)
    T = 20000
    X = np.sort(rng.choice(T, 600, replace=False))
    Y = np.unique(np.concatenate([X + 3, rng.choice(T - 5, 500, replace=False)]))
    Y = Y[Y < T]
    V = np.ones(T, bool); delays = list(range(1, 11))
    a = te_spike_delays_masked(Y, X, V, delays)
    b = tc.te_spike_delays(Y, np.isin(Y - 1, Y), X, T, delays)
    V2 = np.repeat(rng.random(T // 10) > 0.2, 10)
    Xm, Ym = X[V2[X]], Y[V2[Y]]
    c = te_spike_delays_masked(Ym, Xm, V2, delays)
    x = np.zeros(T, int); x[Xm] = 1; y = np.zeros(T, int); y[Ym] = 1
    dd = []
    for d in delays:
        t = np.arange(d - 1, T - 1); W = V2[t + 1] & V2[t] & V2[t + 1 - d]; t = t[W]
        cnt = np.zeros((2, 2, 2)); np.add.at(cnt, (y[t + 1], y[t], x[t + 1 - d]), 1)
        p = cnt / cnt.sum(); p_y0x0 = p.sum(0, keepdims=True); p_y1y0 = p.sum(2, keepdims=True); p_y0 = p.sum((0, 2), keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd.append(np.nansum(np.where(p > 0, p * np.log2((p * p_y0) / (p_y0x0 * p_y1y0)), 0)))
    return float(np.max(np.abs(a - b))), float(np.max(np.abs(c - np.array(dd))))


if __name__ == '__main__':
    print('max |masked(no loss) - tmcore| , max |masked - dense| :', selftest())
