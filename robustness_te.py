"""Robustness of the transfer-entropy significance test.

GEANT node-level outgoing traffic, resampled hourly on the clock
(te_hourly.py).  One setting is changed at a time; the significant set is
compared with the baseline (Jaccard index).  The same test is then applied
to synthetic series with KNOWN directed couplings, a shared daily/weekly
cycle and a shared slow drift.
"""
import argparse, json, time
import numpy as np
import pandas as pd
from analysis import load_tms
import te_hourly as th


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def synthetic(n=23, T=2849, density=0.08, strength=0.3, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density) & ~np.eye(n, dtype=bool)          # A[i, j]: i -> j
    rho = max(1.0, np.max(np.abs(np.linalg.eigvals(A.astype(float)))))
    M = 0.4 * np.eye(n) + (strength / rho) * A.astype(float)
    t = np.arange(T)
    cyc = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168)
    drift = np.cumsum(rng.normal(0, 0.05, T))
    amp = rng.uniform(0.5, 1.5, n); load = rng.uniform(0.2, 0.6, n)
    X = np.zeros((T, n)); e = rng.normal(0, 1, (T, n))
    for s in range(1, T):
        X[s] = X[s - 1] @ M + e[s]
    return X + np.outer(cyc, amp) + np.outer(drift, load), A


def reach(A):
    R = A.copy()
    for _ in range(A.shape[0]):
        R2 = R | ((R.astype(int) @ A.astype(int)) > 0)
        if (R2 == R).all(): break
        R = R2
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--out', default='results_robustness.json')
    a = ap.parse_args()
    ts, X = load_tms(a.csv)
    n = int(round(X.shape[1] ** 0.5))
    idx, S = th.hourly_node_series(ts, X, n)
    print(f'hourly grid {len(idx)} h, missing hours {int(np.isnan(S[:, 0]).sum())}', flush=True)
    R_w = th.residuals(idx, S, ('week',))
    R_wt = th.residuals(idx, S, ('week', 'trend'))
    base = dict(bins=4, n_surr=200, min_shift=168, q=0.05, pval='z', condition_global=False, seed=7)
    variants = [('baseline: hour-of-week removed', R_w, {}),
                ('1000 surrogates', R_w, {'n_surr': 1000}),
                ('shift >= 14 d', R_w, {'min_shift': 336}),
                ('another seed', R_w, {'seed': 11}),
                ('empirical p, 1000 surrogates', R_w, {'pval': 'emp', 'n_surr': 1000}),
                ('FDR q = 0.01', R_w, {'q': 0.01}),
                ('3 bins', R_w, {'bins': 3}),
                ('6 bins', R_w, {'bins': 6}),
                ('+ 7-day moving average removed', R_wt, {}),
                ('3 bins, unconditioned (reference)', R_w, {'bins': 3}),
                ('3 bins, conditioned on network mean', R_w, {'bins': 3, 'condition_global': True}),
                ('+ trend removed, 3 bins, unconditioned', R_wt, {'bins': 3}),
                ('+ trend removed, 3 bins, conditioned on network mean', R_wt, {'bins': 3, 'condition_global': True})]
    out = {'geant': [], 'synthetic': [], 'grid_hours': len(idx), 'missing_hours': int(np.isnan(S[:, 0]).sum())}
    ref = None
    for name, R, ch in variants:
        t0 = time.time()
        sig, m = th.te_test(R, **{**base, **ch})
        if ref is None: ref = sig
        row = {'setting': name, 'n_sig': len(sig), 'n_pairs': m, 'fraction': round(len(sig) / m, 3),
               'jaccard_vs_baseline': round(jaccard(sig, ref), 3)}
        out['geant'].append(row); print(row, f'{time.time() - t0:.0f}s', flush=True)
    syn_idx = pd.date_range('2005-01-01', periods=len(idx), freq='1h')
    for strength in (0.0, 0.3):
        for sd in range(5):
            Y, A = synthetic(n=n, T=len(idx), strength=strength, seed=sd)
            if strength == 0.0: A = np.zeros_like(A)
            Rr = reach(A)
            direct = {(i, j) for i in range(n) for j in range(n) if A[i, j]}
            indirect = {(i, j) for i in range(n) for j in range(n) if Rr[i, j] and not A[i, j] and i != j}
            nopath = {(i, j) for i in range(n) for j in range(n) if i != j and not Rr[i, j]}
            Yw = th.residuals(syn_idx, Y, ('week',)); Ywt = th.residuals(syn_idx, Y, ('week', 'trend'))
            for name, R, ch in (('hour-of-week removed', Yw, {}),
                                ('+ trend removed', Ywt, {}),
                                ('hour-of-week removed, conditioned on network mean (3 bins)', Yw, {'bins': 3, 'condition_global': True}),
                                ('+ trend removed, conditioned on network mean (3 bins)', Ywt, {'bins': 3, 'condition_global': True})):
                sig, m = th.te_test(R, **{**base, **ch})
                row = {'coupling_strength': strength, 'seed': sd, 'setting': name,
                       'n_direct': len(direct), 'n_indirect': len(indirect), 'n_no_path': len(nopath),
                       'n_sig': len(sig),
                       'false_pos_rate_no_path': round(len(sig & nopath) / max(1, len(nopath)), 4),
                       'recall_direct': round(len(sig & direct) / len(direct), 3) if direct else None,
                       'detected_indirect': round(len(sig & indirect) / len(indirect), 3) if indirect else None}
                out['synthetic'].append(row); print(row, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
