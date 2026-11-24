"""Does conditioning on the network mean remove true dependencies?

Synthetic series as in robustness_te.py, with two topologies:
  sparse : random sparse lag-1 couplings (density 0.08)
  hub    : the same, plus one source node driving 12 other nodes
and three coupling strengths.  Recall of direct links (all, and hub links
only) and false positives on pairs with no directed path are compared
with and without conditioning on the mean of the other nodes.
"""
import json
import numpy as np
import pandas as pd
import te_hourly as th
from robustness_te import reach


def synthetic(n=23, T=2849, density=0.08, strength=0.3, hub=False, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < density) & ~np.eye(n, dtype=bool)
    hub_links = set()
    if hub:
        tg = rng.choice(np.arange(1, n), 12, replace=False)
        A[0, tg] = True; hub_links = {(0, int(j)) for j in tg}
    rho = max(1.0, np.max(np.abs(np.linalg.eigvals(A.astype(float)))))
    M = 0.4 * np.eye(n) + (strength / rho) * A.astype(float)
    t = np.arange(T)
    cyc = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168)
    drift = np.cumsum(rng.normal(0, 0.05, T))
    amp = rng.uniform(0.5, 1.5, n); load = rng.uniform(0.2, 0.6, n)
    X = np.zeros((T, n)); e = rng.normal(0, 1, (T, n))
    for s in range(1, T):
        X[s] = X[s - 1] @ M + e[s]
    return X + np.outer(cyc, amp) + np.outer(drift, load), A, hub_links


def main():
    n, T = 23, 2849
    idx = pd.date_range('2005-01-01', periods=T, freq='1h')
    base = dict(n_surr=200, min_shift=168, q=0.05, pval='z', seed=7)
    out = []
    for topo in ('sparse', 'hub'):
        for strength in (0.1, 0.2, 0.3):
            for sd in range(2):
                Y, A, hl = synthetic(n, T, strength=strength, hub=(topo == 'hub'), seed=sd)
                R = th.residuals(idx, Y, ('week', 'trend'))
                Rr = reach(A)
                direct = {(i, j) for i in range(n) for j in range(n) if A[i, j]}
                nopath = {(i, j) for i in range(n) for j in range(n) if i != j and not Rr[i, j]}
                for name, ch in (('unconditioned, 4 bins', {'bins': 4}),
                                 ('unconditioned, 3 bins', {'bins': 3}),
                                 ('conditioned on network mean, 3 bins', {'bins': 3, 'condition_global': True})):
                    sig, m = th.te_test(R, **{**base, **ch})
                    row = {'topology': topo, 'strength': strength, 'seed': sd, 'setting': name,
                           'n_direct': len(direct), 'n_sig': len(sig),
                           'recall_direct': round(len(sig & direct) / len(direct), 3),
                           'recall_hub_links': round(len(sig & hl) / len(hl), 3) if hl else None,
                           'false_pos_rate_no_path': round(len(sig & nopath) / max(1, len(nopath)), 4)}
                    out.append(row); print(row, flush=True)
    json.dump(out, open('results_conditioning.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
