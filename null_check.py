"""Calibration of the spike-train TE test.  The 40 neurons of one 30-min
segment and the same 40 neurons of another segment are put side by side
(80 trains on a common 30-min clock).  Pairs that cross the two segments
cannot be coupled, so significant crossing pairs are false positives of the
whole procedure (z-test against 50 circular-shift surrogates, BH-FDR over all
pairs).  Also reports the agreement between the two segments' own networks.

  python null_check.py spike.mat --a 0 --b 1
  python null_check.py --from-npz null_01.npz   # pooled-null metrics from a saved run
"""
import argparse, json, sys
import numpy as np
import tmcore as tc
from loss_decompose import load, window, SEG, N_SURR

def pooled_metrics(TE, NULL, n):
    """The same crossing pairs judged with the pooled null (tmcore.pooled_null_p)."""
    N = 2 * n; cross = np.zeros((N, N), bool); cross[:n, n:] = True; cross[n:, :n] = True
    P, _ = tc.pooled_null_p(TE, NULL); sig = tc.sig_from_p(P); cp = P[cross]
    return {'pooled_cross_frac_p_below_0.05': round(float(np.mean(cp < 0.05)), 4),
            'pooled_cross_frac_p_below_0.01': round(float(np.mean(cp < 0.01)), 4),
            'pooled_cross_frac_p_below_0.001': round(float(np.mean(cp < 0.001)), 4),
            'pooled_cross_significant_fdr': int(sig[cross].sum()),
            'pooled_within_a_significant': int(sig[:n, :n].sum()), 'pooled_within_b_significant': int(sig[n:, n:].sum())}


if __name__ == '__main__' and '--from-npz' in sys.argv:          # recompute the pooled metrics from a saved run
    z = np.load(sys.argv[sys.argv.index('--from-npz') + 1]); row = json.loads(open('results_null_check.jsonl').readline())
    row.update(pooled_metrics(z['TE'], z['NULL'], int(z['n']))); print(json.dumps(row))
    open('results_null_check.jsonl', 'w').write(json.dumps(row) + '\n'); sys.exit()

ap = argparse.ArgumentParser(); ap.add_argument('mat'); ap.add_argument('--a', type=int, default=0)
ap.add_argument('--b', type=int, default=1); ap.add_argument('--surr', type=int, default=N_SURR)
a = ap.parse_args()
S = load(a.mat)
A = window(S, a.a * SEG, SEG); B = window(S, a.b * SEG, SEG)
r = tc.te_network_spikes(A + B, SEG, n_surr=a.surr, log=lambda *x: None)
G = r['sig']; n = len(A)
cross = int(G[:n, n:].sum() + G[n:, :n].sum()); within_a = int(G[:n, :n].sum()); within_b = int(G[n:, n:].sum())
P = r['P_z']; cp = np.r_[P[:n, n:].ravel(), P[n:, :n].ravel()]
row = {'segments': [a.a, a.b], 'surrogates': a.surr, 'cross_pairs': 2 * n * n, 'cross_significant_fdr': cross,
       'cross_frac_p_below_0.05': round(float(np.mean(cp < 0.05)), 4), 'cross_frac_p_below_0.001': round(float(np.mean(cp < 0.001)), 4),
       'within_a_significant': within_a, 'within_b_significant': within_b,
       'overlap_within_a_b': int((G[:n, :n] & G[n:, n:]).sum())}
np.savez_compressed(f'null_{a.a}{a.b}.npz', TE=r['TE_peak'], NULL=r['NULL'], n=n)
row.update(pooled_metrics(r['TE_peak'], r['NULL'], n))
print(json.dumps(row))
with open('results_null_check.jsonl', 'a') as f: f.write(json.dumps(row) + '\n')
