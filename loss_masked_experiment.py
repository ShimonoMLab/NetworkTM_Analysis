"""Packet loss and the estimated spike-train network: reading the gap as
"no spike" (zero-fill) versus telling the estimator which packets were lost
(mask-aware TE, te_masked_spikes.py), and how the agreement with the
complete-data network grows with recording length.

Usage:
  python loss_masked_experiment.py spike.mat --part zero     # zero-fill + recording length
  python loss_masked_experiment.py spike.mat --part masked --cond 0 1 2 3
Results are appended to results_loss_masked.jsonl, one JSON object per line.
"""
import argparse, json, time
import numpy as np
from scipy.stats import spearmanr
import tmcore as tc
import te_masked_spikes as tm
from analysis_spikes import load_asdf
from loss_spikes import drop_mask, PKT

N_NEURONS, T_BINS, N_SURR = 40, 1800000, 50
CONDS = [(0.05, 1), (0.10, 1), (0.05, 10), (0.10, 10)]
SEEDS = (11, 12)


def load(mat):
    spikes, T = load_asdf(mat)
    rates = np.array([len(s) for s in spikes]); top = np.argsort(rates)[-N_NEURONS:]
    S = [np.unique(np.asarray(spikes[i])) for i in top]
    return [s[(s >= 1) & (s < T_BINS)] - 1 for s in S]


def compare(F, TEf, G, TEg):
    off = ~np.eye(F.shape[0], dtype=bool)
    inter = int((F & G).sum()); union = int((F | G).sum())
    return {'n_sig': int(G.sum()), 'kept_of_full': round(inter / max(1, F.sum()), 3),
            'new_not_in_full': int((G & ~F).sum()), 'jaccard': round(inter / max(1, union), 3),
            'spearman_TE': round(float(spearmanr(TEf[off], TEg[off]).correlation), 3)}


def mask_for(cond, seed):
    p, burst = cond
    rng = np.random.default_rng(seed * 100 + int(p * 100) + burst)
    lost = drop_mask(T_BINS // PKT, p, burst, rng)
    return np.repeat(~lost, PKT)[:T_BINS], float(lost.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mat'); ap.add_argument('--part', choices=['zero', 'masked'])
    ap.add_argument('--cond', type=int, nargs='*', default=list(range(len(CONDS))))
    ap.add_argument('--out', default='results_loss_masked.jsonl')
    a = ap.parse_args()
    S = load(a.mat)
    full = tc.te_network_spikes(S, T_BINS, n_surr=N_SURR, log=lambda *x: None)
    F, TEf = full['sig'], full['TE_peak']
    def emit(row):
        print(row, flush=True)
        with open(a.out, 'a') as f: f.write(json.dumps(row) + '\n')
    if a.part == 'zero':
        emit({'kind': 'full', 'n_sig': int(F.sum()), 'neurons': N_NEURONS, 'minutes': T_BINS / 60000, 'surrogates': N_SURR})
        for minutes in (7.5, 15, 22.5):                 # recording length, no loss
            Tm = int(minutes * 60000)
            Sm = [s[s < Tm] for s in S]
            r = tc.te_network_spikes(Sm, Tm, n_surr=N_SURR, log=lambda *x: None)
            emit({'kind': 'length', 'minutes': minutes, **compare(F, TEf, r['sig'], r['TE_peak'])})
        for ci in a.cond:
            for sd in SEEDS:
                V, lost = mask_for(CONDS[ci], sd)
                Sr = [s[V[s]] for s in S]
                t0 = time.time()
                r = tc.te_network_spikes(Sr, T_BINS, n_surr=N_SURR, log=lambda *x: None)
                emit({'kind': 'zero-fill', 'loss_rate': CONDS[ci][0], 'mean_burst_packets': CONDS[ci][1], 'seed': sd,
                      'packets_lost': round(lost, 4), **compare(F, TEf, r['sig'], r['TE_peak']), 'sec': round(time.time() - t0)})
    else:
        for ci in a.cond:
            for sd in SEEDS:
                V, lost = mask_for(CONDS[ci], sd)
                t0 = time.time()
                r = tm.te_network_masked(S, V, n_surr=N_SURR)
                emit({'kind': 'mask-aware', 'loss_rate': CONDS[ci][0], 'mean_burst_packets': CONDS[ci][1], 'seed': sd,
                      'packets_lost': round(lost, 4), **compare(F, TEf, r['sig'], r['TE_peak']), 'sec': round(time.time() - t0)})


if __name__ == '__main__':
    main()
