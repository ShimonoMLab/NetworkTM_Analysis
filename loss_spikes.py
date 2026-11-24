"""How much does packet loss in streaming change the estimated network?

Spike trains (ASDF .mat, 1-ms bins) are cut into packets of 10 ms that carry
all channels, as in a streaming frame. Packets are dropped independently
(rate p) or in bursts (two-state Gilbert-Elliott model, mean burst 10
packets, same overall rate). Spikes in dropped packets are simply missing
(no concealment). The binary-TE network (tmcore.te_network_spikes, per-pair
surrogates, BH-FDR) estimated from the received data is compared with the
network estimated from the complete data.

Usage: python loss_spikes.py path/to/spike.mat [--neurons 40 --bins 1800000]
"""
import argparse, json, time
import numpy as np
from scipy.stats import spearmanr
import tmcore as tc
from analysis_spikes import load_asdf

PKT = 10   # bins (ms) per packet


def drop_mask(n_pkt, p, burst, rng):
    if p == 0:
        return np.zeros(n_pkt, bool)
    if burst <= 1:
        return rng.random(n_pkt) < p
    # Gilbert-Elliott: leave the bad state with prob 1/burst; enter it so that the stationary loss is p
    q_exit = 1.0 / burst; q_enter = p * q_exit / (1 - p)
    lost = np.zeros(n_pkt, bool); bad = False
    u = rng.random(n_pkt)
    for k in range(n_pkt):
        bad = (u[k] >= q_exit) if bad else (u[k] < q_enter)
        lost[k] = bad
    return lost


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mat')
    ap.add_argument('--neurons', type=int, default=40); ap.add_argument('--bins', type=int, default=1800000)
    ap.add_argument('--surr', type=int, default=50); ap.add_argument('--out', default='results_loss_spikes.json')
    a = ap.parse_args()
    spikes, T = load_asdf(a.mat)
    rates = np.array([len(s) for s in spikes]); top = np.argsort(rates)[-a.neurons:]
    Tm = a.bins
    S = [np.asarray(spikes[i]) for i in top]; S = [s[(s >= 1) & (s < Tm)] - 1 for s in S]
    n_pkt = Tm // PKT
    t0 = time.time()
    full = tc.te_network_spikes(S, Tm, n_surr=a.surr, log=lambda *x: None)
    F = full['sig']; off = ~np.eye(len(S), dtype=bool)
    print(f'full: {full["n_sig_fdr"]} significant pairs ({time.time() - t0:.0f}s)', flush=True)
    out = {'neurons': a.neurons, 'bins': Tm, 'packet_ms': PKT, 'surrogates': a.surr,
           'full_n_sig': full['n_sig_fdr'], 'conditions': []}
    rng = np.random.default_rng(1)
    for p, burst in ((0.01, 1), (0.05, 1), (0.10, 1), (0.20, 1), (0.05, 10), (0.10, 10), (0.20, 10)):
        lost = drop_mask(n_pkt, p, burst, rng)
        Sr = [s[~lost[np.minimum(s // PKT, n_pkt - 1)]] for s in S]
        t0 = time.time()
        r = tc.te_network_spikes(Sr, Tm, n_surr=a.surr, log=lambda *x: None)
        G = r['sig']
        inter = int((F & G).sum()); union = int((F | G).sum())
        rho = spearmanr(full['TE_peak'][off], r['TE_peak'][off]).correlation
        row = {'loss_rate': p, 'mean_burst_packets': burst, 'packets_lost': float(lost.mean()),
               'spikes_kept': float(sum(map(len, Sr)) / sum(map(len, S))),
               'n_sig': int(G.sum()), 'kept_of_full': round(inter / max(1, F.sum()), 3),
               'new_not_in_full': int((G & ~F).sum()), 'jaccard': round(inter / max(1, union), 3),
               'spearman_TE': round(float(rho), 3)}
        out['conditions'].append(row); print(row, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
