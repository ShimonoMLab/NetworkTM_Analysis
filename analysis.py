# -*- coding: utf-8 -*-
"""Exploratory analyses of public traffic-matrix time series (GEANT / TOTEM).

  1. Spectral structure of the OD traffic matrix (raw vs. log scale)
  2. Recovery of hidden OD entries: soft-impute vs. three baselines,
     entry-wise random and 48-h block hiding, 10 independent masks
  3. Directed statistical dependence between node-level traffic series
     (hourly on the clock, hour-of-week profile removed from timestamps,
     binned TE using only consecutive observed hours, per-pair
     surrogates, BH-FDR; see te_hourly.py)
  4. Anomaly candidates at the OD level (no fault localization)

The procedures themselves live in tmcore.py and are shared with
analysis_spikes.py (neuronal spike trains).

Data: GEANT intradomain traffic matrices (23 nodes, 529 OD pairs, 15-min
bins, Jan-Apr 2005), TOTEM project: S. Uhlig, B. Quoitin, J. Lepropre,
S. Balon, ACM SIGCOMM CCR 36(1), 2006. Not redistributed here.

Usage:  python analysis.py <path/to/geant-flat-tms.csv> [--seeds 10]
"""
import json, argparse, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tmcore as tc
import te_hourly as th

BINS_PER_DAY = 96          # 15-min bins
STRIDE = 4                 # -> hourly


def load_tms(csv_path):
    df = pd.read_csv(csv_path, header=None)
    ts = pd.to_datetime(df[0], format='%Y-%m-%d-%H-%M')
    return ts, df.iloc[:, 1:].to_numpy(float)


def plot_completion(ax_spec, ax_r, ax_b, spec, comp, seeds, title_spec, unit):
    for name, mk in (('raw', 'o-'), ('log', 's-')):
        sv = spec[name]['sv']; ax_spec.semilogy(sv[:80] / sv[0], mk, ms=2.5, label=name)
    ax_spec.set_xlabel('component'); ax_spec.set_ylabel('normalized singular value')
    ax_spec.set_title(title_spec); ax_spec.legend()
    for axi, scheme, ttl in ((ax_r, 'random', 'Hidden at random (entry-wise)'),
                             (ax_b, 'block', f'Hidden in 48-{unit} blocks')):
        cs = [c for c in comp if c['scheme'] == scheme]
        r = [c['missing'] for c in cs]
        for k, mk, lab in (('soft_impute', 'o-', 'soft-impute (low-rank reg.)'),
                           ('time_interp', '^-', 'linear interp. in time'),
                           ('tod_profile', 'd-', 'time-of-day profile'),
                           ('flow_mean', 's--', 'per-channel mean')):
            if k in cs[0]:
                axi.errorbar(r, [c[k]['mean'] for c in cs], yerr=[c[k]['sd'] for c in cs],
                             fmt=mk, ms=4, capsize=3, label=lab)
        axi.set_xlabel('fraction of hidden entries'); axi.set_ylabel(f'NMAE (mean ± sd, {seeds} masks)')
        axi.set_title(ttl); axi.set_ylim(0, 1.05)
    ax_r.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()
    ts, X = load_tms(a.csv)
    T, F = X.shape; n = int(round(F ** 0.5))
    print(f"T={T}  flows={F}  nodes={n}  {ts.iloc[0]} .. {ts.iloc[-1]}")

    spec = {'raw': tc.spectral_structure(X), 'log': tc.spectral_structure(np.log1p(X))}
    print("completion:")
    L = np.log1p(X)[::STRIDE][:1500]
    comp = tc.completion_experiment(L, period=BINS_PER_DAY // STRIDE, seeds=a.seeds)
    print("transfer entropy:")
    hidx, Sh = th.hourly_node_series(ts, X, n)
    dep = th.te_network_hourly(th.residuals(hidx, Sh, ('week',)))
    print(f"  hourly grid {len(hidx)} h ({int(np.isnan(Sh[:, 0]).sum())} missing); "
          f"FDR-significant {dep['n_sig_fdr']}/{dep['n_pairs']}")
    hits, z = tc.anomaly_candidates(np.log1p(X), period=BINS_PER_DAY)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    plot_completion(ax[0], ax[1], ax[2], spec, comp, a.seeds, 'Spectrum of GEANT OD matrix', 'h')
    fig.tight_layout(); fig.savefig('figures/fig1_spectrum_completion.png', dpi=200)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(dep['Z'], cmap='viridis')
    yy, xx = np.nonzero(dep['sig']); ax.scatter(xx, yy, s=6, c='white', linewidths=0)
    ax.set_title(f"TE surrogate z-score (dots: FDR q<0.05, {dep['n_sig_fdr']}/{dep['n_pairs']} pairs)", fontsize=9)
    ax.set_xlabel('target node'); ax.set_ylabel('source node'); fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig('figures/fig2_te_network.png', dpi=200)

    results = {
        'data': {'T': T, 'flows': F, 'nodes': n, 'start': str(ts.iloc[0]), 'end': str(ts.iloc[-1])},
        'spectral': {k: {kk: v[kk] for kk in ('k90', 'k95', 'top1_energy')} for k, v in spec.items()},
        'completion_nmae': comp,
        'transfer_entropy': {
            'series': 'node outgoing traffic, resampled hourly on the clock, log, hour-of-week median '
                      '(from timestamps) removed, 4 quantile bins, history 1, only consecutive observed hours',
            'surrogates': 'per pair, 200 circular shifts of the source by >= 7 days',
            'test': 'one-sided z-test against per-pair surrogates, Benjamini-Hochberg FDR q=0.05',
            'n_pairs': dep['n_pairs'], 'n_significant_fdr': dep['n_sig_fdr']},
        'anomaly_candidates_z_gt_6': int(len(hits)),
        'top_anomaly_candidates': [{'time': str(ts.iloc[t]), 'od': f'{f // n}->{f % n}', 'z': float(z[t, f])}
                                   for t, f in hits[:10]],
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
