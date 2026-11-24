# -*- coding: utf-8 -*-
"""The same procedures as analysis.py, applied to neuronal spike trains.

Input: spike data in ASDF form (MATLAB .mat), as used in
github.com/ShimonoMLab/NetworkEstimate_WithEICateg: a cell array whose
first N cells hold the spike times (integer bins, 1 ms) of N neurons,
followed by one blank cell and a final cell [*, duration in bins].

  1. Spectral structure of the firing-count matrix (all neurons)
  2. Recovery of hidden entries of the firing-count matrix (10-s bins,
     all neurons, log1p): soft-impute vs. linear interpolation in time
     and per-neuron mean, entry-wise random and 48-sample block hiding,
     repeated masks (no periodic baseline)
  3. Directed dependence between spike trains: binary transfer entropy
     (1-ms bins, history 1) at delays 1..30 ms, peak over delays
     (Ito et al. 2011; Kajiwara et al. 2021). Per-pair surrogates made by
     circularly shifting the source by >= 1 s; the peak over delays is
     taken for each surrogate too, so scanning delays does not inflate
     the test; one-sided z-test; BH-FDR over all ordered pairs.
     For CPU run time, the TE step uses the --top-k most active neurons
     within the first --te-window bins (the GPU code in
     NetworkEstimate_WithEICateg handles full recordings).

Usage:
  python analysis_spikes.py spike.mat [--top-k 40] [--te-window 1800000]
         [--labels cell_categ.mat] [--out results_spikes]
"""
import os, json, argparse, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tmcore as tc
from analysis import plot_completion


def load_asdf(path, var=None):
    m = scipy.io.loadmat(path)
    if var is None:
        var = next(k for k, v in m.items() if not k.startswith('__')
                   and isinstance(v, np.ndarray) and v.dtype == object)
    cells = m[var].ravel()
    spikes = [np.asarray(c, np.int64).ravel() for c in cells[:-2]]
    T = int(np.asarray(cells[-1]).ravel()[1])
    return spikes, T


def load_labels(path):
    c = scipy.io.loadmat(path)['cell_categ'][0, 0]
    return {k: np.asarray(c[k]).ravel() for k in c.dtype.names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('asdf')
    ap.add_argument('--var', default=None)
    ap.add_argument('--rate-bin', type=int, default=10000, help='bins per firing-count sample (10000 = 10 s)')
    ap.add_argument('--tmax', type=int, default=0, help='firing-count samples used for completion (0 = all)')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--top-k', type=int, default=40)
    ap.add_argument('--te-window', type=int, default=1_800_000, help='bins used for TE (1.8e6 = first 30 min)')
    ap.add_argument('--delays', default='1:30')
    ap.add_argument('--n-surr', type=int, default=100)
    ap.add_argument('--labels', default=None, help='cell_categ.mat with exc/inh fields (optional)')
    ap.add_argument('--out', default='results_spikes')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)

    spikes, T = load_asdf(a.asdf, a.var)
    N = len(spikes); counts = np.array([s.size for s in spikes])
    print(f"neurons={N}  duration={T} bins  spikes={counts.sum()}  median rate={np.median(counts) / (T / 1000):.2f} Hz")

    # ---- firing-count matrix
    nb = T // a.rate_bin
    C = np.zeros((nb, N))
    for j, s in enumerate(spikes):
        s = s[(s >= 0) & (s < nb * a.rate_bin)]
        C[:, j] = np.bincount(s // a.rate_bin, minlength=nb)
    spec = {'raw': tc.spectral_structure(C), 'log': tc.spectral_structure(np.log1p(C))}
    L = np.log1p(C[:a.tmax] if a.tmax else C)
    keep = L.sum(0) > 0
    print(f"completion on {L.shape[0]} x {int(keep.sum())} (log1p firing counts per {a.rate_bin} bins):")
    comp = tc.completion_experiment(L[:, keep], period=None, seeds=a.seeds)

    # ---- spike-train TE
    d0, d1 = (int(x) for x in a.delays.split(':'))
    top = np.argsort(-counts)[:a.top_k]
    sub = [spikes[i][spikes[i] < a.te_window] for i in top]
    print(f"TE: top {a.top_k} neurons, first {a.te_window} bins, delays {d0}..{d1}, {a.n_surr} surrogates per pair")
    dep = tc.te_network_spikes(sub, a.te_window, delays=range(d0, d1 + 1), n_surr=a.n_surr,
                               min_shift=1000, log=lambda m: None)
    print(f"  FDR-significant {dep['n_sig_fdr']}/{dep['n_pairs']}")

    lab_summary = None
    if a.labels:
        lab = load_labels(a.labels)
        typ = np.where(lab['inh'][top] == 1, 'inh', 'exc')
        lab_summary = {t: {'neurons': int((typ == t).sum()),
                           'significant_out_edges': int(dep['sig'][typ == t].sum())} for t in ('exc', 'inh')}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    plot_completion(ax[0], ax[1], ax[2], spec, comp, a.seeds, 'Spectrum of firing-count matrix', 's')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'figures', 'figS1_spectrum_completion.png'), dpi=200)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    im = ax[0].imshow(dep['Z'], cmap='viridis')
    yy, xx = np.nonzero(dep['sig']); ax[0].scatter(xx, yy, s=5, c='white', linewidths=0)
    ax[0].set_title(f"Peak-TE surrogate z (dots: FDR q<0.05, {dep['n_sig_fdr']}/{dep['n_pairs']})", fontsize=9)
    ax[0].set_xlabel('target neuron (rank by rate)'); ax[0].set_ylabel('source neuron'); fig.colorbar(im, ax=ax[0], shrink=0.85)
    pd_ = dep['delay_peak'][dep['sig']]
    ax[1].hist(pd_, bins=np.arange(d0 - 0.5, d1 + 1.5), color='0.4')
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_xlabel('delay at peak TE (ms)'); ax[1].set_ylabel('significant pairs'); ax[1].set_title('Peak delays of significant pairs')
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'figures', 'figS2_spike_te.png'), dpi=200)

    res = {
        'data': {'file': os.path.basename(a.asdf), 'neurons': N, 'duration_bins': T, 'spikes': int(counts.sum())},
        'spectral_firing_counts': {k: {kk: v[kk] for kk in ('k90', 'k95', 'top1_energy')} for k, v in spec.items()},
        'completion': {'matrix': f'log1p firing counts, {a.rate_bin}-bin samples, first {a.tmax} samples, '
                                 f'{int(keep.sum())} neurons with spikes', 'nmae': comp},
        'transfer_entropy': {
            'neurons': f'top {a.top_k} by spike count', 'window_bins': a.te_window,
            'method': f'binary TE, 1-bin resolution, history 1, delays {d0}..{d1}, peak over delays',
            'surrogates': f'per pair, {a.n_surr} circular shifts of the source by >= 1000 bins, peak over delays',
            'test': 'one-sided z-test against per-pair surrogates, Benjamini-Hochberg FDR q=0.05',
            'n_pairs': dep['n_pairs'], 'n_significant_fdr': dep['n_sig_fdr'],
            'peak_delay_of_significant_pairs': {'median': float(np.median(pd_)) if pd_.size else None,
                                                'iqr': [float(np.percentile(pd_, 25)), float(np.percentile(pd_, 75))] if pd_.size else None},
            'by_source_type': lab_summary},
    }
    with open(os.path.join(a.out, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(a.out, 'te_matrices.npz'), neurons=top, TE_peak=dep['TE_peak'],
                        delay_peak=dep['delay_peak'], z=dep['Z'], p=dep['P_z'], significant=dep['sig'])


if __name__ == '__main__':
    main()
