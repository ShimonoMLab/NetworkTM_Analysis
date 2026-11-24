"""Packet loss and directed-dependence estimates, judged against a known truth.

Real spike trains keep their own firing statistics (rates, bursts, slow
changes), but the truth is set by construction:
  * null pairs: a neuron from one 30-min segment and a neuron from another
    segment, put on a common clock.  They cannot be coupled.
  * planted pairs: a target made of a real neuron from the second segment plus
    copies of a source neuron's spikes (from the first segment) at a fixed delay
    (2-25 ms, jitter 0-2 ms), each copied with probability c.
All 100 channels (40 + 40 + 20 planted) travel in the same 10-ms packets, so a
lost packet silences every channel at once.  For each condition we test 400
random null pairs and the 20 planted pairs with binary TE (delays 1-30 ms,
peak over delays, 50 circular-shift surrogates per pair), and compute
p-values from the pooled standardized surrogates (tmcore.pooled_null_p),
which is calibrated on null pairs (null_check.py); BH-FDR q = 0.05 over the
420 tests.
  false positives : null pairs with p < 0.01 (nominal 1%) and FDR-significant
  detection       : planted pairs FDR-significant

  python loss_groundtruth.py spike.mat --c 0.05 --rep 0 1 2
"""
import argparse, json, os, time
import numpy as np
import tmcore as tc
import te_masked_spikes as tm
from loss_decompose import load, window, loss_mask, SEG

DELAYS = list(range(1, 31)); N_SURR = 50; N_NULL = 400; N_PLANT = 20
REPS = [(0, 1), (2, 3), (4, 1), (1, 4), (3, 0)]


def build(S, a, b, length, c, seed):
    rng = np.random.default_rng(seed)
    A = window(S, a * SEG, length); B = window(S, b * SEG, length)
    src = rng.choice(40, N_PLANT, replace=False); bg = rng.choice(40, N_PLANT, replace=False)
    d = rng.integers(2, 26, N_PLANT)
    T = []
    for k in range(N_PLANT):
        x = A[src[k]]
        sel = x[rng.random(x.size) < c]
        cp = sel + d[k] + rng.integers(0, 3, sel.size)
        T.append(np.unique(np.r_[B[bg[k]], cp[cp < length]]))
    trains = A + B + T                                  # 0-39 A, 40-79 B, 80-99 planted
    null = set()
    while len(null) < N_NULL:
        i, j = rng.integers(0, 40), rng.integers(40, 80)
        null.add((i, j) if rng.random() < 0.5 else (j, i))
    pairs = sorted(null) + [(int(src[k]), 80 + k) for k in range(N_PLANT)]
    return trains, pairs, {'delays': d.tolist()}


def te_pairs(trains, length, pairs, V=None, seed=7):
    rng = np.random.default_rng(seed)
    S = [np.unique(np.asarray(s, np.int64)) for s in trains]
    if V is not None: S = [s[V[s]] for s in S]
    srcs = sorted({i for i, _ in pairs}); surr = {}
    for i in srcs:
        sh = rng.integers(1000, length - 1000, N_SURR)
        surr[i] = [np.sort((S[i] + s) % length) for s in sh]
        if V is not None: surr[i] = [x[V[x]] for x in surr[i]]
    TE = np.zeros(len(pairs)); NULL = np.zeros((len(pairs), N_SURR))
    if V is None:
        prev = {j: np.isin(S[j] - 1, S[j], assume_unique=True) for _, j in pairs}
        f = lambda Y, X, j: tc.te_spike_delays(Y, prev[j], X, length, DELAYS)
    else:
        Nc = tm.valid_counts(V, DELAYS); tt = {}
        def f(Y, X, j):
            if j not in tt: tt[j] = tm.target_terms(Y, V, DELAYS)
            return tm.te_spike_delays_masked(Y, X, V, DELAYS, Nc, tt[j])
    for k, (i, j) in enumerate(pairs):
        TE[k] = f(S[j], S[i], j).max()
        NULL[k] = [f(S[j], X, j).max() for X in surr[i]]
    return TE, NULL


def score(TE, NULL, n_null):
    P, _ = tc.pooled_null_p(TE[None, :], NULL[None, :, :], mask=np.ones((1, TE.size), bool))
    P = P[0]; sig = tc.bh_fdr(P, 0.05)
    return {'null_p_below_0.01': round(float(np.mean(P[:n_null] < 0.01)), 4),
            'null_p_below_0.05': round(float(np.mean(P[:n_null] < 0.05)), 4),
            'null_fdr_significant': int(sig[:n_null].sum()),
            'planted_detected': round(float(sig[n_null:].mean()), 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mat'); ap.add_argument('--c', type=float, default=0.05)
    ap.add_argument('--rep', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--conds', nargs='*', default=['none', 'none10', 'none20', 'indep-zero', 'indep-mask',
                                                   'burst-zero', 'burst-mask', 'activity-zero', 'activity-mask'])
    ap.add_argument('--long', action='store_true', help='60-min segments (0-60 vs 60-120 min), bursty loss')
    ap.add_argument('--out', default='results_loss_groundtruth.jsonl')
    ap.add_argument('--pairs', nargs='*', help='segment pairs as a,b (overrides the default list); --rep indexes into it')
    ap.add_argument('--dataset', default='spike.mat (NetworkEstimate_WithEICateg)')
    a = ap.parse_args()
    S = load(a.mat)
    plist = [tuple(int(x) for x in p.split(',')) for p in a.pairs] if a.pairs else REPS
    reps = [(0, (0, 2))] if a.long else [(r, plist[r]) for r in a.rep]
    for r, (sa, sb) in reps:                  # r: replicate index, sets the seeds
        length = 2 * SEG if a.long else SEG
        if a.long: sa, sb = 0, 1                              # windows of 60 min starting at 0 and 60 min
        trains, pairs, meta = build(S, sa * (2 if a.long else 1), sb * (2 if a.long else 1), length, a.c, 100 + r)
        for cond in a.conds:
            t0 = time.time()
            if cond.startswith('none'):
                L = length if cond == 'none' else int(cond[4:]) * 60000
                tr = [s[s < L] for s in trains]; TE, NULL = te_pairs(tr, L, pairs); lost = 0.0; vf = 1.0; minutes = L / 60000
            else:
                kind, est = cond.split('-')
                V, lost = loss_mask(trains, length, kind, 0.10, 1000 + 10 * r + (7 if a.long else 0))
                vf = float(np.mean(list(tm.valid_counts(V, DELAYS).values()))) / length; minutes = length / 60000
                TE, NULL = te_pairs([s[V[s]] for s in trains], length, pairs) if est == 'zero' else te_pairs(trains, length, pairs, V)
            row = {'dataset': a.dataset, 'rep': r, 'segments': [sa, sb], 'long': a.long, 'c': a.c, 'cond': cond, 'minutes': minutes,
                   'packets_lost': round(lost, 4), 'valid_fraction': round(vf, 4), **score(TE, NULL, N_NULL),
                   'sec': round(time.time() - t0)}
            print(json.dumps(row), flush=True)
            with open(a.out, 'a') as f: f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
