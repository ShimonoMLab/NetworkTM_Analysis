"""Separating loss-induced error from recording length, against a known truth.

Uses the construction of loss_groundtruth.py (null pairs that cannot be
coupled, planted pairs that are) with two designs:

equiv   Does mask-aware estimation under loss behave like a shorter complete
        recording?  The formulation (README section 7) predicts that it
        behaves like a recording of length f*T, where f is the fraction of
        steps whose bins t+1, t, t+1-d were all received.  100 planted pairs
        per replicate; complete recordings of 15, 20, 25, 30 min give the
        detection-vs-length curve; independent and bursty loss (10%) with the
        mask give the points to place on it.  Five replicates (segment pairs
        of loss_groundtruth.REPS), seeds set by the replicate index.

growth  Does the error grow, stay, or shrink as the recording gets longer?
        Nested windows of 15, 30 and 60 min starting at the same times (three
        replicates: A/B starting at 0/60, 30/90, 90/0 min), bursty loss 10%,
        gaps read as silence vs mask-aware.  A sampling error shrinks with
        length; a distortion (bias) does not.

  python loss_length.py spike.mat equiv  --rep 0 1 2 3 4
  python loss_length.py spike.mat growth --rep 0 1 2
"""
import argparse, json, time
import numpy as np
import te_masked_spikes as tm
from loss_decompose import load, window, loss_mask, SEG
from loss_groundtruth import te_pairs, score, REPS, DELAYS, N_NULL

MIN = 60000
GROWTH_STARTS = [(0, 60), (30, 90), (90, 0)]


def build_at(S, startA, startB, length, c, seed, n_plant):
    rng = np.random.default_rng(seed)
    A = window(S, startA, length); B = window(S, startB, length)
    src = rng.integers(0, 40, n_plant); bg = rng.integers(0, 40, n_plant); d = rng.integers(2, 26, n_plant)
    T = []
    for k in range(n_plant):
        x = A[src[k]]; sel = x[rng.random(x.size) < c]
        cp = sel + d[k] + rng.integers(0, 3, sel.size)
        T.append(np.unique(np.r_[B[bg[k]], cp[cp < length]]))
    trains = A + B + T
    null = set()
    while len(null) < N_NULL:
        i, j = rng.integers(0, 40), rng.integers(40, 80)
        null.add((i, j) if rng.random() < 0.5 else (j, i))
    return trains, sorted(null) + [(int(src[k]), 80 + k) for k in range(n_plant)]


def run_cond(trains, length, pairs, cond, seed):
    if cond == 'none':
        TE, NULL = te_pairs(trains, length, pairs); f = 1.0; lost = 0.0
    else:
        kind, est = cond.split('-')
        V, lost = loss_mask(trains[:80], length, kind, 0.10, seed)
        f = float(np.mean(list(tm.valid_counts(V, DELAYS).values()))) / length
        TE, NULL = te_pairs([s[V[s]] for s in trains], length, pairs) if est == 'zero' else te_pairs(trains, length, pairs, V)
    return TE, NULL, f, lost


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mat'); ap.add_argument('design', choices=['equiv', 'growth'])
    ap.add_argument('--rep', type=int, nargs='*', default=[0]); ap.add_argument('--c', type=float, default=0.02)
    a = ap.parse_args(); S = load(a.mat)
    out = 'results_equivtime.jsonl' if a.design == 'equiv' else 'results_length_growth.jsonl'
    for r in a.rep:
        if a.design == 'equiv':
            sa, sb = REPS[r]; n_plant = 100
            trains, pairs = build_at(S, sa * SEG, sb * SEG, SEG, a.c, 500 + r, n_plant)
            plan = [(m, 'none') for m in (15, 20, 25, 30)] + [(30, 'indep-mask'), (30, 'burst-mask')]
        else:
            sa, sb = GROWTH_STARTS[r]; n_plant = 20
            trains, pairs = build_at(S, sa * MIN, sb * MIN, 60 * MIN, a.c, 700 + r, n_plant)
            plan = [(m, cond) for m in (15, 30, 60) for cond in ('none', 'burst-zero', 'burst-mask')]
        for minutes, cond in plan:
            t0 = time.time(); L = minutes * MIN
            tr = [s[s < L] for s in trains]
            TE, NULL, f, lost = run_cond(tr, L, pairs, cond, 2000 + 10 * r)
            sc = score(TE, NULL, N_NULL)
            row = {'design': a.design, 'rep': r, 'starts_min': [sa * (30 if a.design == 'equiv' else 1), sb * (30 if a.design == 'equiv' else 1)],
                   'minutes': minutes, 'cond': cond, 'n_planted': n_plant, 'packets_lost': round(lost, 4),
                   'valid_fraction': round(f, 4), **sc, 'planted_detected_count': int(round(sc['planted_detected'] * n_plant)),
                   'sec': round(time.time() - t0)}
            print(json.dumps(row), flush=True)
            with open(out, 'a') as fh: fh.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
