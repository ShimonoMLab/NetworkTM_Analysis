"""NOTE (3.5): with the calibrated test (tmcore.pooled_null_p, null_check.py)
the reference network has no FDR-significant pair, so the comparison below
is not informative; see loss_groundtruth.py.  This file is kept for its data
loading and loss patterns, and decomp/ref_x.npz, decomp/len30_0.npz are
used in README section 5.

Separating the two ways packet loss changes an estimated spike-train network:
(i) less data (a sampling cost that shrinks with recording length) and
(ii) distortion (pairs that the complete long recording does not support,
which do not shrink with length).

Reference: the TE network of the 40 most active neurons over the whole
recording (150 min, no loss).  Five non-overlapping 30-min segments give
replicates.  For each estimate we report
  recall    = fraction of reference pairs recovered      (sampling)
  precision = fraction of estimated pairs in the reference (distortion)
Loss patterns: independent, bursty (Gilbert-Elliott, mean 10 packets), and
activity-dependent bursts (a burst is more likely to start after a packet
with many spikes, as when a congested link drops data when traffic rises).
Each job writes one .npz to decomp/; summarize with --summary.

  python loss_decompose.py spike.mat --worker 0 --of 2
  python loss_decompose.py spike.mat --summary
"""
import argparse, glob, json, os, time
import numpy as np
import tmcore as tc
import te_masked_spikes as tm
from analysis_spikes import load_asdf
from loss_spikes import drop_mask, PKT

N_NEURONS, SEG, N_SEG, N_SURR = 40, 1800000, 5, 50
N_LOSS_SEG = 3            # loss conditions on the first three segments
DELAYS = range(1, 31)
OUT = 'decomp'


def load(mat):
    spikes, T = load_asdf(mat)
    rates = np.array([len(s) for s in spikes]); top = np.argsort(rates)[-N_NEURONS:]
    S = [np.unique(np.asarray(spikes[i], np.int64)) for i in top]
    return [s[(s >= 1) & (s <= SEG * N_SEG)] - 1 for s in S]


def window(S, start, length):
    return [s[(s >= start) & (s < start + length)] - start for s in S]


def activity_bursts(S, length, p, burst, rng):
    """Gilbert-Elliott bursts whose start probability is proportional to the
    population spike count of the previous packet; scaled to loss rate p."""
    n_pkt = length // PKT
    c = np.zeros(n_pkt)
    for s in S: c += np.bincount(s // PKT, minlength=n_pkt)[:n_pkt]
    w = np.r_[c[0], c[:-1]] / c.mean()
    q_exit = 1.0 / burst; base = p * q_exit / (1 - p)
    u = rng.random(n_pkt)
    for _ in range(30):                                   # adjust so that the loss rate is p
        qe = np.clip(base * w, 0, 1); lost = np.zeros(n_pkt, bool); bad = False
        for k in range(n_pkt):
            bad = (u[k] >= q_exit) if bad else (u[k] < qe[k]); lost[k] = bad
        r = lost.mean()
        if abs(r - p) < 0.003: break
        base *= p / max(r, 1e-6)
    return lost


def loss_mask(S, length, kind, p, seed):
    rng = np.random.default_rng(seed)
    n_pkt = length // PKT
    if kind == 'indep': lost = drop_mask(n_pkt, p, 1, rng)
    elif kind == 'burst': lost = drop_mask(n_pkt, p, 10, rng)
    else: lost = activity_bursts(S, length, p, 10, rng)
    return np.repeat(~lost, PKT)[:length], float(lost.mean())


def jobs():
    J = [('ref', None, 'none', 'full', N_SEG * SEG)]
    for s in range(N_SEG):
        J += [('len30', s, 'none', 'full', SEG)]
        for kind in ('burst', 'indep', 'activity'):
            if s >= N_LOSS_SEG: break
            for est in ('zero', 'mask'):
                J.append((f'{kind}-{est}', s, kind, est, SEG))
    J += [('len60', 0, 'none', 'full', 2 * SEG), ('len60', 2, 'none', 'full', 2 * SEG)]
    J += [('burst150-zero', None, 'burst', 'zero', N_SEG * SEG), ('burst150-mask', None, 'burst', 'mask', N_SEG * SEG)]
    return J


def run(S, job):
    name, seg, kind, est, length = job
    start = 0 if seg is None else seg * SEG
    W = window(S, start, length)
    info = {'name': name, 'seg': -1 if seg is None else seg, 'minutes': length / 60000, 'kind': kind, 'est': est}
    t0 = time.time()
    if kind == 'none':
        r = tc.te_network_spikes(W, length, n_surr=N_SURR, log=lambda *x: None); f = 1.0; lost = 0.0
    else:
        V, lost = loss_mask(W, length, kind, 0.10, 1000 + 10 * (seg if seg is not None else 9))
        Nc = tm.valid_counts(V, DELAYS); f = float(np.mean(list(Nc.values()))) / length
        if est == 'zero':
            r = tc.te_network_spikes([s[V[s]] for s in W], length, n_surr=N_SURR, log=lambda *x: None)
        else:
            r = tm.te_network_masked(W, V, n_surr=N_SURR)
    info.update(packets_lost=round(lost, 4), valid_fraction=round(f, 4), sec=round(time.time() - t0))
    return r, info


def summary():
    ref = np.load(f'{OUT}/ref_x.npz'); R = ref['sig']
    rows = []
    for fn in sorted(glob.glob(f'{OUT}/*.npz')):
        z = np.load(fn, allow_pickle=True); info = json.loads(str(z['info'])); G = z['sig']
        if info['name'] == 'ref':
            info.update(n_sig=int(G.sum())); rows.append(info); continue
        tp = int((G & R).sum())
        info.update(n_sig=int(G.sum()), recall=round(tp / R.sum(), 3),
                    precision=round(tp / max(1, G.sum()), 3), not_in_ref=int((G & ~R).sum()))
        rows.append(info)
    with open('results_loss_decompose.json', 'w') as f: json.dump(rows, f, indent=1)
    by = {}
    for r in rows:
        if r['name'] == 'ref': print('reference', r['n_sig'], 'pairs'); continue
        by.setdefault(r['name'], []).append(r)
    for k, v in by.items():
        g = lambda key: np.array([x[key] for x in v], float)
        print(f"{k:16s} n={len(v)} sig {g('n_sig').mean():6.1f}  recall {g('recall').mean():.3f} [{g('recall').min():.2f}-{g('recall').max():.2f}]"
              f"  precision {g('precision').mean():.3f} [{g('precision').min():.2f}-{g('precision').max():.2f}]"
              f"  not_in_ref {g('not_in_ref').mean():5.1f}  lost {g('packets_lost').mean():.3f}  valid {g('valid_fraction').mean():.3f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mat')
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--summary', action='store_true'); ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    if a.summary: return summary()
    os.makedirs(OUT, exist_ok=True)
    S = load(a.mat)
    J = jobs()
    cost = lambda j: j[4] / SEG * (3.0 if j[3] == 'mask' else 1.0)
    order = sorted(range(len(J)), key=lambda i: -cost(J[i]))
    loads = [0.0] * a.of; mine = []
    for i in order:                                       # greedy balance by expected cost
        w = int(np.argmin(loads)); loads[w] += cost(J[i])
        if w == a.worker: mine.append(i)
    for i in sorted(mine):
        job = J[i]; fn = f"{OUT}/{job[0]}_{'x' if job[1] is None else job[1]}.npz"
        if a.only and job[0] not in a.only: continue
        if os.path.exists(fn): continue
        r, info = run(S, job)
        np.savez_compressed(fn, sig=r['sig'], TE=r['TE_peak'], Z=r['Z'], NULL=r['NULL'], info=json.dumps(info))
        print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
