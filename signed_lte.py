# -*- coding: utf-8 -*-
"""Signed local transfer entropy (SLTE) on rises of the series, with a toy
check on fluid queue models.  WORKING HYPOTHESIS, not a validated method.

Definition (history 1).  A "rise" plays the role of a spike in Kajiwara et
al. (2021): u_t = 1 if x_t > x_{t-1}, else 0 (no rise: fall or no change);
w_t likewise for y.
Local TE (Lizier et al., 2008):
    i_t = log2 p(w_{t+1} | w_t, u_t) - log2 p(w_{t+1} | w_t)
Sign as in the sorted local TE of Kajiwara et al. (2021): s_t = +1 if
w_{t+1} = u_t, -1 otherwise.      SLTE_{x->y} = mean_t s_t i_t.

Identity (checked numerically by slte_identity).  Fix the history w_t = a
and let d_a = p(w'=1 | a, u=1) - p(w'=1 | a, u=0).  If d_a > 0, every term
s_t i_t with w_t = a is >= 0; if d_a < 0, every such term is <= 0.  Hence
    SLTE = sum_a p(w_t = a) * sign(d_a) * E[ |i_t| | w_t = a ],
the mean size of the local TE in each history state, carrying the sign of
the association.  (The ordinary TE is E[i_t], where positive and negative
local values cancel; SLTE adds them up with the sign of the association.)
SLTE > 0 when, in every history state, a rise of x makes a rise of y more
likely; SLTE < 0 when it makes a rise less likely.

Limits.  Only the occurrence of rises and a lag of one step are used.  A
common driver that raises both series pushes SLTE positive, and an
interaction completed within one sampling step is invisible.
"""
import json
import numpy as np


def _events(x, y):
    u = (np.diff(x) > 0).astype(int)
    w = (np.diff(y) > 0).astype(int)
    return w[1:], w[:-1], u[:-1]


def _local(x, y):
    w1, w0, u0 = _events(x, y)
    c3 = np.zeros((2, 2, 2)); np.add.at(c3, (w1, w0, u0), 1)
    c2 = np.zeros((2, 2)); np.add.at(c2, (w1, w0), 1)
    p3 = c3 / np.maximum(c3.sum(0, keepdims=True), 1)
    p2 = c2 / np.maximum(c2.sum(0, keepdims=True), 1)
    i = np.log2(np.maximum(p3[w1, w0, u0], 1e-12)) - np.log2(np.maximum(p2[w1, w0], 1e-12))
    return w1, w0, u0, i, p3


def slte(x, y):
    w1, w0, u0, i, _ = _local(x, y)
    s = np.where(w1 == u0, 1.0, -1.0)
    return float(np.mean(s * i)), float(np.mean(i)), len(i)


def slte_identity(x, y):
    """sum_a p(a) sign(d_a) E[|i| | a]; equals slte(x, y)[0]."""
    w1, w0, u0, i, p3 = _local(x, y)
    tot = 0.0
    for a in (0, 1):
        m = w0 == a
        if m.any():
            tot += m.mean() * np.sign(p3[1, a, 1] - p3[1, a, 0]) * np.mean(np.abs(i[m]))
    return float(tot)


def test(x, y, n_surr=500, seed=0):
    rng = np.random.default_rng(seed)
    obs, te, ne = slte(x, y)
    null = np.array([slte(np.roll(x, s), y)[0] for s in rng.integers(200, len(x) - 200, n_surr)])
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_surr + 1)
    return {'SLTE': round(obs, 5), 'identity': round(slte_identity(x, y), 5), 'TE': round(te, 5),
            'n_events': ne, 'p_two_sided': round(float(p), 4), 'sign': '+' if obs > 0 else '-'}


def tandem(T, seed):
    """Two queues in tandem; most of queue 2's load comes out of queue 1."""
    rng = np.random.default_rng(seed)
    q1 = q2 = 0.0; d1 = np.zeros(T); d2 = np.zeros(T)
    for t in range(T):
        a1 = rng.gamma(2.0, 0.46)
        out1 = min(q1 + a1, 1.0); q1 += a1 - out1
        a2 = out1 * 0.9 + rng.gamma(2.0, 0.04)
        out2 = min(q2 + a2, 1.0); q2 += a2 - out2
        d1[t], d2[t] = q1, q2
    return d1, d2


def shared_same_step(T, seed, C=1.0):
    """Two AIMD-like senders, one bottleneck; both raise their rate every step,
    and on overload one of them (at random) halves its rate in the same step."""
    rng = np.random.default_rng(seed)
    r = np.array([0.3, 0.3]); thr = np.zeros((T, 2))
    for t in range(T):
        r = r + 0.02 + rng.normal(0, 0.005, 2).clip(-0.01, 0.01)
        tot = r.sum(); thr[t] = r * min(1.0, C / tot)
        if tot > C:
            k = rng.integers(2); r[k] *= 0.5
    return thr[:, 0], thr[:, 1]


def shared_lagged(T, seed, C=1.0):
    """As above, but each sender raises its rate only on some steps (paced
    independently), and the sender that did NOT raise its rate when the
    bottleneck overflowed sees the loss and backs off one step later
    (a response delayed by one round trip)."""
    rng = np.random.default_rng(seed)
    r = np.array([0.4, 0.4]); thr = np.zeros((T, 2)); pending = None
    for t in range(T):
        if pending is not None:
            r[pending] *= 0.5; pending = None
        inc = rng.random(2) < 0.5
        r = r + 0.04 * inc
        tot = r.sum(); thr[t] = r * min(1.0, C / tot)
        if tot > C:
            cand = np.nonzero(~inc)[0]
            pending = int(rng.choice(cand)) if len(cand) else int(rng.integers(2))
    return thr[:, 0], thr[:, 1]


def independent(T, seed):
    a, _ = tandem(T, seed); b, _ = tandem(T, seed + 1000)
    return a, b


def main():
    T = 20000; out = []
    for name, gen in (('propagation: tandem queues, delay', tandem),
                      ('competition, response in the same step: throughput', shared_same_step),
                      ('competition, response one step later: throughput', shared_lagged),
                      ('independent queues, delay', independent)):
        for sd in range(3):
            x, y = gen(T, sd)
            r = {'model': name, 'seed': sd, **test(x, y, seed=sd)}
            out.append(r); print(r, flush=True)
    json.dump(out, open('results_signed_lte.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
