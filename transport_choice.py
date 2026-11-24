"""Choosing the transport for a streamed multichannel recording: a formulation.

Packets of tau = 10 ms carry all channels.  The link loses packets
(independently, or in Gilbert-Elliott bursts) with rate p and has one-way
delay delta.  A process that can wait D ms after a packet was sent uses
what has arrived by then.  Schemes:

  mask      UDP; sequence numbers give the loss positions (loss mask)
  fec(k)    one XOR parity packet after every k data packets
  arq       NACK on a detected gap, retransmission; repeated NACK after a
            timeout (selective repeat)

For each scheme s and deadline D we simulate the residual loss R_s(D) and
compute
  f(R)   fraction of steps t whose bins t+1, t and t+1-d (d = 1..30 ms) were
         all delivered, averaged over d -- the data the mask-aware TE
         estimator can use,
  b_s    bandwidth used / bandwidth of the data.
The mask-aware estimator is calibrated whatever the residual loss (README
section 6) and behaves like a complete recording of length f*T (tested in
section 7 with loss_length.py).  So the recording time needed to reach a
target detection rate is
        T*(s, D) = T*_0 / f(R_s(D)),
and the choice is   min_s T*(s, D)   subject to   b_s * rho <= C,
with rho the data rate and C the capacity.  Reading gaps as silence is not
a scheme here: its error is a bias that does not shrink with T.

  python transport_choice.py          # writes results_transport_choice.json
"""
import json
import numpy as np
import te_masked_spikes as tm
from loss_spikes import drop_mask

TAU, DELTA, P, BURST = 10.0, 20.0, 0.10, 10
N_PKT = 20000                      # 200 s of packets
DELAYS = list(range(1, 31))


def link(n, rng, burst):
    return drop_mask(n, P, burst, rng)


def residual_mask(scheme, D, burst, seed, k=4):
    """Delivered-by-deadline indicator per data packet, and bandwidth ratio."""
    rng = np.random.default_rng(seed)
    if scheme == 'mask':
        lost = link(N_PKT, rng, burst); return ~lost, 1.0
    if scheme == 'fec':
        n_groups = N_PKT // k
        lost = link(n_groups * (k + 1), rng, burst).reshape(n_groups, k + 1)   # k data + parity, in sending order
        got = ~lost[:, :k]
        ok_group = lost.sum(1) <= 1                                          # one loss in the group is recoverable
        t_send = (np.arange(n_groups)[:, None] * k + np.arange(k)[None, :]) * TAU
        t_recover = (np.arange(n_groups) * k + k) * TAU + DELTA              # when the parity arrives
        rec = (~got) & ok_group[:, None] & ((t_recover[:, None] - t_send) <= D)
        on_time = got & (DELTA <= D)
        return (on_time | rec).ravel(), (k + 1) / k
    # arq
    lost = link(N_PKT, rng, burst)
    t_send = np.arange(N_PKT) * TAU
    delivered = ~lost & (DELTA <= D)
    rto = 2 * DELTA + 10.0; extra = 0
    nxt = np.empty(N_PKT); last = np.inf                      # time the next received packet arrives
    for i in range(N_PKT - 1, -1, -1):
        nxt[i] = last
        if not lost[i]: last = t_send[i] + DELTA
    for i in np.flatnonzero(lost):
        t = nxt[i]                                            # gap noticed
        while np.isfinite(t) and t - t_send[i] <= D:
            nack_ok = rng.random() >= P; retx_ok = rng.random() >= P
            if nack_ok: extra += 1
            if nack_ok and retx_ok:
                if t + 2 * DELTA - t_send[i] <= D: delivered[i] = True
                break
            t += rto                                          # NACK or retransmission lost: try again
    return delivered, 1.0 + extra / N_PKT


def valid_fraction(delivered):
    V = np.repeat(delivered, int(TAU))
    Nc = tm.valid_counts(V, DELAYS)
    return float(np.mean(list(Nc.values()))) / V.size


def main():
    rows = []
    for burst, name in ((1, 'independent'), (BURST, 'bursts')):
        for D in (25, 50, 100, 200, 1000):
            for scheme in ('mask', 'fec', 'arq'):
                dl, b = residual_mask(scheme, D, burst, seed=11)
                f = valid_fraction(dl)
                rows.append({'loss': name, 'deadline_ms': D, 'scheme': scheme, 'residual_loss': round(1 - float(dl.mean()), 4),
                             'bandwidth_ratio': round(b, 3), 'valid_fraction_f': round(f, 4), 'time_factor_1_over_f': round(1 / f, 3)})
                print(json.dumps(rows[-1]))
    json.dump({'setup': {'tau_ms': TAU, 'one_way_delay_ms': DELTA, 'loss_rate': P, 'mean_burst_packets': BURST,
                         'fec_group': 4, 'packets': N_PKT}, 'rows': rows}, open('results_transport_choice.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
