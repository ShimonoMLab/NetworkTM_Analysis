"""How often are all channels silent in a packet, without any loss?
For the 40 most active neurons of each recording, the first 30 min, 10-ms
packets: median rate, spikes per channel per packet, the fraction of packets
in which a channel is silent, and the fraction in which all 40 are silent.
Reading a lost packet as silence inserts an all-channel silence; the rarer
such packets are naturally, the stronger the common event it creates
(README section 6).

  python packet_silence.py spike.mat asdf.mat
"""
import json, sys
import numpy as np
from loss_decompose import load, window, SEG
from loss_spikes import PKT

rows = []
for path in sys.argv[1:]:
    W = window(load(path), 0, SEG); n_pkt = SEG // PKT
    rates = np.array([len(s) for s in W]) / (SEG / 1000)
    silent = np.ones(n_pkt, bool); empty = []
    for s in W:
        c = np.bincount(s // PKT, minlength=n_pkt)[:n_pkt]; empty.append(float((c == 0).mean())); silent &= c == 0
    rows.append({'file': path.split('/')[-1], 'median_rate_hz': round(float(np.median(rates)), 2),
                 'spikes_per_channel_per_packet': round(float(rates.mean() * PKT / 1000), 3),
                 'median_fraction_channel_silent': round(float(np.median(empty)), 3),
                 'fraction_all_40_silent': round(float(silent.mean()), 3)})
    print(json.dumps(rows[-1]))
json.dump(rows, open('results_packet_silence.json', 'w'), indent=1)
