# Stage 06: UDP payload-to-TE integration (preliminary)

## Reproduction
Python 3.10+, NumPy. Run `python synthetic_benchmark.py --out results_synthetic --reps 4`. The input arrays are generated locally from fixed seeds; no recording files are required. The two 40,000 x 128 arrays have background probabilities 0.0025 and 0.0055, seeds 74001 and 74002, and a 0.2 probability of copying a source event to its adjacent target one sample later. These are synthetic software fixtures, not neuronal recordings. `metadata.json` lists input hashes, split and selected channels. The first half selects eight active channels and the second half evaluates 56 directions.

## Method
Actual UDP sockets on loopback in two threads of a single process; 10 samples per packet, bit-packed 128 channels (160 data bytes), custom 7-byte header; groups of 4 data packets, XOR recovery of a single missing packet or one feedback-requested retry per group. Loss is injected at the application receiver with a fixed seed (including the parity/retry or simulated feedback decision). A group-end control datagram and a reply are used even in the no-repair baseline, for synchronization. The feedback decision is simulated; synchronization replies themselves are not intentionally dropped. This is a stop-and-wait experimental protocol, not RTP/RTCP, not a throughput or timing guarantee, and not an equal-traffic comparison. Reported byte counts include application headers and both directions, but not UDP/IP/link headers. Actual network loss, reorder, clock drift, congestion and deadlines are not controlled/tested here. Every retained/recovered bit is compared with the source. TE is plug-in binary conditional mutual information at lag one; valid triples require both consecutive samples. Zero filling is separately evaluated. Full-record TE is a reference estimate, not true coupling. Four masks per condition provide descriptive results only.

## Results
48 runs, all received/recovered payloads matched. 129,860 outgoing datagrams, 24,048 synchronization replies. In the independent 10% erasure setting, reception for no repair/XOR/NACK was 90.20/96.44/96.94%; mean TE RMSE was 2.133/1.572/1.555 x 10^-5 bits. For correlated 30% erasure, reception was 69.40/70.74/73.09%; RMSE was 4.314/4.436/4.141 x 10^-5 bits. See the CSVs for all trial values and byte counts. Reused masks, two signals, unequal byte costs; descriptive only.

## Optional two-host run (not executed in the benchmark)
Use two trusted machines on the same authorized private LAN. Copy the script and install NumPy on both. First generate a local fixture on the sender with `python -c "from synthetic_benchmark import synthetic; import numpy as np; np.save('synthetic_low.npy', synthetic(74001,0.0025))"`. For that 40,000 x 128 array (1000 groups), on the receiving machine, choose a private LAN IP and run:

```sh
python udp_te.py receive --bind RECEIVER_LAN_IP --port 50050 --groups 1000 --mode xor --loss 0.1 --persistence 0 --seed 73100 --out received_xor.npz
```

On the sender:

```sh
python udp_te.py send synthetic_low.npy --host RECEIVER_LAN_IP --port 50050 --mode xor
```

Bring `received_xor.npz` to the sender via an approved transfer, then `python udp_te.py evaluate synthetic_low.npy received_xor.npz`. Repeat with `none` and `nack` and seeds 73100–73103. For actual transport-only checks set `--loss 0` and `--persistence 0`. The custom UDP protocol assumes in-order group-end messages and reliable synchronization; on actual network loss or reordering the run may fail or time out. No authentication/encryption; use a trusted isolated test LAN with non-sensitive data, not the public Internet. This is a procedure for future two-physical-host tests, not evidence of their completion. To compare real deadlines, implement synchronized timestamping or round-trip-based bounds and account for actual transport overhead; this code does not claim deadline results.

## Provenance and limitations
The public `spike.mat` (GitHub `ShimonoMLab/NetworkEstimate_WithEICateg`, master, blob SHA `5b8cb735cc72680956aa603f0274d71b532e851c`) has not been evaluated in this synthetic socket benchmark. For a separate recording analysis, use the existing stage 04 `real_multichannel.py` with a verified index base and record provenance. The newly tested arrays are synthetic and are not a substitute for physiological recordings. No raw recordings are distributed.
