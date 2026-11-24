# Stage 06: UDP payload-to-TE integration (preliminary)

## Reproduction
Python 3.10+, NumPy. `python udp_te.py benchmark spikes.npy spikes2.npy --out results --reps 4`. Input format: binary `(time,128)` NumPy arrays. The two input arrays used here were retrieved from previously supplied files; their recording IDs, acquisition interval, and measured-vs-generated origin have not been verified. Do not describe these runs as verified physiological recordings. SHA-256 hashes and selected channels are listed in `metadata.json`; the original arrays are not redistributed. The benchmark uses the first half for channel selection and the second half for testing. No anatomical ground truth is assumed. Each condition shares the same four erasure seeds across modes and inputs; these are not independent recordings.

## Method
Actual UDP sockets on loopback in two threads of a single process; 10 samples per packet, bit-packed 128 channels (160 data bytes), custom 7-byte header; groups of 4 data packets, XOR recovery of a single missing packet or one feedback-requested retry per group. Loss is injected at the application receiver with a fixed seed (including the parity/retry or simulated feedback decision). A group-end control datagram and a reply are used even in the no-repair baseline, for synchronization. The feedback decision is simulated; synchronization replies themselves are not intentionally dropped. This is a stop-and-wait experimental protocol, not RTP/RTCP, not a throughput or timing guarantee, and not an equal-traffic comparison. Reported byte counts include application headers and both directions, but not UDP/IP/link headers. Actual network loss, reorder, clock drift, congestion and deadlines are not controlled/tested here. Every retained/recovered bit is compared with the source. TE is plug-in binary conditional mutual information at lag one; valid triples require both consecutive samples. Zero filling is separately evaluated. Full-record TE is a reference estimate, not true coupling. Four masks per condition provide descriptive results only.

## Results
48 runs (2 arrays x 2 loss scenarios x 3 modes x 4 repetitions), 56 ordered channel pairs per run. 129,860 outbound datagrams and 24,048 synchronization replies; all application byte counts cross-checked, all received/recovered payloads matched the source. `socket_results.csv` gives individual runs, `socket_summary.csv` descriptive averages over 8 runs per scenario/mode. RMSE is first calculated over the 56 directed pairs then averaged over runs. Under independent 10% loss, no repair / XOR / NACK yielded respectively 90.20 / 96.44 / 96.94% reception and mean masked TE RMSE 3.025 / 1.959 / 1.675 x 10^-5 bits. Under correlated 30% loss (persistence 0.8), corresponding reception was 69.40 / 70.74 / 73.09% and RMSE 6.078 / 5.977 / 6.096 x 10^-5 bits. These small estimates are descriptive, not significance tests, and repair does not uniformly improve inference.

## Optional two-host run (not executed in the benchmark)
Use two trusted machines on the same authorized private LAN. Copy the script and install NumPy on both. For a 40,000 x 128 array (1000 groups), on the receiving machine, choose a private LAN IP and run:

```sh
python udp_te.py receive --bind RECEIVER_LAN_IP --port 50050 --groups 1000 --mode xor --loss 0.1 --persistence 0 --seed 73100 --out received_xor.npz
```

On the sender:

```sh
python udp_te.py send spikes.npy --host RECEIVER_LAN_IP --port 50050 --mode xor
```

Bring `received_xor.npz` to the sender via an approved transfer, then `python udp_te.py evaluate spikes.npy received_xor.npz`. Repeat with `none` and `nack` and seeds 73100–73103. For actual transport-only checks set `--loss 0` and `--persistence 0`. The custom UDP protocol assumes in-order group-end messages and reliable synchronization; on actual network loss or reordering the run may fail or time out. No authentication/encryption; use a trusted isolated test LAN with non-sensitive data, not the public Internet. This is a procedure for future two-physical-host tests, not evidence of their completion. To compare real deadlines, implement synchronized timestamping or round-trip-based bounds and account for actual transport overhead; this code does not claim deadline results.

## Provenance and limitations
The public `spike.mat` (GitHub `ShimonoMLab/NetworkEstimate_WithEICateg`, master, blob SHA `5b8cb735cc72680956aa603f0274d71b532e851c`) could not be fetched as binary into the analysis environment. To analyze it once obtained, use the existing stage 04 `real_multichannel.py` with a verified index base and record provenance. The newly tested arrays are not a substitute for verified recording metadata. No raw data or application documents are distributed with this research-code package.
