# Packet erasures and transfer entropy precision

Research prototype, preliminary numerical experiments, 2026-11-24. Study of the relationship between packet erasures, temporal dependence, and the precision of directed-dependence estimation. All newly executed signal experiments here use synthetic signals. No structural connectivity ground truth is assumed for recordings. No claim of a validated protocol or a novel theorem. No raw neural recordings are included.

## Reproduction
Python 3.10+, install dependencies with `python -m pip install -r requirements.txt` (NumPy, SciPy and pandas). Each subdirectory is standalone; run from that directory (scripts may overwrite result CSVs). See stage READMEs and code for parameters. Fixed pseudorandom seeds support reproducibility; elapsed times and floating-point results can vary by environment.

- `01_detection`: artificial known coupling and loss / detection experiment; UDP payload loopback. `python experiment.py`, `python payload_loopback.py`, `python analyze.py`.
- `02_oracle_variance`: oracle covariance approximation with known population. `python variance_validation.py`, `python switching_validation.py`, `python confirm.py`.
- `03_pilot_policy`: unknown-signal pilot estimates, preliminary strategy selection, same-split exploratory lag sensitivity, local echo test. `python validate.py`, `python bandwidth_sensitivity.py`.
- `04_holdout_queue`: new independent lag validation and event-driven finite shared-link FIFO model. `python lag_holdout.py`, `python queue_benchmark.py`. See stage README for real-data workflow.

## New results (stage 04)
A fresh fit/validation/test split (48/64/96 trajectories per regime, N=16000) selected each model's Bartlett lag from 100, 300, 1000, 2000 on validation variance only. Mean absolute relative deviation on fresh test variance, 72 settings: fraction-only 17.06%, long-run covariance 10.07%. By coupling-regime persistence .9/.98/.995: fraction 14.55/13.33/23.29%; long-run 11.27/7.17/11.76%. Reused signals across conditions; no formal inferential claim or proof of universal superiority. Lag tuning uses independent synthetic trajectories and may not transfer to a single nonstationary recording.

An event-driven FIFO with finite capacity eight packets, service times 6/9/12 ms, data generation every 10 ms, abstract equal-sized packet units, shared forward/reverse queue, propagation 8 ms, independent post-service erasures .1/.3, deadlines 50/100 ms, 40 trajectories per setting. At most six offered packet units per four data packets (including control). Actual offered rates differ by mode. XOR uses parity after each group; NACK can request at most one missing packet after a fixed group timeout. No real TCP/RTP stack, cross traffic, actual byte budget, or congestion-control algorithm is simulated. Queue overflow is modelled, not measured. At 10% erasure, 100 ms deadline, service time 6 ms, delivery fraction none/XOR/NACK 89.83/96.43/96.77%; TE RMSE versus complete same-trajectory estimate .001363/.000690/.000945 bits. At 9 ms, respective delivery 89.83/92.26/89.90%; RMSE .001363/.001221/.001672 bits. At 12 ms the offered originals alone exceed capacity. These are illustrative, not a claim that a particular strategy always wins. No inference of anatomy, significance testing, or real link performance.

## Data and remaining validation
The original ASDF file is listed at https://github.com/ShimonoMLab/NetworkEstimate_WithEICateg/blob/master/spike.mat (Git blob SHA 5b8cb735cc72680956aa603f0274d71b532e851c). The binary could not be retrieved into this execution environment. Download locally and check the index base and units with the owner. `real_multichannel.py` selects 4 active channels using the pilot half, evaluates 12 ordered pairs on held-out blocks; loader passed an artificial ASDF fixture test only. Actual recording analysis is outstanding. Physical two-host tests require two authorized machines; the echo probe is not a physical-network TE/FEC integration. Reproduction on multiple independent recordings, uncertainty estimates, nonstationarity, informative loss, physical protocol integration, and prior-art review remain outstanding.

## Publication
Suggested placement: `ShimonoMLab/NetworkTM_Analysis/te_packet_precision/`. Review owner rights, existing repository license, data terms, source provenance, and AI-assisted code independently before publishing. Use the existing repository license only if applicable and authorized. No new license is assigned by this package. Do not upload raw data, private material, or credentials. This package is a preliminary research prototype.

## Stage 05: state-dependent packet erasures
Run `cd 05_state_dependent_loss && python informative_loss.py`. The synthetic benchmark estimates state-specific inclusion probabilities on a separate pilot and evaluates inverse-probability-weighted TE on 128 independent trajectories per mechanism. See the stage README for assumptions, numerical results, negative controls and limitations. Raw spike data are not included.


## Stage 06: packet-to-TE integration
See [06_udp_integration](06_udp_integration/README.md) for an actual localhost socket payload-to-TE benchmark with 48 tests on two provenance-unverified binary arrays; [prior-art comparison](06_udp_integration/literature_comparison.md) delineates related methods and limits. No verified recording or two-physical-host run is claimed.
