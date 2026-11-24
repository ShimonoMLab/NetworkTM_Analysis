# Packet erasures and transfer entropy precision

Research prototype, preliminary numerical experiments, 2026-11-25 (v76). Study of the relationship between packet erasures, temporal dependence, and the precision of directed-dependence estimation. All newly executed signal experiments here use synthetic signals. No structural connectivity ground truth is assumed for recordings. No claim of a validated protocol or a novel theorem. No raw neural recordings are included.

## Reproduction
Python 3.10+, install dependencies with `python -m pip install -r requirements.txt` (NumPy, SciPy, pandas and Matplotlib). Each subdirectory is standalone; run from that directory (scripts may overwrite result CSVs). See stage READMEs and code for parameters. Fixed pseudorandom seeds support reproducibility; elapsed times and floating-point results can vary by environment.

- `01_detection`: artificial known coupling and loss / detection experiment; UDP payload loopback. `python experiment.py`, `python payload_loopback.py`, `python analyze.py`.
- `02_oracle_variance`: oracle covariance approximation with known population. `python variance_validation.py`, `python switching_validation.py`, `python confirm.py`.
- `03_pilot_policy`: unknown-signal pilot estimates, preliminary strategy selection, same-split exploratory lag sensitivity, local echo test. `python validate.py`, `python bandwidth_sensitivity.py`.
- `04_holdout_queue`: new independent lag validation and event-driven finite shared-link FIFO model. `python lag_holdout.py`, `python queue_benchmark.py`. See stage README for real-data workflow.

## Variance prediction: three evaluations
Stage 03 (exploratory pilot, `03_pilot_policy/synthetic_variance.csv`): fraction-only 21.66%, HAC 26.56% mean absolute relative error over 72 settings. Stage 04 and the fresh baseline comparison below use different ensembles; no uniform improvement is established.

### Stage 04
A fresh fit/validation/test split (48/64/96 trajectories per regime, N=16000) selected each model's Bartlett lag from 100, 300, 1000, 2000 on validation variance only. Mean absolute relative deviation on fresh test variance, 72 settings: fraction-only 17.06%, long-run covariance 10.07%. By coupling-regime persistence .9/.98/.995: fraction 14.55/13.33/23.29%; long-run 11.27/7.17/11.76%. Reused signals across conditions; no formal inferential claim or proof of universal superiority. Lag tuning uses independent synthetic trajectories and may not transfer to a single nonstationary recording.

An event-driven FIFO with finite capacity eight packets, service times 6/9/12 ms, data generation every 10 ms, abstract equal-sized packet units, shared forward/reverse queue, propagation 8 ms, independent post-service erasures .1/.3, deadlines 50/100 ms, 40 trajectories per setting. At most six offered packet units per four data packets (including control). Actual offered rates differ by mode. XOR uses parity after each group; NACK can request at most one missing packet after a fixed group timeout. No real TCP/RTP stack, cross traffic, actual byte budget, or congestion-control algorithm is simulated. Queue overflow is modelled, not measured. At 10% erasure, 100 ms deadline, service time 6 ms, delivery fraction none/XOR/NACK 89.83/96.43/96.77%; TE RMSE versus complete same-trajectory estimate .001363/.000690/.000945 bits. At 9 ms, respective delivery 89.83/92.26/89.90%; RMSE .001363/.001221/.001672 bits. At 12 ms the offered originals alone exceed capacity. These are illustrative, not a claim that a particular strategy always wins. No inference of anatomy, significance testing, or real link performance.

## Data and remaining validation
The original ASDF file is listed at https://github.com/ShimonoMLab/NetworkEstimate_WithEICateg/blob/master/spike.mat (Git blob SHA 5b8cb735cc72680956aa603f0274d71b532e851c). The recording has not been evaluated in these synthetic benchmarks. Verify recording provenance, index base and units before running the adapter. `real_multichannel.py` selects 4 active channels using the pilot half, evaluates 12 ordered pairs on held-out blocks; loader passed an artificial ASDF fixture test only. Actual recording analysis is outstanding. Physical two-host tests require two authorized machines; the echo probe is not a physical-network TE/FEC integration. Validation on verified independent recordings, real-network protocol integration and broader generalization remain outstanding. Stages 05–07 contain synthetic informative-loss, localhost checks and independent null calibration; the literature comparison is targeted rather than exhaustive.

## License and data
No license is assigned by this subdirectory. Reuse is subject to the repository owner’s applicable terms; publicly visible code is not automatically open-source licensed. No raw neuronal recordings are distributed.

## Stage 05: state-dependent packet erasures
Run `cd 05_state_dependent_loss && python informative_loss.py`. The synthetic benchmark estimates state-specific inclusion probabilities on a separate pilot and evaluates inverse-probability-weighted TE on 128 independent trajectories per mechanism. See the stage README for assumptions, numerical results, negative controls and limitations. Raw spike data are not included.


## Stage 06: packet-to-TE integration
See [06_udp_integration](06_udp_integration/README.md) for an actual localhost socket payload-to-TE benchmark with 48 tests on two reproducibly generated binary arrays (no measured data); [prior-art comparison](06_udp_integration/literature_comparison.md) delineates related methods and limits. No verified recording or two-physical-host run is claimed.

## v74 independent baseline comparison
`cd 04_holdout_queue && python direct_hac_holdout.py` generates `direct_hac_holdout.csv`. Fit/validation/test ensembles: 64/96/192 trajectories; 72 settings. Average absolute relative error: fraction 12.64%, factorized HAC 15.45%, direct masked HAC 15.51%. The earlier favorable comparison did not replicate. All three models use a complete-pilot-derived score; the direct baseline uses the masked score autocovariance. See the manuscript for derivation and limits.

Generate the manuscript figures from the saved CSVs with `python make_figures.py` from the repository subdirectory.

## v75: independent null calibration
See [07_null_calibration](07_null_calibration/README.md). New fixed-seed simulation (800 calibration and 800 independent evaluation trajectories per condition) shows that a complete-record threshold can be invalid after erasures, even for mask-aware TE. All inputs are synthetic; the matched null assumes the known data-generating mechanism.

## Verification
Run `python verify_results.py` to check archived result aggregates, input provenance, and package integrity. For a fresh end-to-end synthetic localhost test, run `python verify_results.py --rerun-udp`. The recorded verification in `verification.json` is not independent external replication. See `REPRODUCIBILITY.md` for exact scope and links.

## Stage 08: cross-environment and virtual-network verification
See [08_replication](08_replication/README.md). Forty-eight new trials across separate Linux network namespaces connected by virtual Ethernet on one physical host reproduced all archived deterministic UDP outcomes under a different NumPy version. A separate entropy-based TE implementation agrees within 1e-12 bits. External CI and physical-host replication remain distinct outstanding tasks.
