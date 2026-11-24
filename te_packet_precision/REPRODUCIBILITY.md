# Reproducibility and scope (v76)

Install dependencies with `python -m pip install -r requirements.txt`. `python verify_results.py` verifies hashes and numerical aggregates; `python verify_results.py --rerun-udp` additionally regenerates both synthetic inputs and reruns 48 UDP trials, comparing all deterministic fields with the archived results. In `verification.json`, the rerun uses Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0. Exact floating-point and file hashes may depend on software versions; compare within tolerances when using other environments. The listed timings are host-dependent. The protocol is an educational stop-and-wait implementation on loopback, not RTP/RTCP or a real network benchmark. Physical delay, reordering and two-host performance remain untested.

| Claim | Source file |
| --- | --- |
| Stage 03 variance prediction errors 21.66%, 26.56% | 03_pilot_policy/synthetic_variance.csv |
| Stage 04 variance prediction errors 17.06%, 10.07% | 04_holdout_queue/holdout_lag.csv |
| Independent recheck 12.64%, 15.45%, 15.51% | 04_holdout_queue/direct_hac_holdout.csv |
| State-dependent loss and residual bias | 05_state_dependent_loss/informative_summary.csv |
| Synthetic UDP 48 trials | 06_udp_integration/synthetic_benchmark.py, metadata.json, socket_results.csv, socket_summary.csv |
| Null calibration | 07_null_calibration/null_power.csv |

Earlier recording-based analyses are separate: [root results for recording 2](https://github.com/ShimonoMLab/NetworkTM_Analysis/blob/main/results_loss_groundtruth_ds2.jsonl) and [record-length experiment](https://github.com/ShimonoMLab/NetworkTM_Analysis/blob/main/results_length_growth.jsonl). These files contain aggregated outcomes for paired surrogate controls and are not raw recordings or anatomical ground truth. The new synthetic tests do not establish the provenance of earlier unverified arrays; those arrays and results are not included here.

### v77 cross-environment evidence
See `08_replication/virtual_results.json`, `virtual_results.csv`, `virtual_link.sh`, `virtual_link.py`, and `check_aggregates.py`. All 48 virtual-link trials passed, independently computed TE agrees to within 1e-12 bits. Same physical machine, separate processes and network stacks; not a third-party or physical two-host test.

State-dependent loss was regenerated on NumPy 2.2.4: 768 replicate and six aggregate rows numerically identical. `08_replication/verify_remote.py` checks the exact released file tree against GitHub main after uploading. It has not returned a successful remote match because the current public tree is an older release.
