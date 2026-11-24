# TE precision under packet erasures: pilot-estimated benchmark (v70)

## Status and provenance
This is an experimental prototype, not a validated communication system. No real recording was processed in this run. ShimonoMLab/NetworkEstimate_WithEICateg contains `spike.mat` on branch `master` (7,448,054 bytes, Git blob SHA `5b8cb735cc72680956aa603f0274d71b532e851c`). The repository API confirmed the file, but the binary could not be retrieved by the available tools. The raw recording is not distributed in this package. The ASDF input adapter passed a synthetic MATLAB fixture test (eight blocks per split), which is not a test on the real recording.

Repository: https://github.com/ShimonoMLab/NetworkEstimate_WithEICateg
Data: https://github.com/ShimonoMLab/NetworkEstimate_WithEICateg/blob/master/spike.mat

## Reproduce
Python 3.10+, NumPy, SciPy. In this directory:

```
python -m pip install -r requirements.txt
python validate.py
python bandwidth_sensitivity.py
```

For local execution with the original data, download `spike.mat` from the above repository and run:

```
python validate.py --asdf /path/to/spike.mat --index-base 1 --bin-ms 1 --lag 300
```

Check the recording's ASDF index convention (0 or 1), recording duration and bin units before interpreting results. The adapter selects the two highest-count neurons using only the first-half pilot blocks, uses one ordered pair, and evaluates the second half as contiguous non-overlapping blocks. These blocks need not be independent or stationary. This is a descriptive single-pair pilot, not a generalization or anatomical-ground-truth test. Very sparse data and TE near zero undermine first-order asymptotic approximations; do not interpret the output as proof of connectivity. Extend to multiple prespecified pairs and independent recordings, inspect uncertainty and stationarity, and verify the data-use terms before publication. For alternate bin widths, packet timing is illustrative, not a calibrated physical network model.

## Methods and results (executed)
Three synthetic Markov-modulated coupling conditions (persistence 0.9, 0.98, 0.995), 96 pilot and 192 independent test trajectories per condition, 16,000 samples each. Erasures are generated independently of signal, with rates 0.1 or 0.3 and persistence 0 or 0.8. Three transmission strategies and two deadlines (50 and 100 ms) yield 72 test settings. Signals and masks are reused across strategies for paired comparisons. Pilot estimates: empirical joint probabilities with half-count regularization, centered local conditional mutual information and autocovariance, Bartlett lag window K=300, and mask pair probabilities from pilot masks. No true population parameters used for estimation. Test variance is the empirical sample variance of masked plug-in TE across trajectories; it is not variance of a structural-connectivity estimator. Reported errors are absolute relative deviations of the variance approximation from this empirical variance. No formal significance claim.

Mean absolute relative error across 24 settings per regime (fraction-only / long-run covariance): 0.9: 10.75% / 8.12%; 0.98: 12.05% / 21.02%; 0.995: 42.19% / 50.53%. Overall 21.66% / 26.56%. The empirical long-run model does not generally outperform the simpler model. `lag_sensitivity.csv` evaluates K=100,300,1000,2000 on the SAME slow-regime data split. The long-run error decreases from 50.53% at K=300 to 10.65% at K=2000 (fraction-only at K=2000: 11.28%). This exploratory sensitivity is not independent validation or an objectively selected optimal bandwidth. Select the lag from pilot data only and test on fresh trajectories before asserting improvement.

## Policy selection
The model uses a fixed paced six-slot schedule for four data packets, with at most two repair/control opportunities, not a continuous-traffic queueing model. Fixed 10 ms slot interval, 10 ms forward delay, 5 ms feedback propagation. XOR sends one parity packet; NACK sends at most one equal-sized feedback packet and one retry. All transmissions including feedback are counted in a common cap of 1.5 packet-equivalents per data packet; actual consumption differs by method. Loss on feedback uses the same abstract slot-erasure process; real bidirectional links need not behave this way. The policy minimizes pilot-estimated TE variance among the three modes; it does not guarantee a false-positive constraint, detection power, or throughput in a real network. Across 24 settings, the fraction and long-run policies match the empirical test-set minimum in 17 and 18 settings respectively, with mean ratio to the empirical minimum 1.003865 and 1.003144. Choices agree in 23/24 settings. The test-set minimum is a retrospective noisy reference, not a deployable oracle. No evidence of a large selection advantage. `synthetic_policy.csv` contains all decisions.

## Two-host link probe
`link_probe.py` is a small UDP echo probe with payload and sequence verification and sender-clock RTT. It does not implement FEC or TE estimation on a physical network. It was run with distinct client/server processes on the SAME host (loopback): 100/100 matching echoes, median RTT 0.146378 ms. It is not evidence of a two-device measurement. To run on two machines on an authorized local network, use the receiver's actual private IP address, substituting it for ADDRESS below, with the chosen UDP port permitted by local network policy:

Receiver: `python link_probe.py server --host ADDRESS --count 100`
Sender: `python link_probe.py client --host ADDRESS --count 100 --output physical_link.csv`

No one-way latency is estimated, no synchronized clocks assumed, and probe packet rate is modest. The physical-network TE/FEC/NACK integration, queue measurements, equal-byte performance comparison and real-recording validation remain outstanding. Do not report these as completed.

## Files and publication
`validate.py`, `bandwidth_sensitivity.py`, `link_probe.py`: executable prototypes; `synthetic_variance.csv`, `synthetic_policy.csv`, `lag_sensitivity.csv`, `localhost_probe.csv`: results from the current run. Review the derivation, code, license and data terms before publishing; do not include job application materials or unlicensed raw data. No GitHub write was performed.
