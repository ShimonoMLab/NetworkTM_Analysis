# Transfer entropy under packet erasures: variance benchmark

This standalone synthetic-data benchmark compares an observed-fraction-only variance approximation with a long-run covariance approximation. No real neuronal recording, external network trace, actual network transmission, or proprietary data is used. These are research prototypes, not evidence of a novel theorem or a validated communication protocol.

## Reproduce
Python 3.10+ and NumPy 1.24+; the scripts use fixed pseudorandom seeds. Run in this directory:

```sh
python variance_validation.py
python switching_validation.py
python confirm.py
```

The first two scripts produce condition-level and replicate-level CSVs; the third produces `confirmation.csv` (including marginal percentile bootstrap intervals from 4,000 resamples of 1,000 independent trajectories per condition). Approximation errors are absolute relative deviations from the Monte Carlo sample variance, not prediction errors on real data. Signals are reused across loss conditions, so condition counts must not be interpreted as independent trajectories across all conditions. Initial states are generated independently rather than from the exact joint stationary distribution; finite-record initialization bias is a limitation. N is the number of time points, and the estimator uses N-1 potential lagged pairs.

## Models

The first experiment uses X as a binary Markov process (stationary activity 0.3, persistence 0 or 0.9) and Y with conditional probability 0.16+0.18 Y_previous+b X_previous (b=0.08 or 0.25). The second experiment uses an unobserved symmetric binary Markov regime Z (persistence 0.9, 0.98, or 0.995), iid X, and conditional Y probability 0.12+0.18 Y_previous+(0.02+0.48 Z_previous)X_previous. The coupling regime is not included as a conditioning variable in TE. The observed process is stationary in the population model, even though the coupling changes over time. TE is the lag-one conditional mutual information I(X_previous;Y_current|Y_previous), estimated by empirical cell frequencies on valid triples.

All channels share a packet erasure mask. Packet length is 10 samples; loss probability is 0, 0.1, or 0.3. The binary Markov packet success process has persistence parameter r=0 (independent losses), 0.8, or 0.95, with P(success next | success current)=1-p+p*r. Only consecutive time points that are both present are used. Missingness is generated independently of the signal. There is no FEC or NACK implementation in this new benchmark; it studies the residual loss process rather than protocol performance.

Let W_t be the indicator that t and t-1 are observed, f=E W_t averaged over packet phase, psi_t the centered population pointwise conditional mutual information, and gamma_k=Cov(psi_t,psi_{t+k}). Under the specified independent mask, phase averaging and regularity assumptions (nonzero TE, positive cell probabilities, mixing), the asymptotic variance approximation is

V_L = [f gamma_0 + 2 sum_{k>=1} E(W_t W_{t+k}) gamma_k]/[(N-1) f^2].

The simpler comparison is V_f=L0/[(N-1)f], where L0=gamma_0+2 sum gamma_k. For packet Markov success probability q=1-p, the joint probability that all distinct required packets i_1<...<i_m succeed is q product_j(q+p*r^(i_{j+1}-i_j)). Average this over the ten within-packet phases to compute E(W_t W_{t+k}). Signal autocovariances come from the exact 8- or 16-state population transition matrix. Infinite sums are truncated at 400 or 2,000 lags respectively. Both approximations use known population parameters; this is an oracle benchmark, not a practical estimator from unknown signals. TE=0, informative loss, estimated HAC variance, detection thresholds, real neural recordings and actual link performance remain to be tested.

## Results

- Fixed coupling: 24 conditions, 600 trajectories per condition, 8,000 time points. Across 20 nonzero-loss conditions, mean absolute relative error of the two variance predictions is 5.00% (fraction only) and 5.19% (long-run). No improvement demonstrated.
- Switching coupling: 18 conditions, 600 trajectories per condition, 8,000 time points. Across 15 nonzero-loss conditions, corresponding error is 19.49% vs 7.11%. No universal superiority claimed.
- Fresh-seed confirmation: 4 conditions, 1,000 trajectories, 16,000 time points. For regime persistence 0.98 and independent loss 30%, empirical variance is 6.4683e-5 bit^2 (marginal 95% bootstrap interval 5.9105e-5 to 7.0531e-5); V_f=8.7175e-5 (34.77% relative error), V_L=6.6070e-5 (2.14%). For correlated loss r=0.8 the errors are 11.85% and 3.02%; at r=0.95 they are 1.56% and 3.61%. No multiple-testing adjustment or formal model-selection claim. The practical implication is conditional: temporal signal and loss structure can matter for estimating TE precision, beyond the observed fraction.

Independent validation of the derivation and benchmarks remains necessary.
