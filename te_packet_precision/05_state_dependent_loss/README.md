# Stage 05: state-dependent packet erasures and transfer entropy

Run `python informative_loss.py` (Python 3.10+, NumPy). The script generates all data; no external or recorded data are needed. Outputs `informative_summary.csv` and `informative_replicates.csv`. Fixed seeds. Results are preliminary synthetic benchmarks, not protocol or anatomical validation.

## Experimental design

Three loss mechanisms: independent loss (MCAR), loss conditional on an assumed fully available external binary state, and loss conditional on unobserved packet activity. A persistent binary state modulates source firing and the coupling strength. Each trajectory has 40,000 samples in 10-sample packets. For each scenario, 64 independent synthetic pilot trajectories estimate the state-specific reception probabilities from the state and mask; 128 fresh independent test trajectories are evaluated. Conditional packet reception is independent given the full signal and state in the generator. For the state-dependent scenario, the state is assumed available even for lost packets; transport of state metadata is not simulated. All methods use the same test trajectories; the complete-record estimate is the reference (not a structural connectivity truth).

For binary one-lag TE, let C_t=(X_{t-1},Y_{t-1},Y_t), W_t=M_{t-1}M_t and pi_t=P(W_t=1 | Z), where Z contains the relevant packet states. Within a packet, pi_t is its reception probability; across packets, it is the product of two probabilities **only under conditional independence**. Estimate probabilities of C_t using counts weighted by W_t/pi_t; a fixed 0.5 pseudocount per cell is applied before normalization. If the joint inclusion probability is correctly specified and positive, E[W_t/pi_t | C,Z]=1, so weighted category counts target complete-record frequencies. Consistency requires adequate sampling and ergodicity; weighting does not guarantee finite-sample unbiasedness of nonlinear TE or smaller variance. With unobserved activity-dependent loss, state-only weighting need not identify the complete-data distribution.

## Results

| Loss mechanism | Complete-case RMSE (bits) | State-weighted RMSE (bits) |
|---|---:|---:|
| Independent | 0.002308 | 0.002386 |
| Observed state | 0.046884 | 0.004189 |
| Unobserved activity | 0.041421 | 0.008264 |

For the observed-state setting, the relative RMSE reduction is 91.1%. A paired percentile bootstrap over the 128 independent test trajectories (4,000 resamples, pilot estimates held fixed) gives a 95% interval of [-0.04334, -0.04205] bits for weighted minus complete-case RMSE. The correction is not beneficial in the independent-loss setting; the unobserved-activity setting retains bias (-0.00754 bits). These intervals are conditional on the pilot and simulation model, not population-wide guarantees. The complete-case baseline, not zero-fill, is used. These experiments address point estimation, not tests of edge significance under the null.

The results motivate transmission of reliable state metadata and joint estimation/communication design, but the metadata overhead and reliability must be accounted for in a real implementation. No real neural data are included or newly processed; no two-host experiment was performed. Prior art in missing-data estimation and task-oriented communication means no claim of a novel inverse-probability weighting technique or proof of priority is made.
