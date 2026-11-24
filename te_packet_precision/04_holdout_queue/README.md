# Independent lag validation and queue benchmark

`lag_holdout.py` fits on 48 independent synthetic trajectories, selects each lag using 64 different validation trajectories, and evaluates on 96 fresh test trajectories per persistence regime. `holdout_lag.csv` has 144 rows (72 settings x 2 models). `queue_benchmark.py` has 36 settings, 40 trajectories each; see parent README for model boundaries. No actual neuronal recording or physical link was processed for these new experiments.

## Real recording (not yet executed)
Download the author's ASDF recording from the public repository. Verify the recording's index base (1 or 0) and 1-ms bin convention. Keep raw data outside the repository. Example *if* one-based:

```sh
python real_multichannel.py /path/to/spike.mat --index-base 1 --block-cap 32 --lag 1000 --out real_results
```

This selects the four most active channels in the pilot half, 12 directed pairs, up to 32 evenly spaced nonoverlapping blocks per half, and runs the fixed-paced packet-loss benchmark from `validate.py`. It does not implement the event-driven queue for the real recording, or prove structural connectivity. Blocks from one recording may be correlated or nonstationary; lag is a prespecified illustrative choice. Sparse or nearly zero TE makes asymptotic approximations unreliable. The loader was tested on a generated MATLAB fixture, not the real dataset. Inspect the outputs and uncertainty before reporting.

## Physical link
The optional `link_probe.py` is an echo probe, not a full TE transmission benchmark. Run the server and client on separate authorized local computers, with the receiver's actual private IP address and permitted port (see stage 03 README). No public server or port exposure is needed. Physical integration and real performance testing remain incomplete.
