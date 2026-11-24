# Cross-environment verification (v77)

A fresh 48-trial UDP benchmark used two separate Linux network namespaces and two processes, connected through a virtual Ethernet pair. Both were on the same physical host; erasures were injected at the receiver application, as in stage 06. This is not a physical-LAN experiment, a standard-compliant RTP implementation, or an independent third-party replication. Kernel `netem` was unavailable, so delay, reordering and actual link loss were not tested.

The system Python 3.13.5 / NumPy 2.2.4 environment differed from the stage-06 Python 3.13.5 / NumPy 2.3.5 environment. Regenerated input hashes, payload equality, application byte and datagram counts, valid/received fractions, and both TE RMSE metrics matched the archive (absolute tolerance 1e-12). A separately implemented entropy-identity TE calculation (H(X,Y)+H(Y,Z)-H(Y)-H(X,Y,Z)) was checked against stage 06's conditional mutual information formula in every trial, with maximum absolute difference 5.643001257243929e-16 bits across 48 x 56 masked TE estimates. Timings are not compared. See `virtual_results.csv` and `virtual_results.json`.

- `python 08_replication/check_aggregates.py`: standard-library independent reaggregation of IPW replicate results, UDP trial rows, and null calibration proportions. No regeneration of the underlying null or IPW trials in this check.
- `PYTHON_BIN=python3 bash 08_replication/virtual_link.sh /tmp/virtual_results.csv`: repeat the virtual-network benchmark on Linux with `unshare`, `nsenter`, `ip`, `ss`, and enabled unprivileged user/network namespaces. No changes to host interfaces; a disposable namespace is used. Requires NumPy. Use a separate output path to preserve archived data.
- For physical two-host trials see stage 06. The virtual experiment is not a substitute for that evidence. Do not expose the experimental unauthenticated UDP protocol on an untrusted network.
- For third-party replication, add the supplied separate repository-root workflow `.github/workflows/te-reproducibility.yml` to the repository. It checks aggregates and repeats the 48 localhost trials on GitHub-hosted Python 3.12 and 3.13. No remote workflow run is claimed until a successful run URL is recorded.

## State-dependent loss replay
The original `05_state_dependent_loss/informative_loss.py` was also rerun from a temporary copy under system NumPy 2.2.4. All 768 replicate rows and all six summary rows agreed with the archived numeric fields exactly (`ipw_replay.json`). This is cross-library replay on the same host, not independent external replication.

## Verify remote publication
From the unmodified extracted release folder, run `python 08_replication/verify_remote.py` after upload. This read-only script compares all local Git blob hashes with the current public main tree; missing, extra, or changed files produce a nonzero exit code. Generated data should be written outside the release folder. A remote match is evidence of publication consistency, not evidence of independent scientific validation.
