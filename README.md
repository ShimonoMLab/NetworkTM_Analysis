# NetworkTM_Analysis

Exploratory analyses of partially observed multivariate time series, applied
with the same procedures to communication-network traffic matrices
(GEANT / TOTEM) and to neuronal spike trains, and of how packet loss in
streaming changes a directed network estimated from the received data, with
and without telling the estimator where packets were lost.

一部しか観測できない多変量時系列からの推定を、同じ手続きで、通信ネットワークの
トラフィック行列（GÉANT, TOTEM）と神経細胞のスパイク列に適用した探索的な解析です。
あわせて、伝送中のパケット欠落が、受信データから推定する有向ネットワークをどう変えるか、
欠落の位置を推定側に渡すと何が変わるかを調べています。

## Files / 構成

| file | role |
| --- | --- |
| `tmcore.py` | shared procedures on a time × channel matrix or on spike-time lists: spectrum, recovery of hidden entries (soft-impute and baselines, random/block hiding, validation-based threshold), transfer entropy with per-pair surrogates and Benjamini–Hochberg FDR, anomaly candidates |
| `te_hourly.py` | timestamp-aware TE test for hourly series: resampling on the clock, hour-of-week profile from timestamps, only consecutive observed hours, optional conditioning on the network mean |
| `analysis.py` | traffic matrices (GEANT OD matrix, flat CSV) |
| `robustness_te.py` | robustness of the TE test on GEANT and on synthetic series with known couplings (section 3) |
| `conditioning_check.py` | whether conditioning on the network mean removes true couplings (section 3) |
| `te_masked_spikes.py` | binary TE for spike trains using only received bins (loss mask from sequence numbers) |
| `null_check.py` | calibration of the spike-train TE test on pairs that cannot be coupled (section 5) |
| `loss_groundtruth.py` | packet loss judged on null and planted pairs: false positives and detection (section 6) |
| `loss_length.py` | loss vs recording length (growth) and the equivalent-length test of mask-aware estimation (section 7) |
| `transport_choice.py` | T*(s, D) = T*_0 / f(R_s(D)) for mask / parity / retransmission by deadline (section 7) |
| `packet_silence.py` | how often all channels are silent in a 10-ms packet, per recording (section 6) |
| `loss_decompose.py` | data loading, loss patterns (independent, bursty, activity-dependent), calibrated networks of 30 / 150 min (section 5) |
| `loss_spikes.py`, `loss_masked_experiment.py` | earlier loss experiments (z-test; superseded, kept for the record) |
| `signed_lte.py` | signed local TE, its identity, and a toy check on fluid queue models (section 4; working hypothesis) |
| `analysis_spikes.py` | spike trains in ASDF form (MATLAB `.mat`, 1-ms bins), as used in [ShimonoMLab/NetworkEstimate_WithEICateg](https://github.com/ShimonoMLab/NetworkEstimate_WithEICateg) |

```
pip install -r requirements.txt
python analysis.py path/to/geant-flat-tms.csv --seeds 10                 # ~15 min
python analysis_spikes.py path/to/spike.mat --out results_spikes          # ~10 min
python analysis_spikes.py path/to/asdf.mat --labels path/to/cell_categ.mat  # E/I summary
python robustness_te.py path/to/geant-flat-tms.csv                       # ~15 min
python signed_lte.py                                                      # ~2 min
python conditioning_check.py                                              # ~20 min
python loss_spikes.py path/to/spike.mat                                   # ~10 min
python te_masked_spikes.py                                                # self-test
python null_check.py path/to/spike.mat --a 0 --b 1                        # ~15 min
python loss_groundtruth.py path/to/spike.mat --c 0.02                      # ~20 min
python loss_groundtruth.py path/to/spike.mat --c 0.02 --long --conds none burst-zero burst-mask
python loss_length.py path/to/spike.mat equiv --rep 0 1 2 3 4             # ~30 min
python loss_length.py path/to/spike.mat growth --rep 0 1 2                 # ~25 min
python transport_choice.py                                                 # ~1 min
python loss_groundtruth.py path/to/asdf.mat --c 0.02 --rep 0 1 2 3 --pairs 0,1 2,3 1,2 3,0 \
    --conds none none20 burst-zero burst-mask indep-zero indep-mask activity-zero activity-mask \
    --dataset ds2 --out results_loss_groundtruth_ds2.jsonl                 # ~80 min
```

## 1. Traffic matrices (GEANT)

GEANT intradomain traffic matrices: 23 nodes, 529 OD pairs, 15-min bins,
Jan–Apr 2005 (TOTEM project; S. Uhlig, B. Quoitin, J. Lepropre, S. Balon,
ACM SIGCOMM CCR 36(1), 2006). Not redistributed here; a flat CSV copy
(`geant-flat-tms.csv`, one row per 15-min bin, 529 columns) is available in
public mirrors of the TOTEM dataset.

- Spectrum: on the raw scale one OD flow carries 99.99% of the
  variance energy; on the log scale 173 / 262 of 529 components are needed
  for 90% / 95%. The matrix is not low-rank in a strict sense.
- Recovery of hidden OD entries (hourly, log scale, 1,500 h × 529; NMAE,
  mean of 10 masks; sd in `results.json`):

| hiding | hidden | soft-impute | interp. in time | time-of-day | per-flow mean |
| --- | --- | --- | --- | --- | --- |
| random | 30% | 0.28 | 0.18 | 0.53 | 0.57 |
| random | 50% | 0.31 | 0.19 | 0.53 | 0.57 |
| random | 70% | 0.38 | 0.22 | 0.54 | 0.57 |
| random | 90% | 0.90 | 0.30 | 0.58 | 0.58 |
| block | 30% | 0.36 | 0.39 | 0.53 | 0.58 |
| block | 50% | 0.42 | 0.39 | 0.53 | 0.58 |
| block | 70% | 0.53 | 0.42 | 0.54 | 0.59 |
| block | 90% | 0.98 | 0.47 | 0.62 | 0.62 |

  Linear interpolation in time is best with entry-wise hiding at every rate.
  With 48-h blocks the gap narrows; soft-impute is slightly better at 30%.
  (These recovery runs take every 4th 15-min row; the 1,500-row stretch
  contains one 7-h gap in the data, which is bridged.)
- Transfer entropy between node-level outgoing traffic, resampled hourly on
  the clock (2,849 h, 151 missing), hour-of-week profile removed using the
  timestamps, only consecutive observed hours, per-pair surrogates shifted
  by ≥ 7 days, FDR q = 0.05: 9 of 506 ordered pairs significant. These p-values come from a z-test
  against 200 surrogates; with empirical p-values from 1000 surrogates none
  of the 506 pairs passes (section 3), so the 9 pairs are not a firm result
  (compare the calibration of the spike-train test, section 5). Earlier
  versions reported 215; that count came from removing the weekly profile
  by row index, which goes out of phase after the gaps in the data (7 h,
  4.8 days, 33 h). See section 3.
- Anomaly candidates per OD pair (z > 6 after removing the daily profile):
  916.

## 2. Neuronal spike trains (ASDF)

Example: `spike.mat` from ShimonoMLab/NetworkEstimate_WithEICateg
(Kajiwara et al., PLOS Comput Biol 2021): 386 neurons,
9,000,018 1-ms bins (2.5 h), 2,660,857 spikes. Results in `results_spikes/`.

- Spectrum of firing counts (10-s bins, all neurons): raw 87 / 166,
  log 225 / 276 of 386 components for 90% / 95%.
- Recovery of hidden entries of the firing-count matrix (10-s bins, log1p,
  900 × 386; NMAE, mean of 10 masks; no periodic baseline):

| hiding | hidden | soft-impute | interp. in time | per-neuron mean |
| --- | --- | --- | --- | --- |
| random | 30% | 0.29 | 0.32 | 0.33 |
| random | 50% | 0.31 | 0.32 | 0.33 |
| random | 70% | 0.32 | 0.32 | 0.33 |
| random | 90% | 0.63 | 0.33 | 0.33 |
| block | 30% | 0.31 | 0.34 | 0.33 |
| block | 50% | 0.33 | 0.34 | 0.33 |
| block | 70% | 0.38 | 0.34 | 0.33 |
| block | 90% | 0.83 | 0.36 | 0.35 |

  Here cross-neuron (low-rank) structure gives the best recovery at 30–50%
  hidden with entry-wise hiding and at 30% with block hiding, while temporal
  interpolation gains little over the per-neuron mean. The gains are modest:
  single-neuron counts are noisy at this resolution.
- Transfer entropy between spike trains: binary TE (1-ms bins, history 1)
  at delays 1–30 ms, peak over delays, as in the delayed TE of Ito et al.
  (2011) and the GPU code of NetworkEstimate_WithEICateg. For each ordered
  pair, 100 surrogates shift the source circularly by ≥ 1 s, and the peak
  over delays is taken for every surrogate as well, so scanning 30 delays
  does not inflate the test; one-sided z-test; FDR q = 0.05.
  On the 40 neurons with the most spikes, within the first 30 min (CPU run time),
  18 of 1560 ordered pairs were significant with the z-test used up to
  version 3.4; that test is anti-conservative and with the calibrated test
  none of the 1560 pairs is significant (section 5).
  The fast counting implementation (`te_spike_delays`) reproduces a direct
  dense-array computation exactly on synthetic data with a known 5-ms delay.

## 3. Robustness of the TE significance test / 有意判定の頑健性

`robustness_te.py` (results in `results_robustness.json`) changes one
setting at a time on GEANT, and applies the same test to synthetic series
whose directed couplings are known.

**Where the earlier 215 came from.** The GEANT file has gaps of 7 h,
4.8 days and 33 h. Taking every 4th row and removing the hour-of-week
median by row index puts the weekly phase out of step after each gap, so a
weekly pattern shared by all nodes is left in the residuals.

| treatment of time | significant / 506 |
| --- | --- |
| every 4th row, weekly profile by row index (versions up to 3.1) | 215 |
| every 4th row, weekly profile from timestamps, gaps bridged | 41 |
| hourly on the clock, weekly profile from timestamps, only consecutive observed hours (now) | 9 |

**GEANT, corrected series.** Jaccard = overlap with the baseline set (with so
few pairs it moves a lot when one pair changes).

| setting | significant / 506 | Jaccard |
| --- | --- | --- |
| baseline (4 bins, 200 surrogates, shift ≥ 7 d, z-test, q = 0.05) | 9 | 1.00 |
| 1000 surrogates / shift ≥ 14 d / another seed | 5 / 9 / 7 | 0.56 / 0.50 / 0.78 |
| empirical p (1000 surrogates) / q = 0.01 | 0 / 0 | – |
| 3 bins / 6 bins | 20 / 0 | 0.26 / – |
| + 7-day moving average removed | 1 | – |
| 3 bins, conditioned on the mean of the other nodes | 4 (vs 20 unconditioned) | – |
| + 7-day trend removed, 3 bins, conditioned | 3 (vs 9 unconditioned) | – |

Once the weekly pattern is removed on the clock, at most a few percent of
node pairs show a directed dependence at the hourly scale, and almost none
survive removal of slow drift or conditioning on network-wide variation.
Hourly node totals of a backbone network are dominated by common demand;
this data set says little about directed dependence between nodes.

**Synthetic series** (23 nodes, 2,849 h; stable AR(1) dynamics, sparse lag-1
directed couplings, a shared daily/weekly cycle and a shared slow random
drift; 5 seeds per row). False positives are counted on pairs with no
directed path.

| couplings | treatment | false positives (no path) | recall (direct links) |
| --- | --- | --- | --- |
| none | weekly profile removed | 0.4–74% (depends on the drift) | – |
| none | + 7-day trend removed | 0–0.4% | – |
| none | conditioned on network mean | 0.2–0.6% | – |
| sparse | weekly profile removed | 6.8–71% | 100% |
| sparse | + 7-day trend removed | 0.6–6.3% | 100% |
| sparse | conditioned on network mean | 0.9–2.3% | 100% |
| sparse | trend removed and conditioned | 0.4–3.6% | 95–100% |

Circular-shift surrogates do not remove a drift shared by all nodes; left
in, it makes most uncoupled pairs significant. Removing slow trends or
conditioning on the network mean keeps false positives at a few percent
while all (or nearly all) direct links are still found; indirect,
multi-step dependencies are mostly not flagged. The synthetic model is
linear and simple: a check of the test, not a model of traffic.

欠測区間で週内周期の位相がずれ、全ノードに共通の周期が残ったことが、以前の215対の主因でした。
時刻に沿って補正すると有意な対は9対となり、ゆっくりした変動の除去やネットワーク全体の変動による
条件付けでほぼ消えます。配線を与えた模擬データでは、同じ処理で誤検出を数％に抑えつつ、直接の結合を
ほぼすべて検出できました。


**Does conditioning remove true dependencies?** `conditioning_check.py`
(results in `results_conditioning.json`): synthetic series with weekly
profile and 7-day trend removed, 3 bins, 2 seeds per row; "hub" adds one
source driving 12 nodes. Each cell: unconditioned / conditioned on the mean
of the other nodes.

| topology | coupling | recall, direct links | recall, hub links | false positives (no path) |
| --- | --- | --- | --- | --- |
| sparse | 0.1 | 28%–32% / 3%–5% | – / – | 0%–3% / 0% |
| sparse | 0.2 | 100% / 81%–85% | – / – | 2% / 0%–1% |
| sparse | 0.3 | 100% / 100% | – / – | 2%–6% / 1%–2% |
| hub | 0.1 | 15%–18% / 0%–2% | 17%–33% / 0%–8% | 1% / 0%–1% |
| hub | 0.2 | 84%–85% / 34%–44% | 67%–100% / 33%–58% | 1%–2% / 0% |
| hub | 0.3 | 100% / 90%–96% | 100% / 75%–100% | 2%–5% / 1% |

Conditioning lowers false positives but also removes weak couplings and,
markedly, links from a hub that drives many nodes (the hub moves the
network mean it is conditioned on). So a
small count after conditioning is not evidence that dependencies are absent.
It is used as a check, after the series have been put on the clock and slow
drift has been removed.

ネットワーク全体の変動による条件付けは誤検出を減らす一方、弱い結合や、多くの節点を駆動するハブからの結合も消します。条件付け後に有意な対が
少ないことは、依存がない証拠にはなりません。時刻に沿った補正とゆっくりした変動の除去を基本とし、条件付けは
確認に用います。

## 4. Signed local TE: working hypothesis / 符号付き局所TE（作業仮説）

`signed_lte.py`. A rise of a series plays the role of a spike in the sorted
local TE of Kajiwara et al. (2021): u_t = 1 if x rises at t, 0 otherwise;
w_t likewise for y. Local TE (Lizier et al., 2008)

    i_t = log2 p(w_{t+1} | w_t, u_t) − log2 p(w_{t+1} | w_t)

is multiplied by s_t = +1 when w_{t+1} = u_t and −1 otherwise, and averaged:
SLTE = mean_t s_t i_t.

Identity (checked numerically in the code). For each history state
w_t = a, let d_a = p(w' = 1 | a, u = 1) − p(w' = 1 | a, u = 0). All terms with
w_t = a have the sign of d_a, so

    SLTE = Σ_a p(w_t = a) · sign(d_a) · E[ |i_t| | w_t = a ].

The ordinary TE is E[i_t], in which positive and negative local values
cancel; SLTE adds their sizes with the sign of the association. It is
positive when a rise of x makes a rise of y more likely in every history
state, and negative when it makes it less likely.

Toy fluid models (not TCP, not ns-3), 20,000 steps, 3 seeds, two-sided
circular-shift test (500 surrogates):

| model | series | expected | obtained | significant |
| --- | --- | --- | --- | --- |
| two queues in tandem | queueing delay | + | + | 3/3 |
| two AIMD-like senders, one bottleneck, back-off one step after the overflow | throughput | − | − | 3/3 |
| the same, back-off within the same step | throughput | − | + | 3/3 (wrong sign) |
| independent queues | queueing delay | 0 | ± | 0/3 |

Propagation and competition are told apart when the response takes at
least one sampling step. When the back-off happens within the step, the
lagged measure cannot see it, and the rate increases shared by both senders
push the sign positive. Measurements must therefore resolve times shorter
than the round trip; whether this holds for real TCP flows is to be checked
in the simulator and testbed.

単純な流体モデルでは、直列の待ち行列での波及は正、1刻み遅れて応答する帯域の奪い合いは負として
有意に現れました。同じ刻みの中で応答が完結する場合は、遅れを使う量では見えず、両者に共通する
送信量の増加によって符号が正になりました。往復遅延より細かい時間分解能の計測が前提となります。
実際のTCPでの確認はこれからです。

## 5. Calibration of the spike-train TE test / 有意判定の較正

`null_check.py` (results in `results_null_check.jsonl`, both the z-test and the pooled-null values; arrays in `null_01.npz`).
The 40 neurons of one 30-min segment and the same 40 neurons of another
30-min segment are placed side by side on a common clock (80 trains). The
3,200 pairs that cross the two segments **cannot be coupled**, so every
significant crossing pair is a false positive of the whole procedure.

| p-value from | crossing pairs p < 0.05 | p < 0.01 | p < 0.001 | FDR-significant (q = 0.05) |
| --- | --- | --- | --- | --- |
| z-test against the 50 surrogates of each pair (versions up to 3.4) | 7.4% | – | 1.84% | 49 |
| pooled standardized surrogates (`tmcore.pooled_null_p`, now) | 5.2% | 1.06% | 0.03% | 0 |

The z-test treats the surrogate peaks as normal; the peak of TE over 30
delays has a heavier right tail, so small p-values were too frequent
(about 18 times the nominal rate at p < 0.001). The pooled null standardizes
each pair's observed peak by its own surrogates and each surrogate by the
other surrogates of the same pair (leave-one-out), and pools the standardized
surrogates of all tested pairs; it keeps the tail of the null instead of
assuming one. **Every spike-train count in versions up to 3.4 (section 2:
18 of 1560 pairs; the earlier loss sections: "about one third") rests on the
z-test and is withdrawn.** With the calibrated test, the 40 most active
neurons give 0 FDR-significant pairs of 1560 both in the first 30 min and in
the whole 150 min (`loss_decompose.py`, `decomp/`); the fraction of pairs with
p < 0.01 is 0.9% at 30 min and 1.9% at 150 min, i.e. dependence exists in
aggregate but no single pair passes FDR with this statistic (binary, 1-ms
bins, history 1, peak over 1–30 ms). This concerns this CPU pipeline and
neuron subset, not the published analyses.

検定の較正：結合しえない組（別の時間帯の細胞どうし）で、従来の z 検定は名目より
はるかに多く有意を出していました（p < 0.001 で約18倍）。代理データを全組でまとめる
判定に改め、名目どおりになることを確かめました。版3.4までのスパイク列の有意数と
「約3分の1」は、この z 検定に基づくため撤回します。

## 6. Packet loss judged against a known truth / 正解の分かる組での欠落の評価

`loss_groundtruth.py` (results in `results_loss_groundtruth.jsonl`). Real
spike trains keep their own statistics, and the truth is set by construction:
**null pairs** join a neuron of one segment and a neuron of another segment
(cannot be coupled); **planted pairs** are targets made of a real neuron of
the second segment plus copies of a source neuron's spikes at a fixed delay
(2–25 ms, jitter 0–2 ms, each spike copied with probability 0.02). All 100
channels travel in the same 10-ms packets, so a lost packet silences every
channel at once. 400 null pairs and 20 planted pairs per replicate, 50
surrogates per pair, pooled null (section 5), BH-FDR over the 420 tests.
Five replicates (segment pairs 0/1, 2/3, 4/1, 1/4, 3/0 of one recording),
each with its own seeds for the planted pairs, the null pairs and the loss
patterns (the replicate index sets them).
Loss rate 10%: independent; bursts (Gilbert–Elliott, mean 10 packets);
activity-dependent bursts (a burst is more likely to start after a packet
with many spikes). Mean over replicates (per-replicate values in the file):

| condition (30 min) | null pairs p < 0.01 (nominal 1%) | p < 0.05 (nominal 5%) | null pairs FDR-significant (sum of 5) | planted detected |
| --- | --- | --- | --- | --- |
| no loss | 1.15% | 4.4% | 8 | 0.87 |
| no loss, first 20 min | 0.75% | 4.2% | 4 | 0.77 |
| no loss, first 10 min | 0.90% | 4.5% | 5 | 0.55 |
| independent, gaps read as silence | 1.30% | 5.0% | 6 | 0.82 |
| independent, mask-aware | 1.10% | 4.9% | 8 | 0.84 |
| bursts, gaps read as silence | **2.20%** | **8.3%** | 15 | 0.86 |
| bursts, mask-aware | 1.10% | 4.6% | 9 | 0.85 |
| activity-dependent bursts, silence | **2.05%** | **8.6%** | 13 | 0.85 |
| activity-dependent bursts, mask-aware | 1.25% | 4.8% | 9 | 0.83 |

Longer recording, one replicate (0–60 min vs 60–120 min), bursty loss:

| condition (60 min) | p < 0.01 | p < 0.05 | null FDR-significant | planted detected |
| --- | --- | --- | --- | --- |
| no loss | 1.25% | 5.8% | 1 | 0.85 |
| bursts, gaps read as silence | **4.75%** | **11.7%** | 9 | 0.80 |
| bursts, mask-aware | 1.50% | 5.2% | 2 | 0.80 |

**Second recording** (`results_loss_groundtruth_ds2.jsonl`): 128 neurons,
130 min (`asdf.mat` of the LSTM study, GenerativeNeurosci_ML-LSTM), the 40
most active neurons, four replicates from 30-min segments (pairs 0/1, 2/3,
1/2, 3/0), same planted strength (0.02), same loss rate (10%):

| condition (30 min) | null pairs p < 0.01 | p < 0.05 | null FDR-significant (sum of 4) | planted detected |
| --- | --- | --- | --- | --- |
| no loss | 1.50% | 5.8% | 7 | 0.88 |
| no loss, first 20 min | 1.44% | 4.8% | 1 | 0.79 |
| independent, gaps read as silence | **44%** | **55%** | 795 | (0.99, not meaningful) |
| independent, mask-aware | 1.25% | 6.9% | 6 | 0.84 |
| bursts, gaps read as silence | **69%** | **80%** | 1250 | (1.00, not meaningful) |
| bursts, mask-aware | 1.56% | 6.1% | 7 | 0.85 |
| activity-dependent bursts, silence | **69%** | **80%** | 1248 | (1.00, not meaningful) |
| activity-dependent bursts, mask-aware | 1.81% | 6.6% | 7 | 0.85 |

Why the two recordings differ (`packet_silence.py`, `results_packet_silence.json`):
reading a lost packet as silence inserts an
all-channel silence. In the first recording (median rate 2.4 Hz, 0.03 spikes
per channel per 10-ms packet) 26% of packets are naturally silent on all 40
channels, so an inserted silence is common and carries little shared
information. In the second (median 24 Hz, 0.45 spikes per channel per packet)
no packet is naturally silent on all 40 channels, so every lost packet is a
strong common event, and even independent loss makes about half of the
uncoupled pairs look dependent. We expect that how rare an all-silent packet is (set by
the rates, the packet length and the number of channels) governs how badly
gap-as-silence distorts the estimate; with two recordings this is a
hypothesis, to be tested by varying the packet length. The mask-aware estimate returns to the level of the
no-loss estimate in both recordings. In the second recording that level is
itself slightly above nominal (1.5% at p < 0.01, 5.8% at p < 0.05, 1,600 null
pairs; mask-aware 1.25-1.8% and 6.1-6.9%), and the test was not
re-calibrated on null pairs of this recording.

What this shows, with its limits:
- Reading bursty gaps as silence makes channels that cannot be coupled look
  dependent: the false-positive rate is about twice the nominal level at 30
  min and grows at 60 min (one replicate). A distortion that grows with the
  recording is not a sampling error; it does not go away by recording longer.
- Telling the estimator where packets were lost (sequence numbers, also over
  UDP) brings the false-positive rate back near the nominal level, for
  bursts and for the activity-dependent bursts tried here, while detection
  of the planted couplings is nearly unchanged.
- With independent loss neither choice matters much in the first
  recording, but in the second recording gap-as-silence fails badly (above).
  Masking costs data: a
  step needs bins t+1, t and t+1−d all received, so 82% of steps remain
  (88% with bursts). The detection rate under independent loss with the mask
  (0.84) lies between 20 min (0.77) and 30 min (0.87) without loss. The
  difference from 30 min (3 of 100 planted pairs) is within the sampling
  error (about 0.04), so the cost in recording time still has to be measured
  with more planted pairs.
- Validity of the mask-aware estimate: it removes lost bins from the counts,
  so it is unbiased when whether a packet is lost does not depend on the
  values it carries given the rest. Loss that depends on the target's own
  spikes in that packet would violate this; the activity-dependent bursts
  tried here (driven by the previous packet's population count) did not
  break the calibration, but stronger dependence was not tested.
- Within a replicate, the gap-as-silence and mask-aware estimates use the
  same loss pattern (same seed), so the two are compared pair by pair.
- Two recordings (five and four replicates), one planted strength, loss
  rate 10%. Planted couplings are spike copies, not synaptic transmission.

欠落を発火なしと読むと、連続した欠落では結合しえない組の偽陽性が名目の約2倍になり、
60分ではさらに増えました（1組）。欠落位置を推定に渡すと名目の水準に戻り、写し込んだ結合の
検出率はほぼ保たれました。発火の多い2本目の記録では、独立な欠落でも発火なしと読むと結合しえない組の約半数が有意になり、
欠落位置を使うと名目の水準に戻りました。全チャネルが自然に黙るパケットが珍しい記録ほど、歪みは大きくなります。

Earlier files `loss_spikes.py`, `loss_masked_experiment.py`,
`results_loss_spikes.json`, `results_loss_masked.jsonl` are kept for the
record; their significance counts use the z-test of versions up to 3.4.

## 7. Loss and recording length, and a formulation of the transport choice / 欠落と記録長の分離、方式選択の定式化

**Calibration chosen on one part, evaluated on others.** The pooled null
(section 5) was chosen on segments 0/1 of the first recording. On null pairs
that took no part in that choice the rate of p < 0.01 is 1.5% for segments
2/3 of the first recording (400 pairs; binomial P = 0.21 against 1%) and
1.5% for the second recording (1,600 pairs; P = 0.04), i.e. close to
nominal in the first recording and slightly liberal in the second.

**Separating loss from recording length** (`loss_length.py growth`,
`results_length_growth.jsonl`). Nested windows of 15, 30 and 60 min from the
same start times (three replicates), bursty loss 10%, 20 planted pairs and 400
null pairs each:

| null pairs p < 0.05 (nominal 5%) | 15 min | 30 min | 60 min |
| --- | --- | --- | --- |
| no loss | 5.7% | 3.5% | 4.8% |
| bursts, gaps read as silence | 7.7% | 7.4% | **13.9%** |
| bursts, mask-aware | 6.3% | 3.2% | 3.8% |

| planted detected | 15 min | 30 min | 60 min |
| --- | --- | --- | --- |
| no loss | 0.70 | 0.82 | 0.92 |
| bursts, mask-aware | 0.63 | 0.78 | 0.88 |

A sampling error shrinks as the recording gets longer; a distortion does not.
With gaps read as silence the false-positive rate does not fall with length
and at 60 min is higher in all three replicates (p < 0.01: 4.5%, 2.5%, 5.0%
against 1.3%, 1.3%, 0.7% without loss). With the mask it stays near nominal
and detection rises with length, as for a complete recording.

**Mask-aware estimation behaves like a shorter complete recording**
(`loss_length.py equiv`, `results_equivtime.jsonl`). 100 planted pairs per
replicate, five replicates. Complete recordings of 15, 20, 25 and 30 min give
the detection-vs-length curve (0.81, 0.84, 0.89, 0.92); each 30-min
mask-aware estimate under 10% loss is placed on that curve:

| 10% loss, mask-aware, 30 min | f (usable steps) | predicted f × 30 min | measured equivalent length |
| --- | --- | --- | --- |
| independent | 0.819 | 24.6 min | 24.0 min (per replicate 21.0–27.5) |
| bursts | 0.885 | 26.5 min | 27.9 min (per replicate 25.0–30.0) |

The prediction holds to within about 1.5 min, well inside the spread between
replicates.

**Formulation.** Packets of length tau carry all channels; the link loses
packets with rate p (independently or in bursts) and has one-way delay
delta; a process can wait D after a packet was sent. A scheme s (UDP with the
loss mask; XOR parity over k packets; NACK retransmission) leaves a residual
loss R_s(D) and uses b_s times the data bandwidth. The mask-aware estimate
stays calibrated whatever R_s(D) is (section 6), and behaves like a complete
recording of length f(R_s(D)) T (above), where f is the fraction of steps whose
bins t+1, t and t+1-d are all delivered. The recording time to reach a target
detection rate is therefore

    T*(s, D) = T*_0 / f(R_s(D)),     choose  s* = argmin_s T*(s, D)  subject to  b_s rho <= C,

with T*_0 the time for a complete recording, rho the data rate and C the
capacity. Reading gaps as silence is excluded: its error is a bias that does
not shrink with T. `transport_choice.py` (`results_transport_choice.json`)
evaluates f and b_s by simulating the link (tau = 10 ms, delta = 20 ms,
p = 10%, bursts of mean 10 packets, k = 4):

| deadline D | independent loss: mask / parity / retransmission | bursts: mask / parity / retransmission |
| --- | --- | --- |
| 25 ms | 1.22 / 1.22 / 1.22 | 1.11 / 1.12 / 1.11 |
| 50 ms | 1.22 / **1.10** (bw ×1.25) / 1.22 | 1.11 / 1.11 / 1.11 |
| 100 ms | 1.22 / 1.06 (×1.25) / **1.04** (×1.11) | 1.11 / 1.11 / **1.08** (×1.05) |
| 200 ms | 1.22 / 1.06 / **1.00** (×1.11) | 1.11 / 1.11 / **1.03** (×1.08) |
| 1 s | 1.22 / 1.06 / 1.00 (×1.11) | 1.11 / 1.11 / 1.00 (×1.10) |

(entries: T* / T*_0 = 1/f; bandwidth ratio in parentheses). Reading the
table: with a deadline below the time to detect a gap and retransmit (about
50 ms here), parity is the only scheme that helps, and only for independent
loss; from about 100 ms retransmission wins at about 10% extra bandwidth;
bursts cost less recording time than independent loss at the same rate
(1.11 against 1.22) because a TE step needs two or three packets and bursts
put the losses in the same steps. The transport numbers are from a
simulation; the deadline behaviour was also measured on an emulated lossy
link in the teaching material (NeuralDataStreamingLab, E6).

Sequence-number loss notification, NACK feedback and FEC are already
standardized for real-time streams in RTP/RTCP (RFC 3550, RFC 4585, RFC 8627);
the formulation does not replace them but chooses among them by the
reliability of the estimated network.

Limits: one link model and one loss rate, fixed in advance (the extra load
that retransmissions add, which can itself raise the loss rate, is not yet
modelled); T*_0 and the equivalence were
measured on one recording (five replicates); the formulation assumes the
loss does not depend on the values a packet carries (section 6).

検定は一部の区間で選び、使わなかった区間と別の記録で評価しました（1%に対し1.5%）。発火なしと読むと
偽陽性は記録を延ばしても減らず、60分では3組とも増えました（歪み）。欠落位置を使うと名目付近のまま、
検出率は記録とともに上がり、使える時点の割合 f だけ短い完全な記録と同じ検出率になりました
（予測24.6分に対し実測24.0分）。これにより方式の選択を T*(s, D) = T*_0 / f(R_s(D)) の最小化として
定式化しました。

## Across the two systems / 二つの系を通して

The same procedures pick out different structure. In the traffic matrices,
continuity in time within each OD pair is the strongest cue for recovering
hidden entries; in the neuronal firing counts, structure shared across
neurons helps more, at moderate fractions of hidden entries. Which prior helps depends on the system and on how the
observations are missing, and should be checked rather than assumed.

同じ手続きを当てると、通信では各ノード対の時間的な連続性が、神経では細胞間で共有される
構造が、欠けた値の復元に効きました。どの事前の仮定が効くかは、系と観測の欠け方によって変わります。

## Scope and limitations / 範囲と限界

- The recovery experiments hide entries of a KNOWN matrix. They are not
  network tomography: estimating OD flows from link loads (y = A x, with
  the routing matrix A) is not implemented here.
- Transfer entropy is a pairwise, directed statistical dependence. In the
  traffic data it is computed on aggregated hourly node series; conditioning
  is limited to the network mean (section 3).
  It does not identify routes or causal links. For spikes, the CPU run is
  limited to a subset of neurons and time; full recordings are better
  served by the GPU code, with this surrogate/FDR step applied to its output.
- Anomaly candidates are not fault localization, and no labelled faults are
  used.

## Revision history / 改訂

- 2026-11-24 (3.5, later the same day): section 7 -- calibration
  evaluated on parts not used to choose it; loss vs recording length;
  equivalence of mask-aware estimation and a shorter complete recording;
  formulation of the transport choice (`loss_length.py`,
  `transport_choice.py`).
- 2026-11-24 (3.5): calibration of the spike-train TE test on pairs that
  cannot be coupled (`null_check.py`): the per-pair z-test used up to 3.4
  was anti-conservative (49 of 3,200 null pairs FDR-significant); replaced by
  a pooled null (`tmcore.pooled_null_p`), which is calibrated. Spike-train
  significance counts of 3.4 withdrawn. Packet-loss experiment redone
  against a known truth (`loss_groundtruth.py`, section 6), in two
  recordings.

- 2026-11-23 (3.4): mask-aware TE for spike trains under packet loss
  (`te_masked_spikes.py`), compared with zero-fill, and recording length
  (`loss_masked_experiment.py`, section 6).
- 2026-11-23 (3.3): check of what conditioning on the network mean costs
  (`conditioning_check.py`); effect of packet loss on the estimated
  spike-train network (`loss_spikes.py`); signed local TE now treats a rise as the event
  (as a spike in Kajiwara et al., 2021), states its identity, and adds a
  competition model with a one-step response.
- 2026-11-23 (3.2): the TE series are now built on the clock
  (`te_hourly.py`). Up to 3.1 the weekly profile was removed by row index,
  which goes out of phase after the gaps in the data; the GEANT count drops
  from 215 to 9 of 506 pairs. Robustness analysis redone on the corrected
  series, with conditioning on the network mean. Signed local TE and a toy
  check on queue models added (`signed_lte.py`). Recovery results unchanged.
- 2026-11-23 (3.1): first robustness analysis of the TE test.

- 2026-11-19: shared core (`tmcore.py`); spike-train runner
  (`analysis_spikes.py`) for ASDF data; GEANT results unchanged
  (identical `results.json`).
- 2026-11-18: stronger baselines, repeated masks, block hiding,
  validation-based threshold, per-pair surrogates with FDR, hour-of-week
  removal before TE. The first version compared soft-impute only with the
  per-flow mean; its statement that low-rank completion "clearly beats the
  baseline" does not hold against these baselines.
- 2026-11-15: first version.
