**Team 38 project brief — 15 November 2026**

The project has completed baseline reproduction and its first generalization study. Post-midterm work adds evidence that wind, rather than collection date alone, drives a measurable generalization gap, and implements a physics–TCN hybrid. The hybrid's comparative results and the proposed uncertainty/CVaR contribution remain unverified or unfinished in the available repository.

This brief draws on the proposal and supplied midterm PDF in `/home/ashutosh/CMU/IDL_Project`, the local code and experiment artifacts, and live GitHub comparisons of all eight branches. It summarizes recorded results; training and evaluation were not rerun. Local `main` matches GitHub at `f90ce26` (23 October). The newest visible work is on `ysorapal/physics-hybrid`, at `d5b6e77` (9 November).

**Research question and how it has evolved.** “Beyond the Training Grid” asks whether drone power models generalize to unfamiliar payloads, speeds, altitudes, routes and wind, whether physics improves those predictions, and whether calibrated uncertainty improves preflight energy-risk estimates. The original expectation was that heavier payloads and higher speeds would expose failures. The midterm found that the TCN already generalizes well across this dataset's operating grid. The strongest remaining questions concern wind, limited training data, and uncertainty. The proposed small transformer was explicitly deferred in the midterm.

**What is established.** The team ported the original TensorFlow TCN/LSTM models to PyTorch and converted all four pretrained checkpoints, with reported output differences below 4e-7. Power MAPE reproduces the paper closely; energy RE has small documented discrepancies. On the paper's test set, retraining over three seeds gives s-TCN **8.86 ± 0.26% MAPE** and b-TCN **8.92 ± 0.16%**, versus published pretrained values of 9.13% and 9.06%. The seed-0 s-LSTM reproduces its baseline at 12.42%, but recurrent training is unreliable across runs. [Main README](https://github.com/wouyang26/idl_project_38/blob/main/README.md)

The study uses 187 available preprocessed flights, with 179 main-route and eight off-route flights under KiltHub route labels. The infrastructure includes held-out conditions, training-size controls, saved split/normalization metadata, and comparison on flights neither model trained or validated on. The midterm reports 78 TCN runs across three seeds. For s-TCN, payload/speed/altitude gaps are at most about 0.4 MAPE percentage points; route and wind are harder. Commanded speed is a weak extrapolation test because wind creates overlap in measured airspeed and other flight phases dominate average power. These findings apply to this dataset, not unrestricted operating conditions. [Midterm source](https://github.com/wouyang26/idl_project_38/blob/main/report/midterm/main.tex)

**Progress beyond the midterm.** Experiment C now has a committed results CSV. Across three seeds, the wind gap is +1.05 points for s-TCN and +0.99 for b-TCN. On eight windy flights from other, familiar dates, gaps remain +0.81/+0.61. On 11 calm flights from held-out windy dates, the date-control gaps are only +0.13/+0.09. This supports wind as the principal explanation in these comparisons, although the samples are small and the wind estimates are noisy. These are rerun results and should be distinguished from the midterm's +0.77/+1.22 wind gaps. [Experiment C results](https://github.com/wouyang26/idl_project_38/blob/ysorapal/physics-hybrid/docs/stress_test/wind_vs_day.csv)

Experiment A1 reports that refitting Liu's five constants using each split's training flights improves power MAPE by about 1.3–2.9 points. On route, refit Liu reaches 10.88%, essentially matching s-TCN's 10.89%; on wind it reaches 12.61%, versus 13.94% with the released constants. This revises the midterm's assumption that the released constants necessarily advantaged Liu. The result table is in the experiment document; its generated raw CSV is not committed. [Experiment A specification and A1 results](https://github.com/wouyang26/idl_project_38/blob/ysorapal/physics-hybrid/docs/experiments/A_physics_hybrid.md)

The hybrid is implemented as `P̂ = P_Liu + TCN correction`, with Liu refit on training flights and a zero-initialized correction output. Training, checkpoint loading, stress-test support, and learning-curve scripts exist. Planned comparisons cover wind/route and training subsets of 5, 10, 20, 35, 52 and 87 flights, plus the full 114. I found no committed A2 hybrid comparison or A3 learning-curve results, nor corresponding artifacts in this local `Results/` directory. Implementation is therefore ahead of demonstrated performance.

**Branch status verified on GitHub.**

| Branch | Relationship to main | Significance |
|---|---|---|
| `main` | Current checkout, `f90ce26` | Consolidated midterm milestone |
| `ashutosh/slstm-experiments` | Identical to main | Your shared branch has no additional commits |
| `ysorapal/physics-hybrid` | 3 commits ahead | Hybrid implementation, A1 table, completed C results |
| `ysorapal/wind-date-control` | 1 commit ahead | C setup and initial A plan; included in physics-hybrid |
| `ysorapal/blstm-sweep` | 2 ahead / 4 behind | Additional seed-0 b-LSTM logs/results; not a training-stability fix |
| `adesai/wind` | 4 behind | Earlier wind work already incorporated into main |
| `ysorapal/heldout-splits` | 22 behind | Earlier split implementation incorporated into main |
| `ysorapal/baseline-setup` | 24 behind | Earlier baseline work incorporated into main |

**Your work and outstanding work.** The midterm credits you with s-LSTM experiments, stress-test runs and evaluation fixes. Your local ignored `Results/` contains seven s-LSTM run configurations, logs, checkpoints and a packaged archive. These support the completed midterm contribution; they do not establish that the collapse issue is fixed. The report diagnoses constant predictions in 16 of 30 LSTM runs, and the inspected branches retain the ReLU head and existing optimizer settings.

I found no implementation/results for a Gaussian variance head, calibrated ensemble intervals or MC-dropout evaluation. Multiple seed runs and ordinary dropout support exist, but uncertainty calibration still needs to be built. The inherited `sim.py`/`Risk.py` CVaR pipeline exists; simulation still loads the original TensorFlow model and does not incorporate the proposed PyTorch model uncertainty.

**Recommended order of work.**

1. Review and integrate the physics-hybrid branch, preserving the distinction between midterm and post-midterm results.
2. Complete or recover A2/A3 outputs, comparing the hybrid with both plain TCN and refit Liu. Report a negative result if physics does not help.
3. Build uncertainty evaluation: 90% interval coverage and interval width on random and held-out flights, followed by propagation into flight-energy/CVaR estimates.
4. Repair LSTM training before extending its sweep; keep the transformer deferred unless the core deliverables are complete.

For the final report, keep the route-control limitation, small test sets, noisy wind estimates, and unstable LSTMs explicit. Also preserve metric conventions: the paper's test list has 37 entries but 35 unique flights, training-log MAPE is on normalized power, and current preprocessing derives normalization from training **and validation** flights. The latter differs from the proposal's training-only wording and matters when interpreting limited-data learning curves.
