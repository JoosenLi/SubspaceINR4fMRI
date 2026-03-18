# HashINR March 17 Status Report

## Scope
All code changes in this iteration were kept strictly under `data/HashINR`.

This pass focused on three linked goals:
- improving the real training hot path,
- automating notebook-equivalent activation evaluation,
- building the two-GPU staged search machinery.

## What Was Implemented

### 1. Hot-path training changes
- `data/HashINR/models/temporal_basis.py`
  - replaced the old per-output autograd loop for `dg/dt_norm` with an analytic derivative for the current Fourier + sine-network temporal branch
  - validated against the autograd version with max absolute derivative error about `1.72e-05`
- `data/HashINR/training/train_subspace_inr.py`
  - added explicit device selection through `compute.device` and `compute.gpu_index`
  - made search-time defaults lean:
    - `training.gradient_sanity_check: false`
    - `training.startup_diagnostics: false`
    - `training.profile_first_steps: 0`
    - `logging.tensorboard: false`
    - `logging.save_reconstruction_snapshot: false`
  - reused the already-computed batch temporal derivative for temporal smoothness when `losses.temporal_basis_smooth.reuse_batch_derivative: true`
  - integrated evaluation hooks and run-level `metrics.json`
- `data/HashINR/nufft/nufft_utils.py`
  - kept the differentiable HashINR-side NUFFT bridge
  - trimmed some Python overhead in the framewise operator loop
  - preserved contiguous inputs for `cufinufft`

### 2. Notebook-equivalent evaluation pipeline
Added:
- `data/HashINR/evaluation/__init__.py`
- `data/HashINR/evaluation/activation_metrics.py`
- `data/HashINR/scripts/evaluate_reconstruction_f2.py`

This evaluation path now:
- loads `dataset/sim3DT/Simulation_Mar16_RotatedSpiral/dataloader.mrd`
- rebuilds the activation context from the simulated acquisition
- computes notebook-style z-score activation statistics
- reports:
  - `F2_fixed` at threshold `3.3`
  - `F2_max`
  - `best_thresh`
  - ROI time-series correlation
  - ROI TSNR
- saves:
  - `metrics.json`
  - `activation_scores.json`

The evaluation module now writes compact `metrics.json` files and keeps the full threshold sweep in `activation_scores.json`.

### 3. Automation and search tooling
Added:
- `data/HashINR/scripts/profile_training_efficiency.py`
- `data/HashINR/scripts/search_subspace_inr_hparams.py`
- `data/HashINR/configs/search_subspaceINR4fMRI.yaml`

The search launcher supports:
- process-level parallelism over GPU0 and GPU1
- staged rounds `round0` through `round5`
- run resumption
- per-round CSV and JSON summaries
- global `search_results.csv` and `search_results.json`

### 4. Config updates
Updated:
- `data/HashINR/configs/subspaceINR4fMRI.yaml`
- `data/HashINR/configs/benchmark_nufft.yaml`
- `data/HashINR/configs/search_subspaceINR4fMRI.yaml`

Notable additions:
- `compute`
- `evaluation`
- `search`

## Validation That Was Actually Run

### Analytic temporal derivative check
Validated the new analytic derivative against the previous autograd derivative on GPU.

Result:
- basis max abs diff: `0.0`
- derivative max abs diff: `1.7166e-05`

Conclusion:
- the analytic derivative is numerically consistent with the old autograd path while being much cheaper.

### Baseline activation evaluation
The notebook-equivalent evaluation was run on the simulated baselines.

#### `Global_CG.npy`
Output:
- `data/HashINR/outputs/baseline_eval_Global_CG/metrics.json`
- `data/HashINR/outputs/baseline_eval_Global_CG/activation_scores.json`

Metrics:
- `F2_fixed = 0.686396`
- `F2_max = 0.730561`
- `best_thresh = 2.733021`
- `roi_signal_corr = 0.975023`
- `roi_tsnr = 60.8697`

#### `NUFFT_recon.npy`
Output:
- `data/HashINR/outputs/baseline_eval_NUFFT_recon/metrics.json`
- `data/HashINR/outputs/baseline_eval_NUFFT_recon/activation_scores.json`

Metrics:
- `F2_fixed = 0.249571`
- `F2_max = 0.421814`
- `best_thresh = 2.366943`
- `roi_signal_corr = 0.987124`
- `roi_tsnr = 22.4632`

### Efficiency profiling
#### Direct profiler run
Output:
- `data/HashINR/outputs/efficiency_profile_march17.json`

Measured on `cuda:0`:
- `step_ms = 110.886`
- `spatial_ms = 4.808`
- `temporal_ms = 15.776`
- `nufft_ms = 8.338`
- `backward_ms = 80.359`
- `cpu_overhead_ms = 1.604`
- `power_draw_w_mean = 131.94`
- `power_draw_w_max = 192.0`
- `utilization_gpu_pct_mean = 67.57`
- `utilization_gpu_pct_max = 99.0`

#### Round-0 profiler run
Output:
- `data/HashINR/outputs/search_subspaceINR4fMRI/round0/efficiency/profile.json`

Measured on `cuda:0`:
- `step_ms = 97.234`
- `spatial_ms = 3.466`
- `temporal_ms = 10.850`
- `nufft_ms = 6.833`
- `backward_ms = 74.939`
- `cpu_overhead_ms = 1.146`
- `power_draw_w_mean = 147.71`
- `power_draw_w_max = 211.78`
- `utilization_gpu_pct_mean = 72.59`
- `utilization_gpu_pct_max = 99.0`

Interpretation:
- the remaining low power is **not** mainly caused by logging or Python I/O checks
- CPU-side overhead is only about `1.1-1.6 ms/step`
- the dominant cost is still the GPU-side backward pass, with framewise time-varying NUFFT structure remaining a real limit

## 2x2 NUFFT Benchmark
Output:
- `data/HashINR/outputs/benchmark_nufft_Simulation_Mar16_RotatedSpiral/benchmark/benchmark_results.json`

Measured with:
- frame batch size `16`
- warmup `3`
- repetitions `10`

| Backend | Mode | Setup s | Forward mean s | Adjoint mean s |
| --- | --- | ---: | ---: | ---: |
| `cufinufft` | `preload_per_frame` | `10.2281` | `0.00348` | `0.05838` |
| `cufinufft` | `reuse_single_operator` | `0.04210` | `0.64154` | `0.62981` |
| `gpunufft` | `preload_per_frame` | `48.5697` | `0.08323` | `0.03187` |
| `gpunufft` | `reuse_single_operator` | `0.19409` | `3.61495` | `3.58549` |

Takeaways:
- preload-per-frame is decisively better than reuse-single for this time-varying trajectory dataset
- `cufinufft` is much faster on forward
- `gpunufft` is faster on adjoint
- for the full training path, `cufinufft` remains the better default backend

## Search Automation Status

### Implemented
The staged search launcher now supports:
- `round0`: instrumentation, benchmark, baselines
- `round1`: backend and memory-max system sweep
- `round2`: scale / rank / delay / residual sweep
- `round3`: capacity sweep
- `round4`: optimizer and lambda tuning
- `round5`: long finalists with seed reruns

### Validated so far
`round0` was executed successfully end-to-end.

Outputs:
- `data/HashINR/outputs/search_subspaceINR4fMRI/round0/round0_summary.json`
- `data/HashINR/outputs/search_subspaceINR4fMRI/round0/baselines/baseline_metrics.json`
- `data/HashINR/outputs/search_subspaceINR4fMRI/round0/efficiency/profile.json`

What `round0` completed:
- efficiency profile
- full 2x2 NUFFT benchmark
- baseline F2 evaluation for `Global_CG.npy` and `NUFFT_recon.npy`

### What has not been fully run yet
The long hyperparameter rounds `round1` through `round5` have been implemented, but they were **not** fully executed in this session.

That means:
- the search infrastructure is ready
- the baselines and instrumentation are validated
- the final best-hyperparameter answer still requires actually launching the overnight sweep

## Smoke Training Validation
Ran a one-epoch training smoke test with integrated evaluation after the latest changes.

Run:
- `data/HashINR/outputs/smoke_eval_final_march17`

Observed:
- training completed successfully
- integrated F2 evaluation ran and wrote outputs under the run directory

This was a pipeline validation run, not a quality run.

## Current Best Interpretation Of GPU Efficiency
The current GPU power gap is partly structural.

What the profiler shows:
- Python / CPU overhead is small
- the dominant time is in backward
- the workload still requires per-frame time-varying NUFFT application and adjoint

So the current underutilization is **not** mainly caused by:
- TensorBoard writes
- JSON writes
- startup diagnostics
- anomaly detection

Those were already disabled for the lean path.

What remains:
- framewise NUFFT orchestration
- sequential operator use because trajectories vary over time
- a lightweight model relative to the NUFFT-heavy training step

## Recommended Next Command
To start the full staged search:

```bash
python data/HashINR/scripts/search_subspace_inr_hparams.py \
  --config data/HashINR/configs/search_subspaceINR4fMRI.yaml \
  --round all
```

If you want to begin from the system sweep only:

```bash
python data/HashINR/scripts/search_subspace_inr_hparams.py \
  --config data/HashINR/configs/search_subspaceINR4fMRI.yaml \
  --round round1
```

## Key Output Paths
- Main config: `data/HashINR/configs/subspaceINR4fMRI.yaml`
- Search config: `data/HashINR/configs/search_subspaceINR4fMRI.yaml`
- Efficiency profile: `data/HashINR/outputs/efficiency_profile_march17.json`
- Benchmark results: `data/HashINR/outputs/benchmark_nufft_Simulation_Mar16_RotatedSpiral/benchmark/benchmark_results.json`
- Round-0 summary: `data/HashINR/outputs/search_subspaceINR4fMRI/round0/round0_summary.json`
- Baseline metrics: `data/HashINR/outputs/search_subspaceINR4fMRI/round0/baselines/baseline_metrics.json`
