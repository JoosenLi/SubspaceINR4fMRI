# Fix Flat-Loss Training And Produce Final Reconstruction + Benchmarks

## Summary
- The first issue to fix is the non-decreasing data loss. The current log pattern is highly suspicious:
  `loss_data` stays constant at `4.6606e+04`, while `g` collapses to ~0 and `grad_norm` decays to ~`1e-13`.
  That means the optimizer is mostly minimizing the temporal smoothness term while the k-space data term is not effectively driving updates.
- The most likely root cause is the current NUFFT training path in `data/HashINR/nufft/nufft_utils.py`: it calls `operator.op(...)` directly in `forward()` and never wraps the operator in a custom autograd bridge.
  The legacy code in `data/HashINR/utilis/utilis_INR.py:433` had an explicit `NufftAutograd` wrapper for exactly this reason.
- The second issue is throughput. The current implementation is underusing the GPU because it combines:
  `frame_batch_size: 1`,
  Python-side per-frame NUFFT loops,
  and full-grid temporal derivative regularization on all `240` frames every batch.
- The benchmark script also needs to change: it currently varies backend only, but your target comparison is the 2x2 matrix:
  backend `{gpunufft, cufinufft}` x operator mode `{preload_per_frame, reuse_single_operator}`.

## Key Changes
- Restore differentiable k-space data consistency by adding an explicit autograd-safe NUFFT application path in the new backend layer.
  The backend forward used during training must backpropagate through `adj_op(...)`, not rely on raw `operator.op(...)` autograd behavior.
- Add a one-step differentiability sanity test before any long training:
  verify that `loss_data.backward()` produces non-zero gradients on the reconstructed image and on representative model parameters.
  If this test fails, stop and fix the backend before any tuning.
- Keep preload-per-frame operators as the default training mode, but separate operator construction from differentiable application so both preload and reuse modes share the same backward-safe path.
- Remove the current per-batch full-grid temporal smoothness cost as the default.
  Replace it with a bounded temporal regularization batch:
  evaluate `L_g_smooth` on `16` uniformly spaced `t_norm` samples per optimizer step.
  This keeps the regularizer active without burning most of the step on temporal derivative bookkeeping.
- Tune for GPU utilization after the gradient fix by increasing `training.frame_batch_size` empirically on the real GPU.
  Use this exact sweep order: `{1, 2, 4, 8, 12, 16}` and stop at first OOM.
  Choose the largest stable batch size that reduces wall-clock time per effective epoch.
- Drop conservative memory-saving behavior where it does not help:
  keep full spatial-grid evaluation as the default,
  keep chunked spatial evaluation disabled by default,
  and only use chunking as a fallback knob if a chosen batch size actually needs it.
- Retune initialization only if needed after the autograd fix.
  Run a 20-step sanity run first.
  If `loss_data` still drops by less than `1%` and parameter gradients stay tiny, run a small init sweep:
  `temporal_branch.output_init_scale` in `{1e-4, 1e-3, 1e-2}`,
  `residual_head.output_init_scale` in `{1e-4, 1e-3}`,
  `delay_head.output_init_scale` in `{1e-4, 1e-3}`.
  Keep the smallest scales that produce a visible data-loss decrease.
- Redesign the benchmark script to run the full 2x2 comparison:
  `gpunufft + preload`,
  `gpunufft + reuse`,
  `cufinufft + preload`,
  `cufinufft + reuse`.
  Use about `10` repeated iterations after warmup and report mean/std separately for setup, first call, forward, and adjoint.
- Write the final summary report to `data/HashINR/reports.md`.
  That report should include:
  root-cause diagnosis for the flat loss,
  the GPU-utilization diagnosis,
  the benchmark table for the 2x2 NUFFT comparison,
  the final chosen training config,
  and the output paths for the best reconstruction/checkpoint.

## Public Interfaces And Files To Change
- `data/HashINR/nufft/nufft_utils.py`
  Add the differentiable NUFFT application path used by training and benchmarking.
- `data/HashINR/nufft/cufinufft_backend.py`
  Use the shared backward-safe forward/adjoint interface.
- `data/HashINR/nufft/gpunufft_backend.py`
  Use the shared backward-safe forward/adjoint interface.
- `data/HashINR/training/train_subspace_inr.py`
  Switch data-consistency loss to the differentiable NUFFT path, reduce temporal smoothness overhead, and run the batch-size sweep/tuning workflow.
- `data/HashINR/scripts/benchmark_nufft_backends.py`
  Benchmark both backend and operator-definition mode.
- `data/HashINR/configs/subspaceINR4fMRI.yaml`
  Update defaults toward the tuned fast-training configuration.
- `data/HashINR/reports.md`
  Add the final diagnosis + benchmark + chosen-settings report.

## Test Plan
- Differentiability test:
  one random image frame through the training NUFFT path must produce non-zero image gradients after `loss_data.backward()`.
- One-step optimizer test:
  on a tiny subset, one optimizer step must change at least one trainable tensor and reduce `loss_data` on the next forward pass.
- Throughput test:
  measure per-step time split across spatial branch, temporal branch, NUFFT forward, and loss/backward before and after the fix.
- Batch-size sweep:
  run the exact sweep `{1, 2, 4, 8, 12, 16}` and record memory + step time.
- Benchmark matrix:
  run the 2x2 backend/mode comparison for about `10` iterations and save both raw results and the summarized table.
- Final training run:
  use the best-performing stable configuration, save the best reconstruction/checkpoint under `data/HashINR/outputs`, and summarize it in `data/HashINR/reports.md`.

## Assumptions
- The current flat-loss behavior is treated as a blocking bug, not a hyperparameter issue, until the NUFFT data term is confirmed to backpropagate correctly.
- Full spatial-grid evaluation remains the default because your `48 GB` GPU has ample headroom and the current memory footprint is far too low.
- The benchmark target is explicitly the 2x2 comparison of backend and operator-definition mode, not backend-only timing.
- The final deliverables are:
  the tuned reconstruction run,
  the 2x2 NUFFT benchmark,
  the GPU-utilization diagnosis/fix summary,
  and `data/HashINR/reports.md`.
