# HashINR Code Architecture

This folder documents the current `HashINR` package as implemented in `data/HashINR`.

Reading order:
1. `README.md`
2. `01_System_Overview.md`
3. `02_File_and_Function_Map.md`
4. `03_Runtime_Data_Flow.md`
5. `04_Gradient_and_Parameter_Flow.md`

## What This Documents
- the file/module structure of `HashINR`
- the role of each important file
- the main classes and functions inside those files
- the runtime path from config loading to reconstruction and evaluation
- the training path from forward model to loss to backward to parameter updates

## Package Intent
`HashINR` is a modular PyTorch + tiny-cuda-nn + NUFFT codebase for accelerated 3D+time fMRI reconstruction with the `subspaceINR4fMRI` model:

`I(x,t) = I_init(x) + I_res(x) + I_dyn(x,t)`

where:
- `I_init(x)` is the fixed precomputed background
- `I_res(x)` is an optional learned residual static correction
- `I_dyn(x,t)` is a low-rank dynamic term with shared temporal bases
- an optional delay branch modifies the dynamic basis through a first-order Taylor approximation

## Top-Level Runtime Entry Points
- Training CLI: `data/HashINR/scripts/train_subspace_inr_3d_fmri.py`
- Legacy-compatible wrapper: `data/HashINR/subspaceINR4fMRI.py`
- Evaluation CLI: `data/HashINR/scripts/evaluate_reconstruction_f2.py`
- Efficiency profiler: `data/HashINR/scripts/profile_training_efficiency.py`
- NUFFT benchmark: `data/HashINR/scripts/benchmark_nufft_backends.py`
- Hyperparameter search launcher: `data/HashINR/scripts/search_subspace_inr_hparams.py`

## Main Runtime Layers
- `configs/`
  - experiment, model, loss, backend, evaluation, and search settings
- `data/`
  - loading and shape normalization of k-space, trajectory, background, and time grids
- `models/`
  - spatial encoder, heads, temporal basis network, complex tensor helpers, and the full `SubspaceINR4fMRI` model
- `nufft/`
  - backend factory and framewise differentiable NUFFT wrapper
- `training/`
  - training loop, losses, optimizer/scheduler utilities, checkpoint/log setup
- `evaluation/`
  - notebook-equivalent simulated activation evaluation and F2 metrics
- `scripts/`
  - runnable orchestration entry points
- `utils/`
  - generic support helpers
- `utilis/`
  - legacy utilities kept mainly for reference and plotting/statistics helpers

## Core Design Facts
- Spatial representation is shared:
  - one hash-grid encoder feeds residual, coefficient, and optional delay heads
- Time uses normalized coordinates:
  - `t_norm = frame_index / (n_frames - 1)`
- Delay is structural:
  - when `model.use_delay=false`, no delay head is instantiated and no temporal derivative is used in synthesis
- Residual is structural:
  - when `model.use_residual=false`, the residual head is not instantiated and `I_res` is omitted
- NUFFT is framewise because trajectories vary across time
- Backprop through k-space consistency uses a HashINR-side autograd bridge:
  - forward uses backend `op(...)`
  - backward uses backend `adj_op(...)`
