# Data Flow

## Raw Inputs On Disk
- `kspace.npy`: non-Cartesian k-space samples, usually stored as `[T, C, shots, samples]` or `[T, C, S]`.
- `traj.npy`: sample locations, usually stored as `[T, shots, samples, 3]` or `[T, S, 3]`.
- `averagedBG.npy`: fixed complex background initializer, stored as `[X, Y, Z, 2]` or complex-valued `[X, Y, Z]`.
- optional `csm.npy`: coil sensitivity maps, stored as `[C, X, Y, Z, 2]` or complex-valued `[C, X, Y, Z]`.
- optional `dataloader.mrd`: simulation metadata used by activation-map evaluation.

## Loaded Tensor Conventions
The core loader is `data/fmri_dataset.py`.

After `flatten_kspace_and_traj(...)` and `build_data_bundle_from_arrays(...)`:
- k-space: `torch.complex64`, shape `[T, C, S]`
- trajectory: `numpy.float32`, shape `[T, S, 3]`
- background init: `torch.float32`, shape `[X, Y, Z, 2]`
- time coordinates: `torch.float32`, shape `[T, 1]`
- spatial coords: `torch.float32`, shape `[V, 3]`, where `V = X * Y * Z`

Important conventions:
- complex numbers use real/imag in the last dimension before conversion to complex tensors.
- time uses normalized coordinates:
  `t_norm = frame_index / (n_frames - 1)`.
- `data.scale_factor` multiplies both k-space and the fixed background so they stay in the same scale convention.

## End-To-End Training Flow
1. `scripts/train_subspace_inr_3d_fmri.py` loads YAML and optional dotted overrides.
2. `training.train_subspace_inr_3d_fmri(...)` loads `FMRIDataBundle` from disk.
3. `training._train_impl(...)` resolves the device, seeds RNGs, creates output directories, and builds the model and NUFFT backend.
4. For each frame batch, the trainer:
   - selects frame indices,
   - gathers target k-space `[B, C, S]`,
   - gathers `t_norm_batch` `[B, 1]`,
   - evaluates spatial components on the full voxel grid,
   - evaluates temporal bases on the current batch times,
   - synthesizes predicted images `[B, X, Y, Z, 2]`,
   - converts them to complex `[B, X, Y, Z]`,
   - applies framewise NUFFT to get predicted k-space `[B, C, S]`,
   - computes losses,
   - backpropagates through the NUFFT adjoint,
   - updates model parameters.
5. Periodically, the trainer saves checkpoints, reconstruction snapshots, and optional evaluation artifacts.

## Model-Side Tensor Flow
The model implementation lives in `models/subspace_inr.py`.

### Spatial branch
Input:
- normalized spatial coordinates `[V, 3]`

Output of `evaluate_spatial_components(...)`:
- `coeff`: `[V, M]`
- optional `residual`: `[V, 2]`
- optional `tau`: `[V, 1]`

The spatial encoder is shared across all heads.

### Temporal branch
Input:
- normalized time `[B, 1]`

Output of `evaluate_temporal_basis(...)`:
- `basis`: `[B, M, 2]`
- optional `basis_derivative`: `[B, M, 2]`

The derivative is `dg / dt_norm`, not derivative with respect to seconds.

### Synthesis
`models/subspace_inr.py` forms the prediction as:
- no delay:
  `dynamic = einsum(coeff, basis)`
- with delay:
  `dynamic = einsum(coeff, basis) - tau * einsum(coeff, basis_derivative)`

Then:
- `static = I_init`
- optional `static += I_res`
- `prediction = static + dynamic`

Final image batch shape:
- `[B, X, Y, Z, 2]`

## Loss And Gradient Flow
The main losses live in `training/losses.py`.

Per batch, the trainer computes:
- `L_data`: complex MSE between predicted and measured k-space.
- optional `L_res`: residual energy penalty.
- optional `L_tau`: spatial smoothness of the delay field.
- optional `L_g`: temporal basis smoothness.

Backprop path:
- `L_data` -> predicted k-space -> differentiable NUFFT wrapper -> predicted complex image -> synthesized batch -> spatial/temporal model parameters.
- `L_res` -> residual head and shared spatial encoder.
- `L_tau` -> delay head and shared spatial encoder.
- `L_g` -> temporal basis network.

The NUFFT backward is handled in `nufft/nufft_utils.py` with a HashINR-side autograd bridge:
- forward uses backend `op(...)`
- backward uses backend `adj_op(...)`

## What Actually Controls Memory
Two config knobs are easy to confuse:
- `training.frame_batch_size`:
  the actual number of reconstructed frames per optimizer step. This changes image-batch, k-space-batch, and gradient memory.
- `losses.temporal_basis_smooth.num_time_samples`:
  only the number of extra temporal samples used for temporal smoothness when a separate derivative pass is needed.

With the current default:
- `model.use_delay: true`
- `losses.temporal_basis_smooth.reuse_batch_derivative: true`

`num_time_samples` is largely inactive, because the temporal smoothness loss reuses the derivative already computed for the current frame batch.

## NUFFT Backend Modes
The backend factory in `nufft/backend_factory.py` supports:
- preload mode:
  one operator per frame, faster runtime, higher persistent GPU memory.
- reuse mode:
  one operator updated with per-frame sampling locations, lower memory, slower runtime.

Because trajectories vary across frames, training remains framewise even when multiple frames are grouped in one optimizer batch.

## Evaluation Flow
The evaluation module in `evaluation/activation_metrics.py` mirrors the notebook-style simulated activation scoring pipeline.

Input:
- reconstructed magnitude sequence `[T, X, Y, Z]`
- simulation metadata from `dataloader.mrd`

Output:
- `F2_fixed` at threshold `3.3`
- `F2_max`
- best threshold
- ROI signal correlation
- ROI TSNR
- optional saved activation arrays

## Validation Scripts
- `scripts/quick_validate.sh`:
  deterministic metadata/import/config validation with no training.
- `scripts/run_smoke_experiment.sh`:
  tiny end-to-end run on a small frame subset from the real dataset using the real trainer.
