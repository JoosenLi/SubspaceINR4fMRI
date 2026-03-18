# Runtime Data Flow

This file explains how data moves through the code from the moment a training command starts until a checkpoint, reconstruction, or evaluation artifact is produced.

## 1. Configuration Loading

### CLI entry
Typical training starts here:
- `data/HashINR/scripts/train_subspace_inr_3d_fmri.py`

The runtime sequence is:
1. parse CLI args
2. load YAML with `utils.load_yaml_config(...)`
3. apply optional dotted overrides with `utils.apply_dotlist_overrides(...)`
4. call `training.train_subspace_inr_3d_fmri(config)`

## 2. Building The Data Bundle

File:
- `data/HashINR/data/fmri_dataset.py`

`train_subspace_inr_3d_fmri(config)` calls:
- `load_fmri_data_bundle(config, PROJECT_ROOT)`

This function:
1. resolves file paths from config
2. loads:
   - `kspace.npy`
   - `traj.npy`
   - `averagedBG.npy`
   - optional `csm.npy`
3. normalizes shape conventions with `flatten_kspace_and_traj(...)`
4. scales:
   - k-space by `data.scale_factor`
   - background image by `data.scale_factor`
5. creates normalized time coordinates:
   - `t_norm = frame_index / (n_frames - 1)`
6. returns an `FMRIDataBundle`

At this point, the bundle holds the fixed training inputs:
- sampled k-space data
- non-Cartesian trajectories
- fixed background initialization
- optional coil maps
- normalized time coordinates

## 3. Training Initialization

File:
- `data/HashINR/training/train_subspace_inr.py`

`_train_impl(config, bundle)` performs:
1. device resolution via `resolve_compute_device(...)`
2. output directory creation via `prepare_run_directories(...)`
3. random seeding
4. creation of:
   - spatial voxel grid `spatial_coords`
   - flattened fixed background `init_image_flat`
   - full time coordinate tensor `full_time_coords`
5. optional preload of k-space to GPU
6. model construction:
   - `SubspaceINR4fMRI(config)`
7. optimizer and scheduler construction
8. NUFFT backend construction:
   - `create_nufft_backend(...)`

## 4. Model Construction

File:
- `data/HashINR/models/subspace_inr.py`

`SubspaceINR4fMRI(config)` builds:

### Shared spatial path
- `TCNNHashGridEncoder`

### Spatial heads
- optional `residual_head`
- required `coefficient_head`
- optional `delay_head`

### Temporal path
- `TemporalBasisNetwork`

### Fixed constants
- `rank`
- `tau_norm_max = tau_max_seconds / total_duration_seconds`

## 5. One Training Step

For each frame batch in the epoch:

### Step 5.1. Select frame indices
`_make_epoch_frame_batches(...)` returns frame index batches.

For a batch:
- `frame_indices_cpu`
- `frame_indices_device`

### Step 5.2. Gather target k-space
If `preload_kspace_to_device=true`:
- target k-space is gathered by indexing the preloaded GPU tensor

Otherwise:
- the batch is moved from CPU to GPU on demand

### Step 5.3. Gather batch time coordinates
`t_norm_batch = full_time_coords.index_select(0, frame_indices_device)`

This yields a `[B,1]` normalized-time tensor.

## 6. Forward Pass Inside The Model

### 6.1 Spatial forward
Call:
- `model.evaluate_spatial_components(spatial_coords)`

Internal flow:
1. `encode_spatial(...)`
2. `TCNNHashGridEncoder.forward(coords)`
3. optional residual head on shared features
4. coefficient head on shared features
5. optional delay head on shared features

Output:
- `SpatialComponents`
  - `residual: [V,2]` or `None`
  - `coeff: [V,M]`
  - `tau: [V,1]` or `None`

Important:
- this is evaluated on the full spatial grid by default
- `V = X * Y * Z`

### 6.2 Temporal forward
Call:
- `model.evaluate_temporal_basis(t_norm_batch, need_derivative=model.use_delay)`

Internal flow:
1. `FourierFeatureEncoding.forward_with_derivative(...)`
2. sine-network hidden layers
3. output linear layer
4. optional analytic derivative propagation

Output:
- `basis: [B,M,2]`
- `basis_derivative: [B,M,2]` or `None`

### 6.3 Image synthesis
Call:
- `model.synthesize_batch(...)`

Internal formula without delay:
- `dynamic = einsum(coeff, basis)`

With delay:
- `delayed_dynamic = einsum(coeff, basis_derivative)`
- `dynamic = dynamic - tau * delayed_dynamic`

Then:
- `static = I_init`
- if enabled, `static += I_res`
- `prediction = static + dynamic`

Returned shape:
- `[B, X, Y, Z, 2]`

### 6.4 Convert to complex
Call:
- `real_imag_to_complex(prediction_ri)`

Returned shape:
- `[B, X, Y, Z]` complex tensor

## 7. Forward Through The Measurement Model

Call:
- `nufft_backend.forward(prediction_complex, frame_indices_cpu)`

File:
- `data/HashINR/nufft/nufft_utils.py`

If gradients are needed:
- `_FramewiseNUFFTAutograd.apply(...)` is used

Forward internals:
1. get per-frame operator
2. call backend `op(...)`
3. normalize output shape
4. stack k-space predictions into `[B, C, S]`

This yields predicted k-space at the sampled coordinates for that frame batch.

## 8. Loss Computation

### 8.1 Data consistency
Call:
- `complex_mse(kspace_prediction, kspace_target)`

This is the main reconstruction supervision.

### 8.2 Residual energy
If residual is enabled:
- `residual_energy_loss(spatial_components.residual)`

### 8.3 Delay smoothness
If delay is enabled:
- `tau_smoothness_loss(spatial_components.tau, spatial_shape)`

### 8.4 Temporal basis smoothness
If enabled:
- `temporal_basis_smoothness_loss(temporal_derivative)`

The derivative is either:
- reused from the batch derivative path, or
- recomputed on a separate time sample set

### 8.5 Total loss
`total_loss = data + lambda_res * residual + lambda_tau * tau + lambda_g * temporal`

The lambdas are warmed up per epoch via `_loss_weights(...)`.

## 9. Backward Pass

Call:
- `total_loss.backward()`
  or `scaler.scale(total_loss).backward()` when AMP is enabled

The key path is:

```text
loss
  -> k-space prediction
  -> NUFFT autograd backward
  -> image-domain gradient via adjoint NUFFT
  -> synthesized image
  -> spatial components and temporal bases
  -> model parameters
```

Inside `_FramewiseNUFFTAutograd.backward(...)`:
1. receive gradient w.r.t. predicted k-space
2. call backend `adj_op(...)` frame by frame
3. stack image-domain gradients
4. return image gradient to PyTorch autograd

This makes the NUFFT data-consistency loss differentiable with respect to the model outputs.

## 10. Optimizer Step

After gradients are populated:
1. optional grad clipping
2. gradient norm logging
3. optimizer step
4. optional scheduler step per epoch

Updated parameter groups can include:
- spatial encoder parameters
- coefficient head parameters
- optional residual head parameters
- optional delay head parameters
- temporal basis network parameters

## 11. Evaluation Flow

If evaluation is enabled:
1. `_reconstruct_full_sequence(...)`
2. convert complex reconstruction to magnitude
3. `evaluate_magnitude_reconstruction(...)`

Evaluation uses:
- `dataloader.mrd`
- phantom/ROI
- activation waveform
- `cal_z_score(...)`
- `get_f2_analysis(...)`

Outputs:
- `metrics.json`
- `activation_scores.json`
- optional arrays

## 12. Search Flow

File:
- `data/HashINR/scripts/search_subspace_inr_hparams.py`

Round flow:
1. `round0`
   - profile
   - benchmark
   - baseline evaluation
2. `round1`
   - backend / batch / LR sweep
3. `round2`
   - scale / rank / delay / residual sweep
4. `round3`
   - capacity sweep
5. `round4`
   - LR and regularization tuning
6. `round5`
   - longer finalists and seed reruns

Each trial runs as an independent training subprocess on one GPU.
