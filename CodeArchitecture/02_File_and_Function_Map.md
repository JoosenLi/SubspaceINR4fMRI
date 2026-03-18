# File And Function Map

This file is a guided map of the important files and the main functions or classes inside them.

## Package Root

### `data/HashINR/__init__.py`
- `SubspaceINR4fMRI`
  - package-level model export
- `train_subspace_inr_3d_fmri`
  - package-level training export
- `train_subspace_inr_from_arrays`
  - package-level in-memory array training export

### `data/HashINR/subspaceINR4fMRI.py`
- `dynamic_Recon3D(...)`
  - legacy-compatible wrapper around `train_subspace_inr_from_arrays(...)`
- `parse_args()`
  - CLI parser for config/device/override
- `main()`
  - training CLI entry

## Configs

### `data/HashINR/configs/subspaceINR4fMRI.yaml`
Main experiment config for:
- data paths
- temporal convention
- model structure
- NUFFT backend
- losses
- optimizer/scheduler
- evaluation

### `data/HashINR/configs/benchmark_nufft.yaml`
Config for backend and operator-mode benchmarking.

### `data/HashINR/configs/search_subspaceINR4fMRI.yaml`
Config for multi-round hyperparameter search orchestration.

### `data/HashINR/configs/legacy_dynamic_real3d.yaml`
Legacy comparison or reference config kept out of the new main path.

## Data Layer

### `data/HashINR/data/__init__.py`
Exports data bundle, loaders, grid helpers, and batch utilities.

### `data/HashINR/data/fmri_dataset.py`
- `FMRIDataBundle`
  - runtime container for loaded tensors/arrays
- `FrameKspaceDataset`
  - optional frame dataset wrapper
- `_complex_to_real_imag_last_dim(...)`
  - converts complex arrays to real/imag last-dimension format
- `_to_complex_numpy(...)`
  - converts real/imag arrays to `complex64`
- `flatten_kspace_and_traj(...)`
  - normalizes shot-major or frame-major storage into frame-major flattened sample layout
- `build_data_bundle_from_arrays(...)`
  - builds `FMRIDataBundle` directly from arrays
- `load_fmri_data_bundle(...)`
  - loads arrays from config paths and builds the bundle

### `data/HashINR/data/grid_utils.py`
- `create_normalized_spatial_grid(image_shape)`
  - builds `[V,3]` voxel coordinates in `[0,1]^3`
- `create_normalized_time_grid(n_frames)`
  - builds `[T,1]` normalized time coordinates
- `spatial_shape_from_config(img_size)`
  - converts `[T,X,Y,Z]` config shape to spatial `(X,Y,Z)`

### `data/HashINR/data/batch_sampler.py`
- `create_frame_dataloader(...)`
  - generic PyTorch dataloader helper

## Model Layer

### `data/HashINR/models/__init__.py`
Exports:
- `SubspaceINR4fMRI`
- `SpatialComponents`
- complex helpers
- spatial encoder
- temporal basis

### `data/HashINR/models/complex_ops.py`
- `real_imag_to_complex(...)`
  - `[... ,2]` to complex tensor
- `complex_to_real_imag(...)`
  - complex tensor to `[... ,2]`
- `complex_mse(...)`
  - complex-valued MSE used for data consistency
- `complex_energy(...)`
  - complex magnitude-squared energy for residual penalty

### `data/HashINR/models/spatial_encoder_tcnn.py`
- `TCNNHashGridEncoder`
  - tiny-cuda-nn hash-grid encoder for 3D spatial coordinates

### `data/HashINR/models/heads.py`
- `TCNNHead`
  - tiny-cuda-nn MLP head used for:
    - residual output
    - coefficient output
    - delay output

### `data/HashINR/models/temporal_basis.py`
- `FourierFeatureEncoding`
  - normalized-time Fourier input encoding
  - also returns analytic encoding derivative
- `SineLayer`
  - SIREN-style hidden layer
  - also supports analytic derivative propagation
- `TemporalBasisNetwork`
  - maps `t_norm` to complex temporal bases
  - optionally returns analytic `dg/dt_norm`
  - also contains `forward_with_autograd_derivative(...)` for validation

### `data/HashINR/models/subspace_inr.py`
- `SpatialComponents`
  - container for:
    - `residual`
    - `coeff`
    - `tau`
- `SubspaceINR4fMRI`
  - full model
- `SubspaceINR4fMRI.__init__`
  - validates rank/output dimensions
  - builds encoder, heads, and temporal branch
- `encode_spatial(...)`
  - passes voxel coordinates through the shared hash encoder
- `evaluate_spatial_components(...)`
  - returns residual, coefficients, and optional delay
- `evaluate_temporal_basis(...)`
  - returns temporal bases and optional derivatives
- `synthesize_batch(...)`
  - combines fixed background, spatial outputs, and temporal bases into predicted frames

## NUFFT Layer

### `data/HashINR/nufft/__init__.py`
Exports the backend factory.

### `data/HashINR/nufft/backend_factory.py`
- `create_nufft_backend(...)`
  - selects `cufinufft` or `gpunufft`
  - validates single-coil vs multi-coil rules

### `data/HashINR/nufft/nufft_utils.py`
- `_synchronize_if_needed(...)`
  - timing helper
- `cuda_timed_call(...)`
  - GPU-safe timed function execution
- `normalize_kspace_frame(...)`
  - keeps NUFFT forward outputs shape-consistent
- `normalize_image_frame(...)`
  - keeps NUFFT adjoint outputs shape-consistent
- `to_numpy_samples(...)`
  - forces trajectory dtype/layout for backend construction
- `frame_indices_to_list(...)`
  - lightweight frame index conversion
- `_FramewiseNUFFTAutograd`
  - custom autograd bridge:
    - `forward`: backend `op(...)`
    - `backward`: backend `adj_op(...)`
- `FramewiseNUFFTBackendBase`
  - common preload/reuse operator logic
  - `forward_direct(...)`
  - `forward(...)`
  - `adjoint_direct(...)`
  - `describe(...)`

### `data/HashINR/nufft/cufinufft_backend.py`
- `CuFINUFFTBackend`
  - concrete backend subclass using `mrinufft.get_operator("cufinufft")`

### `data/HashINR/nufft/gpunufft_backend.py`
- `GPUNUFFTBackend`
  - concrete backend subclass using `mrinufft.get_operator("gpunufft")`

## Training Layer

### `data/HashINR/training/__init__.py`
Exports:
- `train_subspace_inr_3d_fmri`
- `train_subspace_inr_from_arrays`

### `data/HashINR/training/losses.py`
- `residual_energy_loss(...)`
  - L2-like energy on `I_res`
- `tau_smoothness_loss(...)`
  - finite-difference smoothness on the delay field
- `temporal_basis_smoothness_loss(...)`
  - derivative energy on temporal bases

### `data/HashINR/training/schedulers.py`
- `build_scheduler(...)`
  - currently supports cosine annealing or no scheduler

### `data/HashINR/training/trainer_utils.py`
- `set_random_seed(...)`
  - reproducibility
- `resolve_compute_device(...)`
  - `cpu`/`cuda:N` selection
- `prepare_nufft_config(...)`
  - injects GPU id into backend kwargs
- `build_optimizer(...)`
  - optimizer builder
- `warmup_lambda(...)`
  - warmup schedule for loss weights
- `prepare_run_directories(...)`
  - creates run/checkpoint/log/reconstruction/evaluation directories
- `maybe_log_histograms(...)`
  - optional TensorBoard histogram logging

### `data/HashINR/training/train_subspace_inr.py`
Key helpers:
- `_loss_weights(...)`
  - computes epoch-specific effective loss weights
- `_collect_startup_diagnostics(...)`
  - optional magnitude diagnostics at initialization
- `_make_epoch_frame_batches(...)`
  - frame shuffling and batching
- `_select_temporal_regularization_coords(...)`
  - subsampled time coordinates for temporal regularization
- `_run_gradient_sanity_check(...)`
  - verifies that the k-space loss backpropagates non-zero gradients
- `_reconstruct_full_sequence(...)`
  - full-sequence synthesis for evaluation or snapshot saving
- `_save_reconstruction_snapshot(...)`
  - snapshot save helper
- `_train_impl(...)`
  - main training loop
- `train_subspace_inr_3d_fmri(...)`
  - file-based training entrypoint
- `train_subspace_inr_from_arrays(...)`
  - array-based training entrypoint

### `data/HashINR/training/legacy_regularizers.py`
Legacy helper regularizers kept separate from the new main path.

## Evaluation Layer

### `data/HashINR/evaluation/__init__.py`
Exports reconstruction and evaluation helpers.

### `data/HashINR/evaluation/activation_metrics.py`
- `SimulationEvaluationContext`
  - holds simulated activation metadata
- `load_simulation_context(...)`
  - loads `dataloader.mrd`, phantom/ROI, and activation waveform context
- `reconstruction_to_magnitude(...)`
  - converts stored reconstruction to magnitude volume
- `compute_roi_timeseries(...)`
  - ROI timeseries and TSNR
- `compute_expected_signals(...)`
  - reference simulated activation waveform
- `evaluate_magnitude_reconstruction(...)`
  - notebook-equivalent z-score/F2 evaluation
- `reconstruct_checkpoint(...)`
  - reconstructs a saved model checkpoint into a 4D image
- `evaluate_reconstruction_path(...)`
  - evaluates a saved `.npy` reconstruction
- `evaluate_checkpoint(...)`
  - reconstructs then evaluates a saved checkpoint

## Script Layer

### `data/HashINR/scripts/train_subspace_inr_3d_fmri.py`
- `parse_args()`
- `main()`
Thin CLI around `train_subspace_inr_3d_fmri(...)`.

### `data/HashINR/scripts/evaluate_reconstruction_f2.py`
- `parse_args()`
- `main()`
CLI wrapper for notebook-equivalent evaluation.

### `data/HashINR/scripts/profile_training_efficiency.py`
- `GPUSampler`
  - polls `nvidia-smi` while a short training run is executing
- `parse_args()`
- `main()`

### `data/HashINR/scripts/benchmark_nufft_backends.py`
- `benchmark_backend_mode(...)`
  - benchmarks one backend/mode pair
- `main()`
  - runs the full 2x2 benchmark matrix

### `data/HashINR/scripts/search_subspace_inr_hparams.py`
- `Trial`
  - search trial specification
- `_build_round1_trials(...)` through `_build_round5_trials(...)`
  - round-specific trial generation
- `_run_round0(...)`
  - instrumentation/baseline stage
- `_run_trials(...)`
  - multi-GPU subprocess launcher
- `_save_round_summary(...)`
  - round-level CSV/JSON writer
- `main()`
  - orchestrates the staged search

### `data/HashINR/scripts/inspect_config.py`
Small config inspection helper.

## Utility Layer

### `data/HashINR/utils/legacy_compat.py`
- `deep_update(...)`
- `load_yaml_config(...)`
- `resolve_path(...)`
- `parse_override_value(...)`
- `apply_dotlist_overrides(...)`

### `data/HashINR/utils/checkpoint_utils.py`
- `save_checkpoint(...)`

### `data/HashINR/utils/logging_utils.py`
- `NullSummaryWriter`
- `create_summary_writer(...)`

### `data/HashINR/utils/memory_utils.py`
- `reset_peak_memory_stats(...)`
- `get_peak_memory_stats(...)`

### `data/HashINR/utils/tensor_utils.py`
- `safe_item(...)`
- `compute_grad_norm(...)`
- `assert_finite(...)`
- `detach_cpu(...)`
- `chunked_forward(...)`
- `tensor_stats(...)`

## Legacy Support

### `data/HashINR/utilis/utilis_plot.py`
Still used for:
- `cal_z_score(...)`
- `cluster_threshold_3d(...)`
- `get_f2_analysis(...)`

### `data/HashINR/utilis/utilis_INR.py`
Legacy DD-INR/NeRP-style utilities retained mainly as reference.
