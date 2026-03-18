# System Overview

## 1. Architectural Layers

```text
Config/YAML
  -> Data loading and normalization
  -> Model construction
  -> NUFFT backend construction
  -> Training loop
      -> spatial forward
      -> temporal forward
      -> image synthesis
      -> NUFFT forward
      -> losses
      -> backward through NUFFT adjoint
      -> optimizer step
  -> optional evaluation
  -> checkpointing / logs / search summaries
```

## 2. High-Level Package Layout

```text
data/HashINR/
├── configs/
├── data/
├── evaluation/
├── models/
├── nufft/
├── scripts/
├── training/
├── utils/
├── utilis/
├── subspaceINR4fMRI.py
└── reports.md
```

## 3. Main Reconstruction Algorithm

The model reconstructs a complex 4D image sequence with real/imag stored in the last dimension.

The learned reconstruction at each voxel and frame is:

`I_hat(x,t) = I_init(x) + I_res(x) + sum_m c_m(x) g_m(t_norm)`

or, with delay enabled:

`I_hat(x,t) = I_init(x) + I_res(x) + sum_m c_m(x) [g_m(t_norm) - tau_norm(x) * dg_m/dt_norm]`

Important implementation conventions:
- `I_init(x)`:
  - fixed, loaded from `averagedBG.npy`
  - scaled together with k-space by `data.scale_factor`
- `I_res(x)`:
  - optional complex residual branch
- `c_m(x)`:
  - real-valued spatial coefficients
- `g_m(t_norm)`:
  - complex-valued temporal bases
- `tau_norm(x)`:
  - optional normalized local delay with `tanh` bound

## 4. Main Objects

### Data object
`data/HashINR/data/fmri_dataset.py`
- `FMRIDataBundle`

This is the central runtime bundle passed into training. It holds:
- `kspace`
- `traj`
- `background_init`
- `csm`
- `time_coords`
- `spatial_shape`
- `n_frames`
- `num_coils`
- `scale_factor`

### Model object
`data/HashINR/models/subspace_inr.py`
- `SubspaceINR4fMRI`
- `SpatialComponents`

This object owns:
- the shared spatial encoder
- the coefficient head
- the optional residual head
- the optional delay head
- the temporal basis network

### NUFFT object
`data/HashINR/nufft/nufft_utils.py`
- `FramewiseNUFFTBackendBase`

This object owns:
- one operator per frame in preload mode, or
- one reusable operator with `update_samples(...)` in reuse mode

### Training object
`data/HashINR/training/train_subspace_inr.py`
- `_train_impl(...)`

This function orchestrates:
- model construction
- per-epoch frame batching
- forward pass
- loss computation
- backpropagation
- checkpointing
- optional evaluation

## 5. End-to-End Call Chain

For normal training:

```text
scripts/train_subspace_inr_3d_fmri.py
  -> utils.load_yaml_config(...)
  -> training.train_subspace_inr_3d_fmri(config)
    -> data.load_fmri_data_bundle(...)
    -> training._train_impl(config, bundle)
      -> models.SubspaceINR4fMRI(config)
      -> nufft.create_nufft_backend(...)
      -> epoch loop
```

For array-driven use from another Python program or notebook:

```text
subspaceINR4fMRI.py: dynamic_Recon3D(...)
  -> training.train_subspace_inr_from_arrays(...)
  -> training._train_impl(...)
```

## 6. Why The Code Is Split This Way

- `data/` separates dataset shape normalization from training logic
- `models/` isolates the INR parameterization from NUFFT and optimization
- `nufft/` isolates backend differences and autograd handling
- `training/` contains the optimization policy
- `evaluation/` keeps notebook-equivalent analysis independent from training
- `scripts/` keep runnable workflows thin and explicit
