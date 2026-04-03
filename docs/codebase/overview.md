# Project Overview

## Goal
`HashINR` reconstructs accelerated 3D+time fMRI from undersampled non-Cartesian k-space with the `subspaceINR4fMRI` model:

`I(x,t) = I_init(x) + I_res(x) + I_dyn(x,t)`

where:
- `I_init(x)` is a fixed background initializer loaded from disk,
- `I_res(x)` is an optional learned static residual,
- `I_dyn(x,t)` is a low-rank dynamic term built from spatial coefficients and shared temporal bases,
- an optional delay branch models local timing shifts.

## Top-Level Layout
- `configs/`: experiment, training, benchmark, and search YAML files.
- `data/`: dataset loading, trajectory flattening, time-grid creation, and spatial-grid helpers.
- `evaluation/`: notebook-equivalent fMRI activation scoring and F2 metrics.
- `models/`: shared spatial hash encoder, spatial heads, temporal basis network, and complex tensor helpers.
- `nufft/`: backend factory plus `cufinufft` and `gpunufft` wrappers with differentiable framewise application.
- `training/`: main training loop, losses, optimizer/scheduler helpers, and runtime utilities.
- `scripts/`: runnable CLI and validation entry points.
- `utils/`: config loading, checkpointing, logging, tensor stats, and path helpers.
- `utilis/`: legacy helper code kept for reference and plotting/statistics.
- `CodeArchitecture/`: longer-form architecture notes from the earlier refactor.
- `docs/codebase/`: lightweight day-to-day codebase docs.
- `outputs/`: symlinked large-storage destination for checkpoints, reconstructions, benchmarks, and reports.

## Main Entry Points
- `scripts/quick_validate.sh`: cheap deterministic validation for imports, config, and dataset metadata.
- `scripts/run_smoke_experiment.sh`: tiny end-to-end training run on a small frame subset from the real dataset.
- `scripts/train_subspace_inr_3d_fmri.py`: full training CLI from YAML config.
- `scripts/evaluate_reconstruction_f2.py`: simulated activation-map evaluation against notebook-equivalent metrics.
- `scripts/benchmark_nufft_backends.py`: 2x2 benchmark over backend and operator mode.
- `scripts/profile_training_efficiency.py`: step-time and GPU telemetry profiler.
- `scripts/search_subspace_inr_hparams.py`: staged hyperparameter search launcher.
- `subspaceINR4fMRI.py`: legacy-compatible Python entry point that re-exports the new trainer.

## Core Runtime Objects
- `FMRIDataBundle` in `data/fmri_dataset.py`:
  runtime bundle containing k-space, trajectory, background init, time coordinates, coil count, and spatial shape.
- `SubspaceINR4fMRI` in `models/subspace_inr.py`:
  model owning the shared spatial encoder, coefficient head, optional residual head, optional delay head, and temporal basis network.
- `FramewiseNUFFTBackendBase` in `nufft/nufft_utils.py`:
  backend abstraction that preloads one operator per frame or reuses a single operator with sample updates.
- `_train_impl(...)` in `training/train_subspace_inr.py`:
  orchestration point that wires data, model, NUFFT, losses, checkpointing, and optional evaluation together.

## Validation And Smoke Workflows
- Cheap validation is intentionally CPU-heavy but GPU-light:
  it compiles Python sources, loads config, and checks dataset metadata without training.
- Smoke training is intentionally tiny but real:
  it slices a few frames from the real dataset, shrinks model capacity, runs a couple of epochs, and writes checkpoints/reconstructions under `./outputs`.

## Output Policy
- Large artifacts must be written under `./outputs/`.
- In this environment, `./outputs` is a symlink to large storage.
- Code and docs should treat `./outputs` as the canonical output path.

## Related Docs
- `docs/codebase/data_flow.md`: concise runtime data and tensor flow.
- `CodeArchitecture/`: full architecture reference with file/function map and gradient flow.
- `reports.md`: rolling project report.
