# 2026-03-27 Validation Scripts And Codebase Docs

## Summary
Updated the lightweight codebase docs and validation scripts to match the current HashINR repository.

## Files changed
- `docs/codebase/overview.md`
- `docs/codebase/data_flow.md`
- `scripts/setup_outputs.sh`
- `scripts/quick_validate.sh`
- `scripts/run_smoke_experiment.sh`

## Main changes
- Replaced the template overview with the actual HashINR package layout and entry points.
- Rewrote `docs/codebase/data_flow.md` to reflect the current data bundle, tensor conventions, training loop, and NUFFT-backed gradient flow.
- Added `scripts/setup_outputs.sh` so the validation commands in `AGENTS.md` now point to a real script.
- Made `scripts/quick_validate.sh` cheap and deterministic:
  - compile Python sources,
  - load the main config,
  - verify dataset metadata with `numpy` memmaps,
  - import the main runtime modules.
- Replaced the template smoke script with a tiny real training run:
  - uses the first few frames from the real dataset,
  - shrinks model capacity,
  - uses reuse-single-operator mode,
  - disables evaluation and heavy logging,
  - writes outputs under `./outputs`.

## Notes
- The smoke script still requires CUDA because the project depends on `tinycudann`.
- The smoke script is intentionally minimal and deterministic, but it does exercise the real trainer, NUFFT path, checkpointing, and reconstruction snapshot path.

## Validation run
- `bash scripts/quick_validate.sh`
  - passed
- `bash scripts/run_smoke_experiment.sh`
  - passed on `cuda:0`
  - wrote outputs under `outputs/subspaceINR4fMRI_smoke`
  - smoke run currently emits the `mri-nufft` sample-rescaling warnings that also appear in the main pipeline
