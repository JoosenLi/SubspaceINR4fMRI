# /volatile/home/ql284910/data/HashINR/AGENTS.md

## Project scope
This file applies only to the HashINR project.

## Paths
- Repo root: /volatile/home/ql284910/data/HashINR
- Output symlink: ./outputs -> /neurospin/mind/ql284910/HashINR_outputs
- Environment: /volatile/home/ql284910/data/Env_INR
<!-- - Reference environment baseline: /volatile/home/ql284910/data/Env_INR -->

## Core tools
- Reconstruction: mri-nufft
- Simulation: snake-fmri
- Analysis: nilearn
- When behavior is unclear, inspect installed package source before changing logic.

## Scope restriction
- Only edit files under /volatile/home/ql284910/data/HashINR
- Do not modify sibling repositories under /volatile/home/ql284910/data

## Validation commands
- Setup outputs: bash scripts/setup_outputs.sh
- Quick validation: bash scripts/quick_validate.sh
- Smoke experiment: bash scripts/run_smoke_experiment.sh

## Reporting
- After each non-trivial task, write a report under docs/experiments/YYYY-MM-DD/
- Keep docs/codebase/overview.md and docs/codebase/data_flow.md updated when code structure changes.

## Coding rules
- Prefer explicit tensor shapes.
- Preserve MRI physics assumptions.
- Prefer speed over memory efficiency when the extra memory cost is reasonable and it significantly reduces repeated operator setup cost.