#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/..:${PYTHONPATH:-}"

cd "$ROOT_DIR"

bash scripts/setup_outputs.sh >/dev/null

SMOKE_DEVICE="${HASHINR_SMOKE_DEVICE:-cuda:0}"
SMOKE_FRAMES="${HASHINR_SMOKE_FRAMES:-4}"
SMOKE_EPOCHS="${HASHINR_SMOKE_EPOCHS:-2}"
SMOKE_BATCH="${HASHINR_SMOKE_BATCH:-2}"

python - <<'PY'
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from HashINR.training import train_subspace_inr_from_arrays
from HashINR.utils import load_yaml_config, resolve_path

project_root = Path.cwd()
config = load_yaml_config(project_root / 'configs' / 'subspaceINR4fMRI.yaml')

smoke_device = os.environ.get('HASHINR_SMOKE_DEVICE', 'cuda:0')
smoke_frames = int(os.environ.get('HASHINR_SMOKE_FRAMES', '4'))
smoke_epochs = int(os.environ.get('HASHINR_SMOKE_EPOCHS', '2'))
smoke_batch = int(os.environ.get('HASHINR_SMOKE_BATCH', '2'))

if not torch.cuda.is_available():
    raise RuntimeError('Smoke experiment requires CUDA because HashINR uses tinycudann.')
if not smoke_device.startswith('cuda'):
    raise RuntimeError(f'Expected a CUDA smoke device, got {smoke_device!r}.')

gpu_index = int(smoke_device.split(':', 1)[1]) if ':' in smoke_device else 0

data_cfg = config['data']
background_path = resolve_path(data_cfg['background_init_path'], project_root)
kspace_path = resolve_path(data_cfg['kspace_path'], project_root)
traj_path = resolve_path(data_cfg['traj_path'], project_root)

kspace = np.load(kspace_path)
traj = np.load(traj_path)
background = np.load(background_path)

full_kspace = kspace
kspace = full_kspace[:smoke_frames]
if traj.ndim == 4 and full_kspace.ndim == 4 and traj.shape[0] != kspace.shape[0]:
    shots_per_frame = int(full_kspace.shape[2])
    expected_shot_major_frames = int(full_kspace.shape[0]) * shots_per_frame
    if traj.shape[0] == expected_shot_major_frames:
        traj = traj[: smoke_frames * shots_per_frame]
    else:
        traj = traj[:smoke_frames]
else:
    traj = traj[:smoke_frames]

runtime_config = deepcopy(config)
runtime_config['experiment']['run_name'] = 'subspaceINR4fMRI_smoke'
runtime_config['experiment']['case_name'] = f'Smoke{smoke_frames}Frames'
runtime_config['experiment']['output_root'] = './outputs'
runtime_config.setdefault('compute', {})
runtime_config['compute']['device'] = 'cuda'
runtime_config['compute']['gpu_index'] = gpu_index
runtime_config['data']['img_size'][0] = smoke_frames
runtime_config['time']['n_frames'] = smoke_frames
runtime_config['time']['total_duration_seconds'] = float(smoke_frames) * float(runtime_config['time']['tr_seconds'])
runtime_config['data']['scale_factor'] = float(config['data'].get('scale_factor', 1.0))

runtime_config['model']['rank'] = 4
runtime_config['model']['use_residual'] = False
runtime_config['model']['use_delay'] = False
runtime_config['model']['chunked_spatial_eval'] = False
runtime_config['coefficient_head']['output_dim'] = 4
runtime_config['temporal_branch']['output_dim'] = 8
runtime_config['spatial_encoder']['num_levels'] = 8
runtime_config['spatial_encoder']['level_dim'] = 2
runtime_config['spatial_encoder']['log2_hashmap_size'] = 15
runtime_config['coefficient_head']['hidden_dim'] = 16
runtime_config['coefficient_head']['num_hidden_layers'] = 1
runtime_config['temporal_branch']['fourier_num_frequencies'] = 8
runtime_config['temporal_branch']['hidden_dim'] = 16
runtime_config['temporal_branch']['num_hidden_layers'] = 1

runtime_config['nufft']['backend'] = 'cufinufft'
runtime_config['nufft']['preload_per_frame_operators'] = False
runtime_config['nufft']['reuse_single_operator'] = True

runtime_config['training']['num_epochs'] = smoke_epochs
runtime_config['training']['frame_batch_size'] = min(smoke_batch, smoke_frames)
runtime_config['training']['shuffle_frames_each_epoch'] = False
runtime_config['training']['preload_kspace_to_device'] = False
runtime_config['training']['gradient_sanity_check'] = False
runtime_config['training']['startup_diagnostics'] = False
runtime_config['training']['profile_first_steps'] = 0

runtime_config['losses']['residual_energy']['enabled'] = False
runtime_config['losses']['tau_smooth']['enabled'] = False
runtime_config['losses']['temporal_basis_smooth']['enabled'] = False
runtime_config['warmup']['residual_lambda_warmup_epochs'] = 0
runtime_config['warmup']['tau_lambda_warmup_epochs'] = 0
runtime_config['warmup']['temporal_smooth_lambda_warmup_epochs'] = 0

runtime_config['logging']['tensorboard'] = False
runtime_config['logging']['log_interval'] = 1
runtime_config['logging']['save_interval'] = 1
runtime_config['logging']['save_reconstruction_snapshot'] = True
runtime_config['logging']['save_reconstruction_every'] = 1
runtime_config['logging']['log_peak_memory'] = False
runtime_config['checkpointing']['keep_last_k'] = 1
runtime_config['evaluation']['enabled'] = False
runtime_config['benchmark']['enabled'] = False

summary = train_subspace_inr_from_arrays(
    kspaces=kspace,
    trajs=traj,
    back_img=background,
    config=runtime_config,
    case_name=f'Smoke{smoke_frames}Frames',
)

summary_path = Path(summary['run_dir']) / 'smoke_summary.json'
with summary_path.open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2)

print('Smoke experiment finished.')
print(json.dumps({
    'run_dir': summary['run_dir'],
    'checkpoint_dir': summary['checkpoint_dir'],
    'recon_dir': summary['recon_dir'],
    'summary_path': str(summary_path),
    'device': summary['device'],
    'final_metrics': summary['final_metrics'],
}, indent=2))
PY
