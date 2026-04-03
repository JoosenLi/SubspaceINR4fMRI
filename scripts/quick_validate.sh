#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/..:${PYTHONPATH:-}"

cd "$ROOT_DIR"

bash scripts/setup_outputs.sh >/dev/null

echo "[1/4] Compile Python sources"
mapfile -t PY_FILES < <(find . \
  -path './outputs' -prune -o \
  -path './__pycache__' -prune -o \
  -path './.git' -prune -o \
  -name '*.py' -print | sort)
python -m py_compile "${PY_FILES[@]}"

echo "[2/4] Load and inspect main config"
python scripts/inspect_config.py --config configs/subspaceINR4fMRI.yaml >/dev/null

echo "[3/4] Check dataset metadata and paths"
python - <<'PY'
from pathlib import Path
import numpy as np
import yaml

project_root = Path.cwd()
with (project_root / 'configs' / 'subspaceINR4fMRI.yaml').open('r', encoding='utf-8') as handle:
    config = yaml.safe_load(handle)
data_cfg = config['data']
time_cfg = config['time']
eval_cfg = config.get('evaluation', {})

def resolve_path(path_value):
    if path_value in (None, '', 'null'):
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path

kspace_path = resolve_path(data_cfg['kspace_path'])
traj_path = resolve_path(data_cfg['traj_path'])
bg_path = resolve_path(data_cfg['background_init_path'])
mrd_path = resolve_path(eval_cfg.get('dataloader_mrd_path'))

for path in [kspace_path, traj_path, bg_path]:
    if path is None or not path.exists():
        raise FileNotFoundError(path)

kspace = np.load(kspace_path, mmap_mode='r')
traj = np.load(traj_path, mmap_mode='r')
bg = np.load(bg_path, mmap_mode='r')

assert int(kspace.shape[0]) == int(time_cfg['n_frames']), (kspace.shape, time_cfg['n_frames'])
assert int(kspace.shape[0]) == int(data_cfg['img_size'][0]), (kspace.shape, data_cfg['img_size'])
assert bg.shape[:3] == tuple(data_cfg['img_size'][1:]), (bg.shape, data_cfg['img_size'])
assert traj.shape[-1] == 3, traj.shape
if mrd_path is not None and not mrd_path.exists():
    raise FileNotFoundError(mrd_path)

print({
    'kspace_shape': tuple(int(v) for v in kspace.shape),
    'traj_shape': tuple(int(v) for v in traj.shape),
    'background_shape': tuple(int(v) for v in bg.shape),
    'mrd_exists': bool(mrd_path is not None and mrd_path.exists()),
})
PY

echo "[4/4] Validate shell entry points"
bash -n scripts/setup_outputs.sh
bash -n scripts/quick_validate.sh
bash -n scripts/run_smoke_experiment.sh
test -x scripts/setup_outputs.sh
test -x scripts/quick_validate.sh
test -x scripts/run_smoke_experiment.sh

echo "Quick validation passed"
