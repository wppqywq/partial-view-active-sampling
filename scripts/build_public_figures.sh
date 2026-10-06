#!/usr/bin/env bash
#SBATCH --job-name=active-sensing-public-figures
#SBATCH --partition=short
#SBATCH --extra=vram:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm on the research server}"
# Slurm copies the batch script; use its submission directory as project root.
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl-$SLURM_JOB_ID"
python_bin=${FV3_PYTHON:-/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python}
"$python_bin" scripts/plot_results.py
"$python_bin" - <<'PY'
import ast
from pathlib import Path
files = sorted(Path('.').rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='ascii'), filename=str(path))
print(f'Python source syntax and ASCII checks passed: {len(files)} files.')
PY
