#!/usr/bin/env bash
#SBATCH --job-name=fv3-mean-progress
#SBATCH --partition=short
#SBATCH --extra=vram:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/mnt/disk2/youyouyang/proposal2/foveated_v3/mpl
export FV3_PLOT_OUT=/mnt/disk2/youyouyang/proposal2/foveated_v3/progress/$SLURM_JOB_ID
mkdir -p "$FV3_PLOT_OUT"
cp /home/youyouyang/proposal2/foveated_v3/plot_mean_progress.py "$FV3_PLOT_OUT/"
/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python "$FV3_PLOT_OUT/plot_mean_progress.py" > "$FV3_PLOT_OUT/plot.log" 2>&1
