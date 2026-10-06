#!/usr/bin/env bash
#SBATCH --job-name=foveated-v3-prepare
#SBATCH --partition=short
#SBATCH --extra=vram:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
fv3_source="${FV3_SOURCE_DIR:-/home/youyouyang/proposal2/foveated_v3}"
fv3_root=/mnt/disk2/youyouyang/proposal2/foveated_v3
fv3_run="$fv3_root/prepare/$SLURM_JOB_ID"
mkdir -p "$fv3_run" "$fv3_root/tmp" "$fv3_root/mpl"
cp "$fv3_source/prepare.py" "$fv3_source/observer.py" "$fv3_source/run_prepare.sh" "$fv3_run/"
cp /home/youyouyang/proposal2/d_partial/observer.py "$fv3_run/old_observer.py"
cp /mnt/disk2/youyouyang/proposal2/foveated_v3/prepare/22738/observer.py "$fv3_run/rejected_observer.py"
if [[ -d "$fv3_source/vendor" ]]; then cp -r "$fv3_source/vendor" "$fv3_run/"; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$fv3_root/mpl" TMPDIR="$fv3_root/tmp"
cd "$fv3_run"
sha256sum ./*.py run_prepare.sh > source.sha256
/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python prepare.py --out "$fv3_run" > prepare.log 2>&1
