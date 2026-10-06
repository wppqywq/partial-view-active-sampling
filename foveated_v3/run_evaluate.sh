#!/usr/bin/env bash
#SBATCH --job-name=foveated-v3-paths
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --extra=vram:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
fv3_root=/mnt/disk2/youyouyang/proposal2/foveated_v3
fv3_source=${FV3_SOURCE_DIR:?Frozen source directory required}
fv3_mean=${FV3_MEAN_RUN:?New frozen mean run required}
fv3_var=${FV3_VARIANCE_RUN:?New accepted variance run required}
fv3_gain=${FV3_GAIN_RUN:?New completed gain run required}
fv3_run="$fv3_root/evaluation/runs/$SLURM_JOB_ID"
mkdir -p "$fv3_run" "$fv3_root/tmp" "$fv3_root/cache"
cp "$fv3_source/evaluate_paths.py" "$fv3_source/run_evaluate.sh" "$fv3_run/"
cp "$fv3_gain/train_gain.py" "$fv3_gain/common.py" "$fv3_gain/signals.py" "$fv3_run/"
for file in train_mean.py observer.py frozen_mean.py frozen_data.py frozen_legacy_observer.py; do
  cp "$fv3_mean/$file" "$fv3_run/$file"
done
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XFORMERS_DISABLED=1
export PYTHONPATH=/mnt/disk2/youyouyang/proposal2/predictor_d/vendor/dinov2
export MPLCONFIGDIR="$fv3_root/cache/matplotlib" TMPDIR="$fv3_root/tmp"
cd "$fv3_run"
sha256sum ./*.py run_evaluate.sh > source.sha256
nvidia-smi > gpu.txt
/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python -u evaluate_paths.py --mean-run "$fv3_mean" --variance-run "$fv3_var" --gain-run "$fv3_gain" --run-dir "$fv3_run" --records-root "$fv3_root/evaluation/records" > evaluate.log 2>&1 &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi
set -e
printf 'EVALUATE_EXIT_%s\n' "$status" > shell_status.txt
exit "$status"
