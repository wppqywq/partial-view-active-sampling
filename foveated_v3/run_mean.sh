#!/usr/bin/env bash
#SBATCH --job-name=foveated-v3-mean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --extra=vram:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
root=/mnt/disk2/youyouyang/proposal2/foveated_v3
source_dir=${FV3_SOURCE_DIR:?Frozen source directory required}
budget_json=${FV3_BUDGET_JSON:?Preparation budget artifact required}
visual_review_json=${FV3_VISUAL_REVIEW_JSON:?Reviewed observation artifact required}
python=/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python
run="$root/mean/runs/$SLURM_JOB_ID"
mkdir -p "$run" "$root/tmp" "$root/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XFORMERS_DISABLED=1
export TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache" MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
export PYTHONPATH=/mnt/disk2/youyouyang/proposal2/predictor_d/vendor/dinov2
cp "$source_dir/train_mean.py" "$source_dir/observer.py" "$source_dir/run_mean.sh" "$run/"
cp "$budget_json" "$run/protocol.json"
cp "$budget_json" "$run/budget.json"
cp "$visual_review_json" "$run/visual_review.json"
cp /mnt/disk2/youyouyang/proposal2/direct_mean_v2/runs/22427/train.py "$run/frozen_mean.py"
cp /mnt/disk2/youyouyang/proposal2/predictor_d/runs/22351/train.py "$run/frozen_data.py"
cp /mnt/disk2/youyouyang/proposal2/predictor_d/runs/22351/observer.py "$run/frozen_legacy_observer.py"
cd "$run"
sha256sum *.py protocol.json budget.json visual_review.json run_mean.sh > source.sha256
"$python" -m pip freeze > environment.txt
nvidia-smi > gpu.txt
date --iso-8601=seconds > started.txt
"$python" -u train_mean.py --run-dir "$run" --budget-json "$run/budget.json" --visual-review-json "$run/visual_review.json" "$@" > train.log 2>&1 &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
  wait "$child"
  status=$?
fi
set -e
printf 'TRAIN_EXIT_%s\n' "$status" > shell_status.txt
date --iso-8601=seconds > ended.txt
exit "$status"
