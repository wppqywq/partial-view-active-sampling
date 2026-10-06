#!/usr/bin/env bash
#SBATCH --job-name=foveated-v3-human-choice
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
mean_run=${FV3_MEAN_RUN:?New frozen mean run required}
variance_run=${FV3_VARIANCE_RUN:?New calibrated variance run required}
gain_run=${FV3_GAIN_RUN:?Completed new gain run required}
python=/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python
run="$root/human_choice/runs/$SLURM_JOB_ID"
mkdir -p "$run" "$root/tmp" "$root/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XFORMERS_DISABLED=1
export TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache" MPLCONFIGDIR="$root/cache/matplotlib" TMPDIR="$root/tmp"
export PYTHONPATH=/mnt/disk2/youyouyang/proposal2/predictor_d/vendor/dinov2
cp "$source_dir/human_choice.py" "$source_dir/run_human_choice.sh" "$run/"
cp "$gain_run/train_gain.py" "$gain_run/common.py" "$gain_run/signals.py" "$run/"
cp "$mean_run/train_mean.py" "$mean_run/observer.py" "$mean_run/frozen_mean.py" "$mean_run/frozen_data.py" "$mean_run/frozen_legacy_observer.py" "$mean_run/protocol.json" "$run/"
cd "$run"
sha256sum *.py protocol.json run_human_choice.sh > source.sha256
"$python" -m pip freeze > environment.txt
nvidia-smi > gpu.txt
date --iso-8601=seconds > started.txt
"$python" -u human_choice.py --run-dir "$run" --mean-run "$mean_run" --variance-run "$variance_run" --gain-run "$gain_run" "$@" > human_choice.log 2>&1 &
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
printf 'HUMAN_CHOICE_EXIT_%s\n' "$status" > shell_status.txt
date --iso-8601=seconds > ended.txt
exit "$status"
