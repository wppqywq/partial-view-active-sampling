#!/usr/bin/env bash
# Parent schedules this after D integration; no downloads, fitting, or resubmit.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
root=/mnt/disk2/youyouyang/proposal2/reference_c
source_dir=/home/youyouyang/proposal2/reference_c
python=/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache"
export TMPDIR="$root/tmp" PYTHONPATH="$root/vendor:$root/deps"
run="$root/runs/$SLURM_JOB_ID"
mkdir -p "$root/tmp" "$run"
cp "$source_dir/development.py" "$source_dir/development.sh" "$run/"
cp /home/youyouyang/proposal2/d_partial/observer.py "$run/observer.py"
cd "$run"
sha256sum development.py development.sh observer.py > source.sha256
"$python" -m pip freeze > environment.txt
date --iso-8601=seconds
"$python" development.py --root "$root" --output "$run" \
    --manifest /mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json "$@"
date --iso-8601=seconds
