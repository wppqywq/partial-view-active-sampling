#!/usr/bin/env bash
# Submit from reference_c; resource flags are supplied on the command line.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
root=/mnt/disk2/youyouyang/proposal2/reference_c
python=/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python
revision=874f12e1ee519860f49860638cf7f6375956d45a
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TORCH_HOME="$root/cache/torch" XDG_CACHE_HOME="$root/cache" MPLCONFIGDIR="$root/cache/matplotlib"
export TMPDIR="$root/tmp" PYTHONPATH="$root/vendor:$root/deps"
mkdir -p "$root/tmp" "$root/runs/$SLURM_JOB_ID"
run="$root/runs/$SLURM_JOB_ID"
cp "$SLURM_SUBMIT_DIR/run.py" "$SLURM_SUBMIT_DIR/run.sh" "$run/"
cd "$run"
sha256sum run.py run.sh > source.sha256
date --iso-8601=seconds
if [[ ${1:-demo} == prepare ]]; then
  if [[ ! -d "$root/vendor/.git" ]]; then
    git clone -q https://github.com/matthias-k/DeepGaze.git "$root/vendor"
    git -C "$root/vendor" checkout -q "$revision"
  fi
  test "$(git -C "$root/vendor" rev-parse HEAD)" = "$revision"
  if [[ ! -d "$root/deps/boltons" ]]; then
    "$python" -m pip install --disable-pip-version-check --no-deps --target "$root/deps" boltons==25.0.0
  fi
  curl -fL --retry 2 --max-time 180 https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy -o "$root/centerbias.npy"
  curl -fL --retry 2 --max-time 180 https://raw.githubusercontent.com/scipy/dataset-face/main/face.dat -o "$root/face.dat.bz2"
  "$python" -c 'from deepgaze_pytorch import DeepGazeIII; m=DeepGazeIII(pretrained=True); print("Official pretrained model loaded on CPU")'
else
  test "$(git -C "$root/vendor" rev-parse HEAD)" = "$revision"
  "$python" run.py --root "$root" --output "$run" "$@"
fi
sha256sum run.py run.sh > source.sha256
"$python" -m pip freeze > environment.txt
date --iso-8601=seconds
