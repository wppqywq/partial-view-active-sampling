#!/usr/bin/env bash
#SBATCH --job-name=active-sensing-data
#SBATCH --partition=short
#SBATCH --extra=vram:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
# Opt-in download into NEW external storage. Never run against a live dataset.
set -euo pipefail
: "${SLURM_JOB_ID:?Run data preparation in a Slurm allocation}"
: "${DATA_ROOT:?Set DATA_ROOT to a new external dataset directory}"
: "${PROJECT_CODE:?Set PROJECT_CODE to the cloned repository root}"
python_bin=${PROJECT_PYTHON:-python}
code=$(cd "$PROJECT_CODE" && pwd)
mkdir -p "$DATA_ROOT"
data_root=$(cd "$DATA_ROOT" && pwd)
case "$data_root/" in "$code/"*) echo 'DATA_ROOT must be outside the source repository.' >&2; exit 1;; esac
if [ -e "$data_root/manifest.json" ]; then
  echo 'Existing prepared dataset detected; refusing to overwrite it.' >&2; exit 1
fi
raw="$data_root/raw"
run="$data_root/runs/$SLURM_JOB_ID"
mkdir -p "$raw" "$run"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$run/mpl"
fetch() {
  local url=$1 destination=$2
  if [ ! -f "$destination" ]; then
    curl -fL --retry 2 --connect-timeout 20 --max-time 1200 "$url" -o "$destination.part"
    mv "$destination.part" "$destination"
  fi
}
base=https://vision.cs.stonybrook.edu/~cvlab_download
fetch "$base/COCOFreeView_fixations_trainval.json" "$raw/fixations.json"
fetch "$base/COCOSearch18-images-TP.zip" "$raw/COCOSearch18-images-TP.zip"
fetch "$base/COCOSearch18-images-TA.zip" "$raw/COCOSearch18-images-TA.zip"
fetch 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip' "$raw/annotations_trainval2014.zip"
fetch 'https://drive.google.com/uc?export=download&id=1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj' "$raw/readme.txt"
if ! rg -q '1680x1050' "$raw/readme.txt"; then
  echo 'Download the actual FreeView readme.txt from its official page; an HTML login page is not valid data.' >&2
  exit 1
fi
(cd "$raw" && sha256sum -c "$code/data_a/RAW_SHA256SUMS")
cp "$code/data_a/audit.py" "$run/audit.py"
cp "$code/scripts/prepare_dataset.sh" "$run/run.sh"
"$python_bin" "$run/audit.py" "$data_root" "$run" > "$run/audit.log" 2>&1
printf 'Dataset manifest: %s/manifest.json\nAudit output: %s\n' "$data_root" "$run"
