#!/usr/bin/env bash
#SBATCH --job-name=freeview-data
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
root=/mnt/disk2/youyouyang/proposal2/coco_freeview
mkdir -p "$root/raw" "$root/runs/$SLURM_JOB_ID"
cp "$0" "$root/runs/$SLURM_JOB_ID/run.sh"
date --iso-8601=seconds
if [[ ${1:-fetch} == fetch ]]; then
  base=http://vision.cs.stonybrook.edu/~cvlab_download
  curl -fL --retry 2 --connect-timeout 15 --max-time 240 "$base/COCOFreeView_fixations_trainval.json" -o "$root/raw/fixations.json.part"
  mv "$root/raw/fixations.json.part" "$root/raw/fixations.json"
  curl -fL --retry 1 --connect-timeout 15 --max-time 90 'https://drive.google.com/uc?export=download&id=1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj' -o "$root/raw/readme.txt" || true
  /mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python -c 'import json,sys,collections; d=json.load(open(sys.argv[1])); print(type(d),len(d)); print(d[0]); print({k:dict(collections.Counter(str(r.get(k)) for r in d)) for k in ["split","condition","task"]})' "$root/raw/fixations.json"
  for kind in TP TA; do
    file="$root/raw/COCOSearch18-images-$kind.zip"
    if [[ ! -f $file ]]; then
      curl -fL --retry 2 --connect-timeout 15 --max-time 720 "$base/COCOSearch18-images-$kind.zip" -o "$file.part"
      mv "$file.part" "$file"
    fi
  done
elif [[ $1 == metadata ]]; then
  curl -fL --retry 2 --connect-timeout 15 --max-time 600 http://images.cocodataset.org/annotations/annotations_trainval2014.zip -o "$root/raw/annotations_trainval2014.zip.part"
  mv "$root/raw/annotations_trainval2014.zip.part" "$root/raw/annotations_trainval2014.zip"
else
  cp "$SLURM_SUBMIT_DIR/audit.py" "$root/runs/$SLURM_JOB_ID/audit.py"
  /mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python "$root/runs/$SLURM_JOB_ID/audit.py" "$root" "$SLURM_SUBMIT_DIR"
fi
date --iso-8601=seconds
