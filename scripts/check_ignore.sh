#!/usr/bin/env bash
# Test actual Git ignore behavior without touching the workspace Git metadata.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
probe=$(mktemp -d /tmp/active-sensing-ignore-check.XXXXXX)
trap 'rm -rf "$probe"' EXIT
git -C "$probe" init -q
cp "$root/.gitignore" "$probe/.gitignore"
for path in .env .env.local id_rsa envs/official/bin/python raw/fixations.json datasets/image.jpg data/image.jpg coco_freeview/manifest.json outputs/features.npy partial_features/image.pt checkpoints/model.pth temporary.npz trial_records.jsonl download.zip download.zip.part; do
  if ! git -C "$probe" -c core.excludesFile=/dev/null check-ignore -q --no-index "$path"; then
    printf 'Expected ignored path is publishable: %s\n' "$path" >&2; exit 1
  fi
done
for path in README.md .env.example requirements.txt data_a/audit.py data_a/RAW_SHA256SUMS scripts/prepare_dataset.sh results/mean_metrics.json; do
  if git -C "$probe" -c core.excludesFile=/dev/null check-ignore -q --no-index "$path"; then
    printf 'Required public file is ignored: %s\n' "$path" >&2; exit 1
  fi
done
echo 'Git ignore checks passed for private/runtime paths and required public artifacts.'
