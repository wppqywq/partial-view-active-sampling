#!/usr/bin/env bash
# Export only reviewed repository content; never touch live experiment storage.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
destination=${1:?Usage: bash scripts/export_public.sh NEW_DIRECTORY}
if [ -e "$destination" ]; then
  echo 'Destination already exists; choose a new directory.' >&2
  exit 1
fi
mkdir -p "$destination"
destination=$(cd "$destination" && pwd)
case "$destination/" in "$root/"*) echo 'Export must be outside the workspace.' >&2; rmdir "$destination"; exit 1;; esac
# A separate temporary Git directory applies this repository's ignore rules,
# without initializing or modifying the protected workspace .git placeholder.
filter_dir=$(mktemp -d /tmp/active-sensing-export-index.XXXXXX)
trap 'rm -rf "$filter_dir"' EXIT
git -C "$filter_dir" init -q
git -c core.excludesFile=/dev/null --git-dir="$filter_dir/.git" --work-tree="$root" \
  ls-files --others --exclude-standard -z -- README.md requirements.txt .gitignore LICENSE docs results scripts support foveated_v3 d_partial data_a reference_c toy |
  tar -C "$root" --null -T - -cf - | tar -C "$destination" -xf -
printf 'Exported source and aggregate results to %s\n' "$destination"
