#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/check_ignore.sh
for file in scripts/*.sh foveated_v3/*.sh data_a/*.sh reference_c/*.sh; do bash -n "$file"; done
if rg -n --glob '*.md' --glob '*.py' --glob '*.sh' --glob '*.json' --glob '*.txt' '[^\x00-\x7F]' README.md docs results scripts support foveated_v3 d_partial data_a reference_c requirements.txt; then
  echo 'Non-ASCII text found.' >&2; exit 1
fi
if rg -n -e '-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}' -- README.md docs results scripts support foveated_v3 d_partial data_a reference_c requirements.txt; then
  echo 'Possible credential found; inspect before export.' >&2; exit 1
fi
if find docs results scripts support foveated_v3 d_partial data_a reference_c -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.ipynb' -o -name '*.pptx' -o -name '*.zip' \) -print | rg .; then
  echo 'Unexpected data/model/draft artifact found.' >&2; exit 1
fi
(cd results && sha256sum -c SHA256SUMS)
echo 'Release content, ASCII, shell syntax, and aggregate integrity checks passed.'
