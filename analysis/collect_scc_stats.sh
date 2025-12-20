#!/usr/bin/env bash
set -euo pipefail

dest="$HOME/Desktop/SCC-Stats"
mkdir -p "$dest"

find . -type f -name 'scc_stats.csv' -print0 |
while IFS= read -r -d '' f; do
  rel="${f#./}"
  dir="$(dirname "$rel")"

  if [[ "$dir" == "." ]]; then
    new="scc.csv"
  else
    new="${dir//\//_}_scc.csv"
  fi

  mv -n -- "$f" "$dest/$new"
done

