#!/usr/bin/env bash
# Laptop-side analysis with the A/A gate in front.
#
#   analyze.sh <results-dir> [arms...]
#
# Refuses to print arm comparisons unless the qualification data passes:
# zero significant rows in benchstat(qual_a, qual_b). The gate's geomean delta
# is the run's measured noise floor; it is printed with every verdict, and any
# effect below it must be read as unresolved.
set -u
D=${1:?usage: analyze.sh <results-dir> [arms...]}
shift || true
cd "$D"

echo "================ QUALIFICATION GATE (A/A) ================"
gate=$(benchstat qual_a.txt qual_b.txt 2>&1)
echo "$gate" | tail -30
sig=$(echo "$gate" | grep -c 'p=0\.0[0-4]' || true)
floor=$(echo "$gate" | awk '/^geomean/{print $NF}')
echo
echo "significant A/A rows: $sig   A/A geomean delta (noise floor): ${floor:-n/a}"
if [ "$sig" -gt 0 ]; then
  echo "GATE FAILED: the run cannot support arm comparisons. Fix the noise, re-run."
  exit 1
fi
echo "GATE PASSED"
echo

if [ $# -eq 0 ]; then set -- $(ls *.txt | grep -Ev '^(qual_|std_|machine)' | sed 's/\.txt$//'); fi
echo "================ MICROBENCH ($*) ================"
files=(); for a in "$@"; do files+=("$a.txt"); done
benchstat "${files[@]}"
echo
echo "================ TIMED std BUILDS (ms) ================"
printf "%-10s %6s %10s %10s %8s\n" arm n mean sd "sd/mean"
for f in std_*.txt; do
  [ -f "$f" ] || continue
  a=${f#std_}; a=${a%.txt}
  awk -v a="$a" '
    $1=="FAIL" {fail++; next}
    {n++; s+=$1; ss+=$1*$1}
    END {
      if (n>0) { m=s/n; sd=(n>1)?sqrt((ss-n*m*m)/(n-1)):0;
        printf "%-10s %6d %10.1f %10.1f %7.2f%%", a, n, m, sd, 100*sd/m }
      if (fail>0) printf "  FAILURES=%d", fail
      print ""
    }' "$f"
done
