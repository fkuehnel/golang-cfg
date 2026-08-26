#!/usr/bin/env bash
# Noise-qualified sweep. Runs on the VM, detached, no contact until sentinel.
#
#   sweep.sh <manifest>
#
# Manifest: one "name=dir-under-$HOME" line per arm; the FIRST arm is the
# reference and supplies the A/A qualification data. Emits per-arm microbench
# files, per-arm std-build wall-times, qual_a/qual_b for the gate, machine.txt
# provenance, and a "SWEEP COMPLETE" sentinel. Analysis happens laptop-side
# (analyze.sh), which refuses arm comparisons if the qualification gate fails.
set -u
MANIFEST=${1:?usage: sweep.sh <manifest>}
ROUNDS=${ROUNDS:-21}        # microbench rounds; round 0 discarded
QPAIRS=${QPAIRS:-8}         # A/A qualification pairs (ref vs itself)
STDROUNDS=${STDROUNDS:-9}   # timed std-build rounds; round 0 discarded
CORE=${CORE:-2}             # isolated core for microbenches
STDCORES=${STDCORES:-2,3}   # isolated cores for timed builds
OUT=${OUT:-$HOME/results-sweep}

declare -A DIR; ARMS=()
while IFS='=' read -r k v; do
  [ -z "$k" ] && continue
  DIR[$k]="$v"; ARMS+=("$k")
done < "$MANIFEST"
REF=${ARMS[0]}

mkdir -p "$OUT"; rm -f "$OUT"/*.txt

irq() { awk -v c=$((CORE+2)) 'NR>1{s+=$c}END{print s}' /proc/interrupts; }
{
  echo "date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:        $(hostname)   machine: $(curl -s -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/machine-type | sed 's|.*/||')"
  echo "cpu:         $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "kernel:      $(uname -r)   cmdline: $(cat /proc/cmdline)"
  echo "bench core:  $CORE (GOMAXPROCS=1)   std cores: $STDCORES (GOMAXPROCS=2, -p 2)"
  echo "rounds:      micro=$ROUNDS (r0 discarded)  qualify pairs=$QPAIRS  std=$STDROUNDS (r0 discarded)"
  echo "irq[$CORE] at start: $(irq)"
  for a in "${ARMS[@]}"; do
    d="$HOME/${DIR[$a]}"
    thr=$(grep -o 'allLoopsSimple(s\.loopnest, [0-9]*)' "$d/src/cmd/compile/internal/ssacompile/regalloc.go" 2>/dev/null | grep -o '[0-9]*')
    echo "arm $a: dir=${DIR[$a]} commit=$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo n/a) thr=${thr:-none} diffvsref=$(diff "$HOME/${DIR[$REF]}/src/cmd/compile/internal/ssacompile/regalloc.go" "$d/src/cmd/compile/internal/ssacompile/regalloc.go" 2>/dev/null | grep -c '^[<>]')"
  done
  echo "bench md5:   $(md5sum "$HOME/${DIR[$REF]}/src/cmd/compile/internal/ssacompile/regalloc_bench_test.go" | cut -d' ' -f1)"
} > "$OUT/machine.txt"

bench_one() {
  local d="$HOME/${DIR[$1]}"
  cd "$d/src/cmd/compile/internal/ssacompile" || return 1
  taskset -c "$CORE" env GOMAXPROCS=1 GOROOT="$d" "$d/bin/go" \
      test -run '^$' -bench 'BenchmarkComputeLive' -count=1 -timeout=60m 2>/dev/null
}

std_one() {  # timed `go build -a std`, private cache on tmpfs, prints ms or FAIL
  local d="$HOME/${DIR[$1]}" c="/dev/shm/gocache.$1"
  rm -rf "$c"
  local s e
  s=$(date +%s%3N)
  if taskset -c "$STDCORES" env GOMAXPROCS=2 GOCACHE="$c" GOROOT="$d" \
      "$d/bin/go" build -p 2 -a std >/dev/null 2>&1; then
    e=$(date +%s%3N); echo $((e - s))
  else
    echo FAIL
  fi
  rm -rf "$c"
}

# ---- Phase Q: A/A qualification, ref vs itself, order alternating ----
for ((q=0; q<QPAIRS; q++)); do
  if (( q % 2 == 0 )); then
    bench_one "$REF" >> "$OUT/qual_a.txt"; bench_one "$REF" >> "$OUT/qual_b.txt"
  else
    bench_one "$REF" >> "$OUT/qual_b.txt"; bench_one "$REF" >> "$OUT/qual_a.txt"
  fi
  echo "qualify pair $q done $(date -u +%H:%M:%S)"
done
echo "QUALIFY DONE"

# ---- Phase M: interleaved microbench sweep ----
for ((r=0; r<ROUNDS; r++)); do
  if (( r % 2 == 0 )); then order=("${ARMS[@]}"); else
    order=(); for ((i=${#ARMS[@]}-1;i>=0;i--)); do order+=("${ARMS[$i]}"); done
  fi
  for a in "${order[@]}"; do
    out=$(bench_one "$a")
    if (( r > 0 )); then echo "$out" >> "$OUT/$a.txt"; fi
  done
  echo "micro round $r done $(date -u +%H:%M:%S)"
done
echo "MICRO DONE"

# ---- Phase S: timed std builds, interleaved ----
for ((r=0; r<STDROUNDS; r++)); do
  if (( r % 2 == 0 )); then order=("${ARMS[@]}"); else
    order=(); for ((i=${#ARMS[@]}-1;i>=0;i--)); do order+=("${ARMS[$i]}"); done
  fi
  for a in "${order[@]}"; do
    ms=$(std_one "$a")
    if (( r > 0 )); then echo "$ms" >> "$OUT/std_$a.txt"; fi
  done
  echo "std round $r done $(date -u +%H:%M:%S)"
done
echo "irq[$CORE] at end: $(irq)" >> "$OUT/machine.txt"
echo "SWEEP COMPLETE $(date -u +%H:%M:%S)"
