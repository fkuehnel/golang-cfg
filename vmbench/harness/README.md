# The honest measurement harness

Standing protocol for every timing claim in this project. Written 2026-08-25
after two harness failures produced confused data:

1. **Lying success detection.** `build128.log` printed `BUILD OK` immediately
   after a `go tool dist: FAILED` compile error, then "timed" a std build of a
   nonexistent binary at 1 ms (`thrsweep.log`: `std build(ms): 1 1 1`). The
   STATUS.md row "SCC path survives a lower dispatch threshold: BROKEN,
   undiagnosed" was undiagnosable *because* the record was fabricated by the
   harness, not by the compiler.
2. **Contact during measurement.** The 2026-08-25 morning sweep, left alone,
   held a 0.15% noise floor (A/A and A/A' controls: 50/50 rows `~`). The
   afternoon threshold sweep ran at ±2-3% because probes were compiled and run
   on core 0 during it — memory-bandwidth and LLC contention reach the
   isolated core even when the scheduler does not.

## The machine

A dedicated GCP c4-highcpu-8:
4 vCPU Emerald Rapids, no SMT, booted with `isolcpus=2,3 nohz_full=2,3
rcu_nocbs=2,3`. OS and ssh live on cores 0-1; measurement owns cores 2-3.
This box is retained deliberately: it has a *demonstrated* 0.15% floor, and
both prior failures were protocol failures, not hardware failures. A bigger
machine would speed builds but adds nothing to measurement honesty.

## Rules

1. **Builds are guilty until proven innocent.** A build claim requires, in
   order: the old binary deleted first, `make.bash` exit 0, `bin/go`
   recreated, `bin/go version` running, `go build os` passing (where the
   earlier threshold arms crashed), and a full `go build -a std`. Any step
   failing restores the stashed binary and reports FAIL with the log tail.
   See `../../vmbench/remote/` and `build_matrix.sh`.
2. **Arms differ by an audited diff.** Before a sweep, the harness greps the
   varied parameter out of each arm's source and counts changed lines against
   the reference arm; both go into `machine.txt`. An arm that cannot state its
   own diff does not run.
3. **No contact during measurement.** Between the sweep's start and its
   completion sentinel: no ssh logins, no compiles, no probe runs, no file
   pulls. Watchers sleep the sweep's expected duration before their first
   poll. Results are pulled once, after the sentinel.
4. **Every sweep carries its own controls.** Interleaved arms, direction
   alternating by round, round 0 discarded, pinned to an isolated core,
   GOMAXPROCS=1. The sweep opens with A/A qualification rounds (arm 1 vs
   itself); the analysis side refuses arm comparisons if the gate shows a
   significant row. Arms that share code on a given benchmark are additional
   free controls — read them before believing any small effect.
5. **Provenance or it didn't happen.** `machine.txt` records commit per arm,
   the varied parameter as greped from source, bench-file md5, kernel cmdline,
   and the pinned core's interrupt count before and after.
6. **Report the floor with the effect.** A number quoted without the run's
   measured A/A spread is not a result. Effects under the floor are "not
   resolved", never "confirmed small".

## Inventory

- `sweep.sh` — qualification + interleaved microbenchmark + timed std-build
  sweep over a manifest of arms. Emits `SWEEP COMPLETE` sentinel.
- `analyze.sh` — laptop-side: benchstat gate on the qualification rounds, then
  arm comparisons only if the gate passes.
- `make_instrumented.sh` (scratch) — builds the LIVESTATS census/convergence
  compiler; see `probes/dispatch-census/README.md` for what it answers.
