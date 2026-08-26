#!/usr/bin/env python3
"""Analyze a LIVESTATS raw log from the instrumented compiler.

  livestats.py <livestats.raw>

Answers two questions from one instrumented std+cmd build:

1. Parameter selection: how many real functions would take each computeLive
   path at every candidate allLoopsSimple threshold T. Derivable offline
   because the census logs maxdepth per function; no per-threshold builds
   needed for the census (only for timing).

2. 3-pass sufficiency: the distribution of changing sweeps per non-trivial
   SCC under the converge-and-count loop. lsweeps > 3 means the old constant
   cap of three would have truncated a still-growing under-approximation;
   lsweeps > 2 already falsifies the CL's "two passes are sufficient for ALL
   SCCs in our 290k-CFGs dataset" comment.
"""
import gzip
import re
import sys
from collections import Counter

path_of = {}


def main(fname):
    disp = []   # (fn, blocks, loops, maxdepth, irred)
    sccs = []   # (fn, size, red, lsweeps, dsweeps)
    iters = []  # (fn, sweeps, prepass, irred)
    rd = re.compile(r'^LIVEDISPATCH (.+?) blocks=(\d+) loops=(\d+) maxdepth=(\d+) irred=(\w+) path=(\w+)$')
    rs = re.compile(r'^LIVESCC (.+?) size=(\d+) red=(\w+) lsweeps=(\d+) dsweeps=(\d+)$')
    ri = re.compile(r'^LIVEITER (.+?) sweeps=(\d+) prepass=(\w+) irred=(\w+)$')
    opener = gzip.open if fname.endswith('.gz') else open
    with opener(fname, mode='rt', errors='replace') as f:
        for line in f:
            m = rd.match(line)
            if m:
                disp.append((m.group(1), int(m.group(2)), int(m.group(3)),
                             int(m.group(4)), m.group(5) == 'true'))
                continue
            m = rs.match(line)
            if m:
                sccs.append((m.group(1), int(m.group(2)), m.group(3) == 'true',
                             int(m.group(4)), int(m.group(5))))
                continue
            m = ri.match(line)
            if m:
                iters.append((m.group(1), int(m.group(2)),
                              m.group(3) == 'true', m.group(4) == 'true'))

    n = len(disp)
    print("functions in census: %d   (SCC records: %d, iter records: %d)"
          % (n, len(sccs), len(iters)))
    if not n:
        return

    print("\n== dispatch census: path share at candidate thresholds ==")
    print("%-6s %10s %10s %10s %10s" % ("T", "single", "acyclic", "iter", "scc"))
    for T in (1, 2, 3, 4, 5, 10, 10**9):
        c = Counter()
        for _, blocks, loops, maxd, _ in disp:
            if blocks == 1:
                c['single'] += 1
            elif loops == 0:
                c['acyclic'] += 1
            elif maxd <= T:
                c['iter'] += 1
            else:
                c['scc'] += 1
        label = 'inf' if T == 10**9 else str(T)
        print("%-6s %9.2f%% %9.2f%% %9.2f%% %9.2f%% (scc n=%d)" % (
            label, 100*c['single']/n, 100*c['acyclic']/n,
            100*c['iter']/n, 100*c['scc']/n, c['scc']))

    irred = [d for d in disp if d[4]]
    print("\nirreducible functions: %d (%.4f%%)" % (len(irred), 100*len(irred)/n))
    for fn, blocks, loops, maxd, _ in sorted(irred, key=lambda d: -d[1])[:10]:
        print("   %-60s blocks=%d loops=%d maxdepth=%d" % (fn, blocks, loops, maxd))

    print("\nmaxdepth distribution (loopy, multi-block functions):")
    md = Counter(d[3] for d in disp if d[1] > 1 and d[2] > 0)
    for k in sorted(md):
        print("   depth %2d: %7d" % (k, md[k]))

    if sccs:
        print("\n== 3-pass sufficiency (non-trivial SCCs on the scc path) ==")
        h = Counter(s[3] for s in sccs)
        tot = len(sccs)
        for k in sorted(h):
            print("   lsweeps=%2d: %7d (%6.3f%%)" % (k, h[k], 100*h[k]/tot))
        over2 = sum(v for k, v in h.items() if k > 2)
        over3 = sum(v for k, v in h.items() if k > 3)
        print("   changing sweeps > 2 (falsifies '2 passes sufficient'): %d (%.4f%%)"
              % (over2, 100*over2/tot))
        print("   changing sweeps > 3 (old cap produced WRONG liveness): %d (%.4f%%)"
              % (over3, 100*over3/tot))
        print("   worst offenders:")
        for fn, size, red, ls, ds in sorted(sccs, key=lambda s: -s[3])[:15]:
            print("   %-60s size=%-4d red=%-5s lsweeps=%d dsweeps=%d"
                  % (fn, size, red, ls, ds))

    full = [i for i in iters if not i[2]]  # prepass=False: iterated to fixpoint
    if full:
        print("\n== iterative path, full fixed-point runs ==")
        h = Counter(i[1] for i in full)
        for k in sorted(h):
            print("   sweeps=%2d: %7d" % (k, h[k]))
        irr = [i for i in full if i[3]]
        print("   of which irreducible: %d" % len(irr))
        for fn, sweeps, _, _ in sorted(irr, key=lambda i: -i[1])[:10]:
            print("   %-60s sweeps=%d" % (fn, sweeps))


if __name__ == '__main__':
    main(sys.argv[1])
