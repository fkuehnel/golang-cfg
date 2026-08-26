#!/usr/bin/env python3
"""Compare final liveness between two regalloc debug dumps.

The two arms label their dumps differently and dump at different points, so
sections cannot be paired by label. For each function we take its LAST section
in each file -- that is the final state whatever the label -- and compare only
functions present in both.

Reports value-set differences (the correctness question) separately from
distance differences (the question the reviewer keeps asking about).
"""
import re, sys, collections

HDR = re.compile(r": live values at end of each block: (.+?)\s*$")
BLK = re.compile(r"^\s*(b\d+):\s*(.*)$")
VAL = re.compile(r"(v\d+)\((-?\d+)\)")

def parse(path):
    last = {}
    cur_name, cur = None, None
    with open(path, errors="replace") as f:
        for line in f:
            m = HDR.search(line)
            if m:
                if cur_name is not None:
                    last[cur_name] = cur
                cur_name, cur = m.group(1), {}
                continue
            if cur is None:
                continue
            b = BLK.match(line)
            if b:
                body = b.group(2)
                body = body.split("avoid=")[0]
                cur[b.group(1)] = {v: int(d) for v, d in VAL.findall(body)}
            elif line.strip() == "":
                pass
    if cur_name is not None:
        last[cur_name] = cur
    return last

A = parse(sys.argv[1]); B = parse(sys.argv[2])
common = sorted(set(A) & set(B))
print(f"functions in A: {len(A)}   in B: {len(B)}   compared: {len(common)}")

# GUARD. "0 differences" and "compared nothing" print identically, so an empty
# side reads as perfect agreement. That has happened: a toolchain silently
# failed to build, its dump was empty, and this script reported zero
# differences across zero functions. Refuse to report success on no data.
MIN = 100
if not A or not B or len(common) < MIN:
    print()
    print(f"FATAL: refusing to report a result. A={len(A)} B={len(B)} "
          f"compared={len(common)} (need >= {MIN} in both).")
    print("An empty or tiny intersection means a dump is missing or the two "
          "sides do not name the same functions -- not that they agree.")
    sys.exit(2)
if len(common) < 0.5 * min(len(A), len(B)):
    print(f"WARNING: only {len(common)} of min({len(A)},{len(B)}) functions "
          f"matched; the dumps may not be comparable.")

setdiff = distdiff = same = 0
missing_in_B = extra_in_B = 0
examples = []
for fn in common:
    a, b = A[fn], B[fn]
    blocks = set(a) & set(b)
    s_ok = d_ok = True
    for blk in blocks:
        va, vb = set(a[blk]), set(b[blk])
        if va != vb:
            s_ok = False
            missing_in_B += len(va - vb)
            extra_in_B  += len(vb - va)
            if len(examples) < 5:
                examples.append((fn, blk, sorted(va-vb), sorted(vb-va)))
        else:
            if any(a[blk][v] != b[blk][v] for v in va):
                d_ok = False
    if not s_ok: setdiff += 1
    elif not d_ok: distdiff += 1
    else: same += 1

print()
print(f"  identical (values AND distances): {same}")
print(f"  same values, DIFFERENT distances: {distdiff}")
print(f"  DIFFERENT value sets:             {setdiff}")
print()
print(f"  value occurrences present in A but missing in B: {missing_in_B}")
print(f"  value occurrences present in B but not in A:     {extra_in_B}")
if examples:
    print("\n  first value-set differences:")
    for fn, blk, miss, extra in examples:
        print(f"    {fn} {blk}: missing_in_B={miss} extra_in_B={extra}")
