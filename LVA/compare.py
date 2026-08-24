#!/usr/bin/env python3
"""
Compare "final ... live values at end of each block" sections between two debug dumps.

A section begins with a line like:
  final: live values at end of each block: <FUNC>
or:
  final (SCC 2-pass): live values at end of each block: <FUNC>

Within a section, we parse block summaries like:
  b1: v8(459) v9(12)[R0] ... avoid=R0 R1 R2

Comparison is order-insensitive for register lists (e.g. [R0,R1] == [R1,R0])
and for avoid sets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
from typing import Dict, Tuple, Optional, Iterable, List


SECTION_RE = re.compile(
    r"^final(?:\s*\((?P<desc>[^)]*)\))?:\s*live values at end of each block:\s*(?P<func>.+?)\s*$"
)
BLOCK_RE = re.compile(r"^\s*b(?P<bid>\d+):\s*(?P<body>.*)$")
# Match avoid=... at end of line
AVOID_RE = re.compile(r"\bavoid=(?P<avoid>.+?)\s*$")
# vNNN(123)[R0,R1]   -- regs optional, inside brackets
VAR_RE = re.compile(r"^(?P<v>v\d+)\((?P<n>\d+)\)(?:\[(?P<regs>[^\]]+)\])?$")


def file_label(path: str) -> str:
    stem = Path(path).stem
    stem = stem.removeprefix("debug_")
    return stem


def section_label(path: str, desc: Optional[str]) -> str:
    # If parentheses exist, show that; otherwise show the file’s label (e.g. master / iterative)
    return desc.strip() if desc and desc.strip() else file_label(path)


def comparison_header(left_path: str, left_desc: Optional[str],
                      right_path: str, right_desc: Optional[str]) -> str:
    return f"{section_label(left_path, left_desc)} - {section_label(right_path, right_desc)}"


def _split_regs(regs_str: str) -> Tuple[str, ...]:
    # Accept both comma and/or whitespace separated lists.
    toks = [t for t in re.split(r"[,\s]+", regs_str.strip()) if t]
    # Normalize order so comparisons are order-insensitive.
    return tuple(sorted(toks))


@dataclass(frozen=True)
class BlockState:
    vars: Dict[str, Tuple[int, Tuple[str, ...]]]  # vX -> (n, (regs... sorted))
    avoid: Tuple[str, ...]                        # sorted unique


@dataclass
class Section:
    desc: Optional[str]
    header_line: str
    blocks: Dict[int, BlockState]


def parse_file(path: str) -> Dict[str, Section]:
    """
    Returns:
      { func_name : Section(desc, header_line, blocks={bid: BlockState(...)}) }
    """
    sections: Dict[str, Section] = {}
    cur_func: Optional[str] = None
    cur_desc: Optional[str] = None
    cur_header: Optional[str] = None

    def ensure(func: str, desc: Optional[str], header_line: str) -> Section:
        if func not in sections:
            sections[func] = Section(desc=desc, header_line=header_line, blocks={})
        else:
            # Keep first header_line; update desc if we didn't have one.
            if sections[func].desc is None and desc:
                sections[func].desc = desc
        return sections[func]

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            msec = SECTION_RE.match(line)
            if msec:
                cur_desc = msec.group("desc")
                cur_func = msec.group("func").strip()
                cur_header = line
                ensure(cur_func, cur_desc, cur_header)
                continue

            if cur_func is None:
                continue

            # Optional: stop parsing a section when the dump goes into other phases.
            if line.startswith("Begin processing block"):
                cur_func = None
                cur_desc = None
                cur_header = None
                continue

            mb = BLOCK_RE.match(line)
            if not mb:
                continue

            bid = int(mb.group("bid"))
            body = mb.group("body").strip()

            # Extract avoid=... (order-insensitive)
            avoid: Tuple[str, ...] = ()
            mavoid = AVOID_RE.search(body)
            if mavoid:
                avoid_str = mavoid.group("avoid").strip()
                # avoid tokens are whitespace separated in the dump
                toks = [t for t in avoid_str.split() if t]
                avoid = tuple(sorted(set(toks)))
                body = body[:mavoid.start()].strip()

            vars_map: Dict[str, Tuple[int, Tuple[str, ...]]] = {}
            for tok in body.split():
                vm = VAR_RE.match(tok)
                if not vm:
                    continue
                vname = vm.group("v")
                n = int(vm.group("n"))
                regs = _split_regs(vm.group("regs") or "")
                vars_map[vname] = (n, regs)

            sections[cur_func].blocks[bid] = BlockState(vars=vars_map, avoid=avoid)

    return sections




def compare_sections(pathA: str, A: Dict[str, Section],
                     pathB: str, B: Dict[str, Section],
                     max_var_diffs_per_block: int = 25) -> int:
    """Print a human-readable diff.

    Returns:
      0 if no diffs were found
      1 if any diffs were found
    """
    labelA, labelB = file_label(pathA), file_label(pathB)
    diffs_found = False

    keysA, keysB = set(A.keys()), set(B.keys())
    onlyA = sorted(keysA - keysB)
    onlyB = sorted(keysB - keysA)
    common = sorted(keysA & keysB)

    if onlyA:
        diffs_found = True
        print(f"Sections only in {labelA}:")
        for n in onlyA:
            print(f"  - {n}")
        print()

    if onlyB:
        diffs_found = True
        print(f"Sections only in {labelB}:")
        for n in onlyB:
            print(f"  - {n}")
        print()

    # Only print a function section if there is at least one diff within it.
    for func in common:
        secA, secB = A[func], B[func]
        hdr = comparison_header(pathA, secA.desc, pathB, secB.desc)

        blocksA, blocksB = secA.blocks, secB.blocks
        bidsA, bidsB = set(blocksA.keys()), set(blocksB.keys())
        b_onlyA = sorted(bidsA - bidsB)
        b_onlyB = sorted(bidsB - bidsA)
        b_common = sorted(bidsA & bidsB)

        func_lines: List[str] = []
        func_has_diff = False

        if b_onlyA:
            func_has_diff = True
            func_lines.append(
                f"  blocks only in {labelA}: " + ", ".join(f"b{x}" for x in b_onlyA)
            )
        if b_onlyB:
            func_has_diff = True
            func_lines.append(
                f"  blocks only in {labelB}: " + ", ".join(f"b{x}" for x in b_onlyB)
            )

        for bid in b_common:
            a, b = blocksA[bid], blocksB[bid]

            vA, vB = set(a.vars.keys()), set(b.vars.keys())
            v_onlyA = sorted(vA - vB)
            v_onlyB = sorted(vB - vA)

            v_changed: List[Tuple[str, Tuple[int, Tuple[str, ...]], Tuple[int, Tuple[str, ...]]]] = []
            for v in sorted(vA & vB):
                av, bv = a.vars[v], b.vars[v]
                if av != bv:
                    v_changed.append((v, av, bv))

            avoid_changed = a.avoid != b.avoid

            if v_onlyA or v_onlyB or v_changed or avoid_changed:
                func_has_diff = True
                func_lines.append("")
                func_lines.append(f"  == b{bid} ==")

                if v_onlyA:
                    head = ", ".join(v_onlyA[:50]) + (" ..." if len(v_onlyA) > 50 else "")
                    func_lines.append(f"    vars only in {labelA}: {head}")
                if v_onlyB:
                    head = ", ".join(v_onlyB[:50]) + (" ..." if len(v_onlyB) > 50 else "")
                    func_lines.append(f"    vars only in {labelB}: {head}")

                for (v, (an, aregs), (bn, bregs)) in v_changed[:max_var_diffs_per_block]:
                    func_lines.append(
                        f"    {v}: {labelA}=({an},{list(aregs)}) {labelB}=({bn},{list(bregs)})"
                    )
                if len(v_changed) > max_var_diffs_per_block:
                    func_lines.append(
                        f"    ... {len(v_changed) - max_var_diffs_per_block} more changed vars"
                    )

                if avoid_changed:
                    func_lines.append(
                        f"    avoid: {labelA}={list(a.avoid)} {labelB}={list(b.avoid)}"
                    )

        if func_has_diff:
            diffs_found = True
            print(f"=== {func} ({hdr}) ===")
            for ln in func_lines:
                print(ln)
            print()

    if not diffs_found:
        print("No differences found.")

    return 1 if diffs_found else 0



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", help="first debug dump file")
    ap.add_argument("right", help="second debug dump file")
    ap.add_argument("--max-var-diffs-per-block", type=int, default=25)
    args = ap.parse_args()

    A = parse_file(args.left)
    B = parse_file(args.right)
    return compare_sections(args.left, A, args.right, B, max_var_diffs_per_block=args.max_var_diffs_per_block)


if __name__ == "__main__":
    raise SystemExit(main())
