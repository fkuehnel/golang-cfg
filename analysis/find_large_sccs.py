#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

from scc_csv_parser import iter_scc_rows, parse_scc_sizes


def analyze_scc_files(path: Path, pattern: str, limit_rows: Optional[int] = None) -> list[dict]:
    results: list[dict] = []
    count = 0

    for file_path, line_num, row in iter_scc_rows(path, pattern):
        if limit_rows and count >= limit_rows:
            break

        sizes = parse_scc_sizes(row.structure)

        # Empty structure means all-singleton SCCs
        if not sizes:
            max_cluster = 1 if row.blocks > 0 else 0
            nontriv_count = 0
        else:
            max_cluster = max(sizes)
            nontriv_count = sum(1 for s in sizes if s > 1)

        results.append({
            "file": file_path.name,
            "row_number": line_num,
            "func": row.func,
            "blocks": row.blocks,
            "kernels": row.kernels,
            "max_cluster_size": max_cluster,
            "nontriv_scc_count": nontriv_count,
        })
        count += 1

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: ./find_large_sccs.py <file_or_folder> [--top N] [--out out.csv] [--pattern PATTERN]", file=sys.stderr)
        sys.exit(1)

    target_path = Path(sys.argv[1]).expanduser().resolve()
    top_n = 50
    out_csv = None
    pattern = "*_scc.csv"

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            out_csv = args[i + 1]
            i += 2
        elif args[i] == "--pattern" and i + 1 < len(args):
            pattern = args[i + 1]
            i += 2
        else:
            i += 1

    results = analyze_scc_files(target_path, pattern)
    if not results:
        print(f"No rows found in {target_path}", file=sys.stderr)
        sys.exit(0)

    results.sort(key=lambda d: (d["max_cluster_size"], d["blocks"]), reverse=True)

    print(f"Found {len(results)} rows from {target_path}\n")
    header = f"{'file':28} {'row':>6} {'blocks':>8} {'kernels':>8} {'max_scc':>8} {'#nontriv':>8}  func"
    print(header)
    print("-" * len(header))

    for r in results[:top_n]:
        func = r["func"]
        if len(func) > 80:
            func = func[:77] + "..."
        print(
            f"{r['file'][:28]:28} {r['row_number']:6d} {r['blocks']:8d} {r['kernels']:8d} "
            f"{r['max_cluster_size']:8d} {r['nontriv_scc_count']:8d}  {func}"
        )

    if out_csv:
        import pandas as pd
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
