#!/usr/bin/env python3
import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")  # matches inner [ ... ] groups
BLOCK_TOKEN_RE = re.compile(r"b\d+")


def parse_scc_components(s: str) -> list[list[str]]:
    """
    Parse a string like: [[b1] [b6 b4 b10] [b11]]
    into: [['b1'], ['b6','b4','b10'], ['b11']]
    """
    # Find each inner bracket group content
    groups = BRACKET_GROUP_RE.findall(s)
    comps: list[list[str]] = []
    for g in groups:
        toks = BLOCK_TOKEN_RE.findall(g)
        if toks:
            comps.append(toks)
    return comps


def safe_int(x: str) -> int:
    return int(x.strip())


def hist_int(series: pd.Series, outpath: Path, title: str, xlabel: str, logx: bool = False):
    x = series.dropna().astype(int)
    if x.empty:
        return

    plt.figure()

    if logx:
        # Log-spaced bins for integer-ish heavy tails
        xmin = max(1, int(x.min()))
        xmax = int(x.max())
        bins = np.logspace(math.log10(xmin), math.log10(xmax), 50)
        plt.hist(x.to_numpy(), bins=bins)
        plt.xscale("log")
        plt.xlabel(f"{xlabel} (log scale)")
    else:
        mn, mx = int(x.min()), int(x.max())
        bins = np.arange(mn, mx + 2) - 0.5
        plt.hist(x.to_numpy(), bins=bins)
        plt.xlabel(xlabel)

    plt.title(title)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="Directory containing *_scc.csv files")
    ap.add_argument("--pattern", default="*_scc.csv", help="Glob pattern (default: *_scc.csv)")
    ap.add_argument("--outdir", default="scc_struct_out", help="Output directory (default: scc_struct_out)")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap for debugging (0 = no cap)")
    args = ap.parse_args()

    indir = Path(args.directory).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {indir} matching {args.pattern}")

    # Per-row stats (kept compact; ~290k rows is fine)
    rows = []

    # Streaming counters for quick headline stats
    n_rows = 0
    all_singletons = 0
    one_nontrivial = 0

    ctr_nontrivial_count = Counter()
    ctr_one_nontrivial_size = Counter()
    ctr_largest_scc = Counter()

    for f in files:
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                # split into 3 fields at most: blocks, clusters, scc_string
                parts = line.split(",", 2)
                if len(parts) < 3:
                    continue

                try:
                    blocks = safe_int(parts[0])
                    clusters = safe_int(parts[1])
                    scc_str = parts[2]
                except Exception:
                    continue

                comps = parse_scc_components(scc_str)
                sizes = [len(c) for c in comps]
                if not sizes:
                    continue

                num_scc = len(sizes)
                total_nodes = sum(sizes)
                nontriv_sizes = [sz for sz in sizes if sz > 1]
                num_nontriv = len(nontriv_sizes)
                largest = max(sizes)
                nontriv_nodes = sum(nontriv_sizes)
                frac_nontriv = nontriv_nodes / total_nodes if total_nodes > 0 else 0.0

                is_all_singletons = (num_nontriv == 0)
                is_one_nontriv = (num_nontriv == 1)
                one_size = nontriv_sizes[0] if is_one_nontriv else np.nan

                # "merge mass": sum(sz-1) over SCCs, equals blocks-clusters if header matches SCC parsing
                merge_mass = sum(sz - 1 for sz in sizes)  # = total_nodes - num_scc
                header_merge = blocks - clusters

                # concentration (Herfindahl index) of SCC sizes within a CFG: sum (sz/total)^2
                hhi = sum((sz / total_nodes) ** 2 for sz in sizes) if total_nodes > 0 else np.nan

                rows.append({
                    "file": f.name,
                    "blocks_hdr": blocks,
                    "clusters_hdr": clusters,
                    "blocks_parsed": total_nodes,
                    "scc_count_parsed": num_scc,
                    "nontriv_scc_count": num_nontriv,
                    "largest_scc": largest,
                    "nontriv_nodes": nontriv_nodes,
                    "frac_nodes_in_nontriv_scc": frac_nontriv,
                    "is_all_singletons": is_all_singletons,
                    "is_one_nontriv": is_one_nontriv,
                    "one_nontriv_size": one_size,
                    "merge_mass_parsed": merge_mass,
                    "merge_mass_header": header_merge,
                    "hhi_sizes": hhi,
                })

                # counters
                n_rows += 1
                all_singletons += int(is_all_singletons)
                one_nontrivial += int(is_one_nontriv)
                ctr_nontrivial_count[num_nontriv] += 1
                if is_one_nontriv:
                    ctr_one_nontrivial_size[int(one_size)] += 1
                ctr_largest_scc[largest] += 1

                if args.max_rows and n_rows >= args.max_rows:
                    break
        if args.max_rows and n_rows >= args.max_rows:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No valid rows parsed.")

    # Headline summary
    print(f"Files scanned: {len(files)}")
    print(f"Rows parsed  : {len(df)}")
    print()
    print(f"No-cycle SCCs (all singletons): {df['is_all_singletons'].mean():.3f}")
    print(f"Exactly one non-trivial SCC   : {df['is_one_nontriv'].mean():.3f}")
    print()

    # One-nontrivial size stats
    one_df = df[df["is_one_nontriv"]].copy()
    if not one_df.empty:
        s = one_df["one_nontriv_size"].astype(int)
        print("Single non-trivial SCC size (conditional on exactly one):")
        print({
            "n": int(len(s)),
            "min": int(s.min()),
            "median": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99)),
            "max": int(s.max()),
            "mean": float(s.mean()),
        })
        print()

    # Useful structural summaries
    print("Non-trivial SCC count per CFG (top buckets):")
    for k in sorted(ctr_nontrivial_count.keys())[:10]:
        print(f"  {k}: {ctr_nontrivial_count[k]}")
    if len(ctr_nontrivial_count) > 10:
        print("  ...")
    print()

    print("Fraction of nodes in non-trivial SCCs (overall):")
    frac = df["frac_nodes_in_nontriv_scc"].to_numpy()
    print({
        "median": float(np.nanpercentile(frac, 50)),
        "p90": float(np.nanpercentile(frac, 90)),
        "p99": float(np.nanpercentile(frac, 99)),
        "max": float(np.nanmax(frac)),
    })
    print()

    # Save per-row summary
    summary_csv = outdir / "scc_struct_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Wrote per-row summary: {summary_csv}")

    # Plots (combined, not per-file)
    hist_int(df["nontriv_scc_count"], outdir / "nontriv_scc_count_hist.png",
             "Non-trivial SCC count per CFG", "#non-trivial SCCs", logx=False)

    hist_int(df["largest_scc"], outdir / "largest_scc_hist_logx.png",
             "Largest SCC size (log x)", "largest SCC size", logx=True)

    hist_int(df["frac_nodes_in_nontriv_scc"], outdir / "frac_nodes_in_nontriv_hist.png",
             "Fraction of nodes in non-trivial SCCs", "fraction", logx=False)

    if not one_df.empty:
        hist_int(one_df["one_nontriv_size"], outdir / "one_nontriv_size_hist_logx.png",
                 "Size of the single non-trivial SCC (log x)", "SCC size", logx=True)

    print(f"Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
