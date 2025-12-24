#!/usr/bin/env python3
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scc_csv_parser import iter_scc_rows, get_scc_files, parse_scc_components


def hist_int(series: pd.Series, outpath: Path, title: str, xlabel: str, logx: bool = False):
    x = series.dropna().astype(int)
    if x.empty:
        return

    plt.figure()

    if logx:
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

    files = get_scc_files(indir, args.pattern)

    rows = []
    n_rows = 0
    ctr_nontrivial_count = Counter()
    ctr_one_nontrivial_size = Counter()

    for file_path, line_num, row in iter_scc_rows(indir, args.pattern):
        if args.max_rows and n_rows >= args.max_rows:
            break

        scc_str = row.structure
        stripped = scc_str.replace(" ", "")

        # Acyclic CFGs: structure is [] and kernels == blocks
        if stripped in ("", "[]"):
            total_nodes = row.blocks
            num_scc = row.blocks
            nontriv_sizes = []
            num_nontriv = 0
            largest = 1 if row.blocks > 0 else 0
            nontriv_nodes = 0
            frac_nontriv = 0.0
            merge_mass = 0
            hhi = (1.0 / row.blocks) if row.blocks > 0 else np.nan
        else:
            comps = parse_scc_components(scc_str)
            if not comps:
                continue
            sizes_list = [len(c) for c in comps]
            total_nodes = sum(sizes_list)
            num_scc = len(sizes_list)
            nontriv_sizes = [sz for sz in sizes_list if sz > 1]
            num_nontriv = len(nontriv_sizes)
            largest = max(sizes_list)
            nontriv_nodes = sum(nontriv_sizes)
            frac_nontriv = nontriv_nodes / total_nodes if total_nodes > 0 else 0.0
            merge_mass = sum(sz - 1 for sz in sizes_list)
            hhi = sum((sz / total_nodes) ** 2 for sz in sizes_list) if total_nodes > 0 else np.nan

        is_all_singletons = (num_nontriv == 0)
        is_one_nontriv = (num_nontriv == 1)
        one_size = nontriv_sizes[0] if is_one_nontriv else np.nan
        is_loopy = not is_all_singletons  # has at least one cycle

        # For loopy CFGs, count SCCs by size (sizes_list already computed in else block)
        if is_loopy:
            num_singleton = sum(1 for sz in sizes_list if sz == 1)
            num_size2 = sum(1 for sz in sizes_list if sz == 2)
            num_size3 = sum(1 for sz in sizes_list if sz == 3)
        else:
            num_singleton = np.nan
            num_size2 = np.nan
            num_size3 = np.nan

        rows.append({
            "file": file_path.name,
            "row_number": line_num,
            "func": row.func,
            "blocks_hdr": row.blocks,
            "kernels_hdr": row.kernels,
            "blocks_parsed": total_nodes,
            "scc_count_parsed": num_scc,
            "nontriv_scc_count": num_nontriv,
            "largest_scc": largest,
            "nontriv_nodes": nontriv_nodes,
            "frac_nodes_in_nontriv_scc": frac_nontriv,
            "is_all_singletons": is_all_singletons,
            "is_loopy": is_loopy,
            "is_one_nontriv": is_one_nontriv,
            "one_nontriv_size": one_size,
            "num_singleton_scc": num_singleton,
            "num_size2_scc": num_size2,
            "num_size3_scc": num_size3,
            "merge_mass_parsed": merge_mass,
            "merge_mass_header": row.blocks - row.kernels,
            "hhi_sizes": hhi,
        })

        n_rows += 1
        ctr_nontrivial_count[num_nontriv] += 1
        if is_one_nontriv:
            ctr_one_nontrivial_size[int(one_size)] += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No valid rows parsed.")

    print(f"Files scanned: {len(files)}")
    print(f"Rows parsed  : {len(df)}")
    print()
    n_total = len(df)
    n_acyclic = df['is_all_singletons'].sum()
    n_loopy = df['is_loopy'].sum()
    print(f"Acyclic CFGs (all singleton SCCs): {n_acyclic:>7d}  ({100*n_acyclic/n_total:.2f}%)")
    print(f"Loopy CFGs (at least one cycle)  : {n_loopy:>7d}  ({100*n_loopy/n_total:.2f}%)")
    print(f"Exactly one non-trivial SCC      : {int(df['is_one_nontriv'].sum()):>7d}  ({100*df['is_one_nontriv'].mean():.2f}%)")
    print()

    # Stats for loopy CFGs only
    loopy_df = df[df["is_loopy"]].copy()
    if not loopy_df.empty:
        n_loopy = len(loopy_df)
        total_singletons = loopy_df["num_singleton_scc"].sum()
        total_size2 = loopy_df["num_size2_scc"].sum()
        total_size3 = loopy_df["num_size3_scc"].sum()
        total_sccs = loopy_df["scc_count_parsed"].sum()

        print("=== Loopy CFGs only ===")
        print(f"Total SCCs in loopy CFGs: {int(total_sccs)}")
        print(f"  Singleton SCCs (size=1): {int(total_singletons):>7d}  ({100*total_singletons/total_sccs:.2f}%)")
        print(f"  Size-2 SCCs            : {int(total_size2):>7d}  ({100*total_size2/total_sccs:.2f}%)")
        print(f"  Size-3 SCCs            : {int(total_size3):>7d}  ({100*total_size3/total_sccs:.2f}%)")
        print()

        # Per-CFG averages
        print("Per loopy CFG averages:")
        print(f"  Avg singleton SCCs: {loopy_df['num_singleton_scc'].mean():.2f}")
        print(f"  Avg size-2 SCCs   : {loopy_df['num_size2_scc'].mean():.2f}")
        print(f"  Avg size-3 SCCs   : {loopy_df['num_size3_scc'].mean():.2f}")
        print()

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

    print("Non-trivial SCC count per CFG (top buckets):")
    for k in sorted(ctr_nontrivial_count.keys())[:10]:
        print(f"  {k}: {ctr_nontrivial_count[k]}")
    if len(ctr_nontrivial_count) > 10:
        print("  ...")
    print()

    frac = df["frac_nodes_in_nontriv_scc"].to_numpy()
    print("Fraction of nodes in non-trivial SCCs (overall):")
    print({
        "median": float(np.nanpercentile(frac, 50)),
        "p90": float(np.nanpercentile(frac, 90)),
        "p99": float(np.nanpercentile(frac, 99)),
        "max": float(np.nanmax(frac)),
    })
    print()

    summary_csv = outdir / "scc_struct_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Wrote per-row summary: {summary_csv}")

    # plots
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
