#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_summary(x: pd.Series) -> dict:
    x = x.dropna().astype(int)
    if x.empty:
        return {}

    mean = x.mean()
    var = x.var(ddof=1) if len(x) > 1 else 0.0
    disp = (var / mean) if mean > 0 else np.nan  # ~1 for Poisson

    return {
        "n": int(x.size),
        "min": int(x.min()),
        "p10": float(np.percentile(x, 10)),
        "median": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "max": int(x.max()),
        "mean": float(mean),
        "var": float(var),
        "dispersion(var/mean)": float(disp),
    }


def int_hist(x: pd.Series, title: str, xlabel: str, outpath: Path):
    x = x.dropna().astype(int)
    if x.empty:
        return

    # Histogram bins for 0..500 (integer-centered)
    bins = np.arange(0, 501 + 1) - 0.5

    plt.figure()
    plt.hist(x.to_numpy(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    plt.xlim(0, 500)
    plt.ylim(0, 10000)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def ecdf_plot(x: pd.Series, title: str, xlabel: str, outpath: Path):
    x = x.dropna().astype(int)
    if x.empty:
        return

    xs = np.sort(x.to_numpy())
    n = xs.size
    ys = np.arange(1, n + 1) / n

    plt.figure()
    # Step ECDF (classic)
    plt.step(xs, ys, where="post")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF (P[X ≤ x])")

    plt.xlim(0, 500)   # same x-range as requested
    plt.ylim(0, 1.0)   # ECDF is always 0..1

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def ecdf_plot_two(a: pd.Series, b: pd.Series, title: str, xlabel: str, outpath: Path,
                  label_a: str = "blocks", label_b: str = "clusters"):
    a = a.dropna().astype(int)
    b = b.dropna().astype(int)
    if a.empty and b.empty:
        return

    plt.figure()

    if not a.empty:
        xs = np.sort(a.to_numpy())
        ys = np.arange(1, xs.size + 1) / xs.size
        plt.step(xs, ys, where="post", label=label_a)  # color auto

    if not b.empty:
        xs = np.sort(b.to_numpy())
        ys = np.arange(1, xs.size + 1) / xs.size
        plt.step(xs, ys, where="post", label=label_b)  # different color auto

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF (P[X ≤ x])")
    plt.xlim(0, 500)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="Directory containing *_scc.csv files")
    ap.add_argument("--pattern", default="*_scc.csv", help="Glob pattern (default: *_scc.csv)")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk (default: 200000)")
    ap.add_argument("--outdir", default="scc_stats_out", help="Output directory for plots (default: scc_stats_out)")
    args = ap.parse_args()

    indir = Path(args.directory).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {indir} matching {args.pattern}")

    blocks_parts = []
    clusters_parts = []

    for f in files:
        for chunk in pd.read_csv(
            f,
            header=None,
            usecols=[0, 1],
            names=["blocks", "clusters"],
            dtype="Int64",
            chunksize=args.chunksize,
            engine="c",
            on_bad_lines="skip",
        ):
            blocks_parts.append(chunk["blocks"].dropna().astype(int))
            clusters_parts.append(chunk["clusters"].dropna().astype(int))

    blocks = pd.concat(blocks_parts, ignore_index=True) if blocks_parts else pd.Series(dtype=int)
    clusters = pd.concat(clusters_parts, ignore_index=True) if clusters_parts else pd.Series(dtype=int)

    print(f"Files: {len(files)}")
    print("Combined blocks  :", count_summary(blocks))
    print("Combined clusters:", count_summary(clusters))

    int_hist(blocks,   "Histogram: #blocks",   "#blocks",   outdir / "blocks_hist.png")
    int_hist(clusters, "Histogram: #SCC kernels", "#kernels", outdir / "clusters_hist.png")

    ecdf_plot_two(
        blocks, clusters,
        title="ECDF: blocks vs SCC kernels",
        xlabel="count",
        outpath=outdir / "blocks_vs_clusters_ecdf.png",
        label_a="#blocks",
        label_b="#kernels",
    )

    print(f"Wrote histograms to: {outdir}")

if __name__ == "__main__":
    main()
