#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scc_csv_parser import iter_scc_rows, get_scc_files


# ---------------------------
# Negative Binomial utilities
# ---------------------------

def nb_mom_fit(x: pd.Series):
    x = x.dropna().astype(int)
    mu = float(x.mean()) if len(x) else float("nan")
    var = float(x.var(ddof=1)) if len(x) > 1 else 0.0

    if not np.isfinite(mu) or mu <= 0 or var <= mu:
        return {"mu": mu, "var": var, "r": float("inf"), "p": 1.0}

    r = (mu * mu) / (var - mu)
    p = r / (r + mu)
    return {"mu": mu, "var": var, "r": r, "p": p}


def nb_logpmf(k: int, r: float, p: float) -> float:
    return (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1 - p)
    )


def nb_pmf_array(kmax: int, r: float, p: float) -> np.ndarray:
    if not np.isfinite(r):
        return np.zeros(kmax + 1, dtype=float)

    pmf = np.empty(kmax + 1, dtype=float)
    for k in range(kmax + 1):
        pmf[k] = math.exp(nb_logpmf(k, r, p))

    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def ecdf_xy(x: pd.Series):
    x = x.dropna().astype(int)
    x = x[x > 0]
    xs = np.sort(x.to_numpy())
    n = xs.size
    ys = np.arange(1, n + 1) / n
    return xs, ys


# ---------------------------
# Plotting
# ---------------------------

def hist_logx_with_nb(
    x: pd.Series, fit: dict, title: str, outpath: Path,
    max_x: int = 500, max_y: int = 5000, nbins: int = 45, nb_min: int = 200,
):
    x = x.dropna().astype(int)
    x_pos = x[x > 0]
    if x_pos.empty:
        return

    bins = np.logspace(0, math.log10(max_x), nbins)
    bins[0] = 1.0
    bins[-1] = float(max_x)

    plt.figure()
    plt.hist(x_pos.to_numpy(), bins=bins)
    plt.xscale("log")
    plt.xlim(1, max_x)
    plt.ylim(0, max_y)
    plt.title(title)
    plt.xlabel("count (log scale)")
    plt.ylabel("Frequency")

    r, p = fit["r"], fit["p"]
    if np.isfinite(r) and 0 < p < 1:
        pmf = nb_pmf_array(max_x, r, p)
        cdf = np.cumsum(pmf)
        n = len(x_pos)

        exp_counts = []
        bin_centers = []

        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            center = math.sqrt(a * b)
            bin_centers.append(center)

            if b <= nb_min:
                exp_counts.append(np.nan)
                continue

            ka = max(nb_min, 0, int(math.ceil(a)))
            kb = min(max_x, int(math.floor(b - 1e-12)))

            if ka > kb:
                exp_counts.append(np.nan)
                continue

            prob = float(cdf[kb] - (cdf[ka - 1] if ka > 0 else 0.0))
            exp_counts.append(n * prob)

        plt.plot(bin_centers, exp_counts, marker="o", linestyle="-", label=f"NB fit (k ≥ {nb_min})")
        plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def combined_ecdf_logx_with_nb(
    blocks: pd.Series, kernels: pd.Series,
    fit_blocks: dict, fit_kernels: dict,
    outpath: Path, max_x: int = 500,
):
    plt.figure()

    xb, yb = ecdf_xy(blocks)
    xk, yk = ecdf_xy(kernels)

    if xb.size:
        plt.step(xb, yb, where="post", label="blocks (empirical)")
    if xk.size:
        plt.step(xk, yk, where="post", label="kernels (empirical)")

    def plot_nb_cdf(fit: dict, label: str):
        r, p = fit["r"], fit["p"]
        if not (np.isfinite(r) and 0 < p < 1):
            return
        pmf = nb_pmf_array(max_x, r, p)
        cdf = np.cumsum(pmf)
        xs = np.arange(1, max_x + 1)
        plt.plot(xs, cdf[1:], linestyle="--", label=label)

    plot_nb_cdf(fit_blocks, "blocks (NB fit)")
    plot_nb_cdf(fit_kernels, "kernels (NB fit)")

    plt.xscale("log")
    plt.xlim(1, max_x)
    plt.ylim(0, 1.0)
    plt.title("Combined ECDF (log x): blocks vs SCC kernels (empirical + NB fit)")
    plt.xlabel("count (log scale)")
    plt.ylabel("ECDF  P[X ≤ x]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def summary(x: pd.Series) -> dict:
    x = x.dropna().astype(int)
    if x.empty:
        return {}
    mu = float(x.mean())
    var = float(x.var(ddof=1)) if len(x) > 1 else 0.0
    disp = (var / mu) if mu > 0 else float("nan")
    return {
        "n": int(x.size),
        "min": int(x.min()),
        "median": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "max": int(x.max()),
        "mean": mu,
        "var": var,
        "dispersion(var/mean)": disp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="Directory containing *_scc.csv files")
    ap.add_argument("--pattern", default="*_scc.csv")
    ap.add_argument("--outdir", default="scc_nb_out")
    ap.add_argument("--max-x", type=int, default=500)
    ap.add_argument("--max-y", type=int, default=5000)
    ap.add_argument("--nb-min", type=int, default=200, help="Only draw NB overlay for k >= this")
    args = ap.parse_args()

    indir = Path(args.directory).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files = get_scc_files(indir, args.pattern)

    blocks_list = []
    kernels_list = []

    for _, _, row in iter_scc_rows(indir, args.pattern):
        blocks_list.append(row.blocks)
        kernels_list.append(row.kernels)

    blocks = pd.Series(blocks_list, dtype=int) if blocks_list else pd.Series(dtype=int)
    kernels = pd.Series(kernels_list, dtype=int) if kernels_list else pd.Series(dtype=int)

    print(f"Files: {len(files)}")
    print("Combined blocks :", summary(blocks))
    print("Combined kernels:", summary(kernels))

    fit_b = nb_mom_fit(blocks)
    fit_k = nb_mom_fit(kernels)
    print("\nNB fit (method-of-moments)")
    print("blocks :", fit_b, "  (r=size/shape, p=success prob)")
    print("kernels:", fit_k, "  (r=size/shape, p=success prob)")

    hist_logx_with_nb(
        blocks, fit_b,
        title="Histogram (log x) + NB tail overlay: #blocks",
        outpath=outdir / "blocks_hist_logx_nb.png",
        max_x=args.max_x, max_y=args.max_y, nb_min=args.nb_min,
    )
    hist_logx_with_nb(
        kernels, fit_k,
        title="Histogram (log x) + NB tail overlay: #SCC kernels",
        outpath=outdir / "kernels_hist_logx_nb.png",
        max_x=args.max_x, max_y=args.max_y, nb_min=args.nb_min,
    )

    combined_ecdf_logx_with_nb(
        blocks, kernels, fit_b, fit_k,
        outpath=outdir / "ecdf_logx_blocks_vs_kernels_nb.png",
        max_x=args.max_x,
    )

    print(f"\nWrote plots to: {outdir}")


if __name__ == "__main__":
    main()
