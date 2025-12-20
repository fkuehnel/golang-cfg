#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Negative Binomial utilities
# Parameterization: NB(r, p) with pmf:
#   P(X=k) = C(k+r-1, k) * p^r * (1-p)^k,  k=0,1,2,...
# Mean = r(1-p)/p = mu
# Var  = mu + mu^2/r
# Method-of-moments: r = mu^2 / (var - mu), p = r / (r + mu)
# ---------------------------

def nb_mom_fit(x: pd.Series):
    x = x.dropna().astype(int)
    mu = float(x.mean()) if len(x) else float("nan")
    var = float(x.var(ddof=1)) if len(x) > 1 else 0.0

    if not np.isfinite(mu) or mu <= 0 or var <= mu:
        # Not over-dispersed (or degenerate). Fall back to "Poisson-like".
        # Represent as very large r (approaches Poisson).
        r = float("inf")
        p = 1.0
        return {"mu": mu, "var": var, "r": r, "p": p}

    r = (mu * mu) / (var - mu)
    p = r / (r + mu)
    return {"mu": mu, "var": var, "r": r, "p": p}


def nb_logpmf(k: int, r: float, p: float) -> float:
    # log( Gamma(k+r) / (Gamma(r) * k!) ) + r*log(p) + k*log(1-p)
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(p) + k * math.log(1 - p))


def nb_pmf_array(kmax: int, r: float, p: float) -> np.ndarray:
    # returns pmf for k=0..kmax
    if not np.isfinite(r):
        # Poisson-limit fallback not implemented here; but your data are over-dispersed,
        # so this shouldn't trigger. Still, return empty-safe.
        pmf = np.zeros(kmax + 1, dtype=float)
        return pmf

    pmf = np.empty(kmax + 1, dtype=float)
    for k in range(kmax + 1):
        pmf[k] = math.exp(nb_logpmf(k, r, p))
    # normalize tiny numerical drift
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def ecdf_xy(x: pd.Series):
    x = x.dropna().astype(int)
    x = x[x > 0]  # log-x can’t show 0
    xs = np.sort(x.to_numpy())
    n = xs.size
    ys = np.arange(1, n + 1) / n
    return xs, ys


# ---------------------------
# Plotting
# ---------------------------

def hist_logx_with_nb(x: pd.Series, fit: dict, title: str, outpath: Path,
                      max_x: int = 500, max_y: int = 5000, nbins: int = 45,
                      nb_min: int = 200):
    x = x.dropna().astype(int)
    x_pos = x[x > 0]  # log-x can’t show 0
    if x_pos.empty:
        return

    # Log-spaced bins from 1..max_x
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

    # NB overlay only for k >= nb_min
    r, p = fit["r"], fit["p"]
    if np.isfinite(r) and 0 < p < 1:
        pmf = nb_pmf_array(max_x, r, p)   # k=0..max_x
        cdf = np.cumsum(pmf)
        n = len(x_pos)

        exp_counts = []
        bin_centers = []

        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            center = math.sqrt(a * b)
            bin_centers.append(center)

            # if the bin is entirely below nb_min, skip drawing (NaN breaks the line)
            if b <= nb_min:
                exp_counts.append(np.nan)
                continue

            ka = int(math.ceil(a))
            kb = int(math.floor(b - 1e-12))
            ka = max(nb_min, 0, ka)   # <-- cutoff on the LOW end
            kb = min(max_x, kb)

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


def combined_ecdf_logx_with_nb(blocks: pd.Series, clusters: pd.Series,
                               fit_blocks: dict, fit_clusters: dict,
                               outpath: Path, max_x: int = 500):
    plt.figure()

    # Empirical ECDFs (solid)
    xb, yb = ecdf_xy(blocks)
    xc, yc = ecdf_xy(clusters)
    if xb.size:
        plt.step(xb, yb, where="post", label="blocks (empirical)")
    if xc.size:
        plt.step(xc, yc, where="post", label="clusters (empirical)")

    # Fitted NB CDFs (dashed, same color order as plotted above)
    # Build CDF up to max_x and plot as a line on log-x.
    def plot_nb_cdf(fit: dict, label: str):
        r, p = fit["r"], fit["p"]
        if not (np.isfinite(r) and 0 < p < 1):
            return
        pmf = nb_pmf_array(max_x, r, p)
        cdf = np.cumsum(pmf)
        xs = np.arange(1, max_x + 1)
        plt.plot(xs, cdf[1:], linestyle="--", label=label)

    plot_nb_cdf(fit_blocks, "blocks (NB fit)")
    plot_nb_cdf(fit_clusters, "kernels (NB fit)")

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


# ---------------------------
# Streaming read (all files)
# ---------------------------

def stream_all_counts(indir: Path, pattern: str, chunksize: int):
    files = sorted(indir.glob(pattern))
    if not files:
        raise SystemExit(f"No files found in {indir} matching {pattern}")

    blocks_parts = []
    clusters_parts = []

    for f in files:
        for chunk in pd.read_csv(
            f,
            header=None,
            usecols=[0, 1],
            names=["blocks", "clusters"],
            dtype="Int64",
            chunksize=chunksize,
            engine="c",
            on_bad_lines="skip",
        ):
            blocks_parts.append(chunk["blocks"].dropna().astype(int))
            clusters_parts.append(chunk["clusters"].dropna().astype(int))

    blocks = pd.concat(blocks_parts, ignore_index=True) if blocks_parts else pd.Series(dtype=int)
    clusters = pd.concat(clusters_parts, ignore_index=True) if clusters_parts else pd.Series(dtype=int)
    return files, blocks, clusters


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
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--outdir", default="scc_stats_out")
    ap.add_argument("--max-x", type=int, default=2000)
    ap.add_argument("--max-y", type=int, default=20000)
    args = ap.parse_args()

    indir = Path(args.directory).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files, blocks, clusters = stream_all_counts(indir, args.pattern, args.chunksize)

    print(f"Files: {len(files)}")
    print("Combined blocks  :", summary(blocks))
    print("Combined clusters:", summary(clusters))

    fit_b = nb_mom_fit(blocks)
    fit_c = nb_mom_fit(clusters)
    print("\nNB fit (method-of-moments)")
    print("blocks  :", fit_b, "  (r=size/shape, p=success prob)")
    print("clusters:", fit_c, "  (r=size/shape, p=success prob)")

    # Log-x histograms with NB overlay (expected counts per log bin)
    hist_logx_with_nb(blocks, fit_b, "Histogram (log x) + NB tail fit: #blocks",
        outdir / "blocks_hist_logx_nb.png",
        max_x=args.max_x, max_y=args.max_y, nb_min=20)

    hist_logx_with_nb(clusters, fit_c, "Histogram (log x) + NB tail fit: #clusters",
        outdir / "clusters_hist_logx_nb.png",
        max_x=args.max_x, max_y=args.max_y, nb_min=20)

    # One combined ECDF plot (empirical solid + NB dashed) with log x
    combined_ecdf_logx_with_nb(
        blocks, clusters, fit_b, fit_c,
        outpath=outdir / "ecdf_logx_blocks_vs_clusters_nb.png",
        max_x=args.max_x
    )

    print(f"\nWrote plots to: {outdir}")


if __name__ == "__main__":
    main()
