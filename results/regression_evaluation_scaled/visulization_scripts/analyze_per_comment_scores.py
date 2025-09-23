#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DIMENSIONS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


def load_scores(root: Path, model: str, size: int) -> pd.DataFrame:
    csv_path = root / model / str(size) / "per_comment_scores.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_trends(df: pd.DataFrame, model: str, size: int, outdir: Path, topic: Optional[str] = None) -> None:
    # Average per comment_index over summaries
    trend = (
        df.groupby("comment_index")[DIMENSIONS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Flatten columns
    trend.columns = [
        "comment_index"
        if c[0] == "comment_index"
        else f"{c[0]}_{c[1]}" for c in trend.columns
    ]

    # Plot each dimension mean vs comment_index
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        ax.plot(trend["comment_index"], trend[f"{dim}_mean"], label=f"{dim}")
        # Add a simple linear fit
        x = trend["comment_index"].values
        y = trend[f"{dim}_mean"].values
        if len(x) > 1 and np.isfinite(y).all():
            coef = np.polyfit(x, y, 1)
            yfit = np.poly1d(coef)(x)
            ax.plot(x, yfit, linestyle="--", color="red", alpha=0.7, label=f"slope={coef[0]:.4f}")
        ax.set_title(dim)
        ax.set_xlabel("comment_index")
        ax.set_ylabel("mean score")
        ax.legend()
    title_suffix = f" — {topic}" if topic else ""
    fig.suptitle(f"Trend by comment_index — {model} (size={size}){title_suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    topic_tag = f"_{topic}" if topic else ""
    out_path = outdir / f"trend_{model}_{size}{topic_tag}.pdf"
    fig.savefig(out_path)
    plt.close(fig)


def plot_minority_comparison(df: pd.DataFrame, model: str, size: int, outdir: Path, topic: Optional[str] = None) -> None:
    # Ensure boolean dtype; keep rows with explicit True/False
    if "is_minority" not in df.columns:
        return
    filt = df["is_minority"].isin([True, False, "True", "False", 1, 0, "1", "0"])
    sub = df.loc[filt].copy()
    if sub.empty:
        return
    # Normalize to boolean
    sub["is_minority"] = sub["is_minority"].map(lambda v: True if str(v).lower() in {"true", "1"} else False)

    sns.set(style="whitegrid")
    # Boxplots per group
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.ravel()
    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        sns.boxplot(data=sub, x="is_minority", y=dim, ax=ax)
        ax.set_title(f"{dim} by is_minority")
        ax.set_xlabel("is_minority")
        ax.set_ylabel(dim)
    title_suffix = f" — {topic}" if topic else ""
    fig.suptitle(f"Group comparison — {model} (size={size}){title_suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    topic_tag = f"_{topic}" if topic else ""
    out_path = outdir / f"minority_box_{model}_{size}{topic_tag}.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    # Trend lines per group (mean over comment_index)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        grouped = (
            sub.groupby(["is_minority", "comment_index"])[dim]
            .mean()
            .reset_index()
        )
        sns.lineplot(data=grouped, x="comment_index", y=dim, hue="is_minority", ax=ax)
        # Fit slopes per group
        for group_val, gdf in grouped.groupby("is_minority"):
            x = gdf["comment_index"].values
            y = gdf[dim].values
            if len(x) > 1 and np.isfinite(y).all():
                coef = np.polyfit(x, y, 1)
                ax.plot(x, np.poly1d(coef)(x), linestyle=":", alpha=0.7,
                        label=f"{'minority' if group_val else 'non-minority'} slope={coef[0]:.4f}")
        ax.set_title(f"{dim} trend by is_minority")
        ax.set_xlabel("comment_index")
        ax.set_ylabel(dim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"Trend by is_minority — {model} (size={size}){title_suffix}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = outdir / f"minority_trend_{model}_{size}{topic_tag}.pdf"
    fig.savefig(out_path)
    plt.close(fig)


def save_summary_stats(df: pd.DataFrame, model: str, size: int, outdir: Path, topic: Optional[str] = None) -> None:
    rows = []
    # Slopes over comment_index overall and by group
    def slope(x, y):
        if len(x) < 2 or not np.isfinite(y).all():
            return np.nan
        return np.polyfit(x, y, 1)[0]

    x = df["comment_index"].values
    for dim in DIMENSIONS:
        rows.append({
            "model": model,
            "sample_size": size,
            "topic": topic if topic else "ALL",
            "dimension": dim,
            "group": "all",
            "slope": slope(x, df[dim].values),
            "mean": df[dim].mean(),
        })
    if "is_minority" in df.columns:
        filt = df["is_minority"].isin([True, False, "True", "False", 1, 0, "1", "0"])
        sub = df.loc[filt].copy()
        if not sub.empty:
            sub["is_minority"] = sub["is_minority"].map(lambda v: True if str(v).lower() in {"true", "1"} else False)
            for grp, gdf in sub.groupby("is_minority"):
                xg = gdf["comment_index"].values
                for dim in DIMENSIONS:
                    rows.append({
                        "model": model,
                        "sample_size": size,
                        "topic": topic if topic else "ALL",
                        "dimension": dim,
                        "group": "minority" if grp else "non_minority",
                        "slope": slope(xg, gdf[dim].values),
                        "mean": gdf[dim].mean(),
                    })
    summ = pd.DataFrame(rows)
    topic_tag = f"_{topic}" if topic else "_ALL"
    out_csv = outdir / f"summary_{model}_{size}{topic_tag}.csv"
    summ.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze per-comment scores vs comment_index and is_minority, per topic.")
    parser.add_argument("--root", type=str, default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/regression_evaluation_scaled",
                        help="Root directory containing <model>/<size>/per_comment_scores.csv")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names; default uses subdirectories under root")
    parser.add_argument("--sizes", type=str, default="500,1000",
                        help="Comma-separated sample sizes")
    parser.add_argument("--out-dir", type=str, default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/regression_evaluation_scaled/visulization_scripts/outputs",
                        help="Directory to save plots and summary CSVs")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.out_dir)
    ensure_outdir(outdir)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [p.name for p in root.iterdir() if p.is_dir()]
    sizes: List[int] = [int(s) for s in args.sizes.split(",") if s.strip()]

    for model in models:
        for size in sizes:
            try:
                df = load_scores(root, model, size)
            except FileNotFoundError:
                continue
            # Basic cleaning
            keep_cols = {"topic", "model", "sample_size", "summary_index", "summary_file", "comment_index", "is_minority", *DIMENSIONS}
            df = df[[c for c in df.columns if c in keep_cols]].copy()
            # Coerce numeric
            for col in ["comment_index", *DIMENSIONS]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["comment_index"]).sort_values("comment_index")

            # Per-topic analysis
            topics = sorted(df["topic"].dropna().unique().tolist()) if "topic" in df.columns else []
            if topics:
                for topic in topics:
                    tdf = df[df["topic"] == topic].copy()
                    if tdf.empty:
                        continue
                    plot_trends(tdf, model, size, outdir, topic)
                    plot_minority_comparison(tdf, model, size, outdir, topic)
                    save_summary_stats(tdf, model, size, outdir, topic)
            else:
                # Fallback to overall
                plot_trends(df, model, size, outdir, None)
                plot_minority_comparison(df, model, size, outdir, None)
                save_summary_stats(df, model, size, outdir, None)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
