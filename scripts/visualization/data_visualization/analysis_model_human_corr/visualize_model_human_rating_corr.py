import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


RESULTS_DIR = Path(
    "/ibex/project/c2328/LLMs-Scalable-Deliberation/results/eval_llm_human_correlation"
)
OUTPUT_DIR = Path(
    "/ibex/project/c2328/LLMs-Scalable-Deliberation/results/analysis_model_human_corr/figures"
)


RATING_DIMS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


def load_rating_entries(file_path: Path) -> List[dict]:
    """Load rating entries from a correlation JSON file.

    The file structure contains a top-level key "rating_results" which is a list of items.
    Each item should include "human_ratings" and "llm_result"->"ratings" with identical keys.
    """
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rating_results = data.get("rating_results", [])
    return rating_results


def collect_pairs_per_dimension(entries: List[dict]) -> Dict[str, Tuple[List[float], List[float]]]:
    """Collect paired human and model rating lists for each rating dimension."""
    dim_to_pairs: Dict[str, Tuple[List[float], List[float]]] = {
        dim: ([], []) for dim in RATING_DIMS
    }
    for item in entries:
        human = item.get("human_ratings", {})
        llm = item.get("llm_result", {}).get("ratings", {})
        for dim in RATING_DIMS:
            if dim in human and dim in llm and human[dim] is not None and llm[dim] is not None:
                try:
                    h = float(human[dim])
                    m = float(llm[dim])
                except (TypeError, ValueError):
                    continue
                dim_to_pairs[dim][0].append(h)
                dim_to_pairs[dim][1].append(m)
    return dim_to_pairs


def compute_pearson_correlations(dim_to_pairs: Dict[str, Tuple[List[float], List[float]]]) -> Dict[str, float]:
    """Compute Pearson r for each dimension. Returns NaN if insufficient data."""
    corr: Dict[str, float] = {}
    for dim, (h_list, m_list) in dim_to_pairs.items():
        if len(h_list) >= 2 and len(m_list) >= 2:
            try:
                r, _ = pearsonr(h_list, m_list)
            except Exception:
                r = float("nan")
        else:
            r = float("nan")
        corr[dim] = r
    return corr


def infer_model_name(file_path: Path) -> str:
    """Infer model name from file name like human_llm_correlation_<model>.json."""
    stem = file_path.stem
    # Expect pattern: human_llm_correlation_<model>
    if stem.startswith("human_llm_correlation_"):
        return stem.replace("human_llm_correlation_", "")
    return stem


def find_input_files(results_dir: Path) -> List[Path]:
    """Find correlation files for human-model rating. We only use human_llm_correlation_*.json."""
    paths = []
    for p in results_dir.glob("human_llm_correlation_*.json"):
        if p.is_file():
            paths.append(p)
    return sorted(paths)


def plot_model_correlations(model_name: str, corr: Dict[str, float], output_dir: Path) -> Path:
    """Create and save a bar chart of rating dimension correlations for a single model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dims = RATING_DIMS
    values = [corr.get(d, np.nan) for d in dims]

    plt.figure(figsize=(6.5, 4.0), dpi=160)
    bars = plt.bar(dims, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])  # accessible palette
    plt.ylim(-1.0, 1.0)
    plt.axhline(0.0, color="#666666", linewidth=1, linestyle="--", alpha=0.6)
    plt.title(f"Human vs {model_name} rating correlations (Pearson r)")
    plt.ylabel("Pearson r")
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, values):
        if np.isfinite(val):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if val >= 0 else -0.04),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
            )

    plt.tight_layout()
    out_path = output_dir / f"model_rating_corr_{model_name}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    files = find_input_files(RESULTS_DIR)
    if not files:
        print(f"No input files found in {RESULTS_DIR}")
        return

    for fp in files:
        try:
            entries = load_rating_entries(fp)
        except Exception as e:
            print(f"Skip {fp.name}: failed to load ({e})")
            continue

        if not entries:
            print(f"Skip {fp.name}: no rating_results")
            continue

        dim_pairs = collect_pairs_per_dimension(entries)
        corr = compute_pearson_correlations(dim_pairs)
        model_name = infer_model_name(fp)
        out = plot_model_correlations(model_name, corr, OUTPUT_DIR)
        print(f"Saved: {out}")


if __name__ == "main":
    # Allow running as a module or script
    main()


