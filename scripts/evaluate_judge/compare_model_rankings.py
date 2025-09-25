#!/usr/bin/env python3
"""
Compare model-wise rankings between human evaluation and DeBERTa (DelibJudge) predictions
on the specified test set.

- Maps each test item to a source model (model_a or model_b) using the user_dir field
  by reading assigned_user_data.json under the corresponding case directory in
  annotation/summary-rating/annotation_output/full_augment.
- Determines whether an item corresponds to model A or B via its id suffix (_A/_B).
- Aggregates per-model human scores (from test.jsonl) and predicted scores
  (from my_model_test_results.json detailed_results), computes averages and ranks.
- Reports rank correlations (Spearman and Kendall) between human and predicted rankings.

Usage:
  python scripts/evaluate_judge/compare_model_rankings.py \
      --test-jsonl /path/to/datasets/summary_rating_dataset/comment_summary_ratings/test.jsonl \
      --eval-results /path/to/results/evaluation/my_model_test_results.json \
      --full-augment-dir /path/to/annotation/summary-rating/annotation_output/full_augment \
      --output-csv /path/to/results/evaluation/model_ranking_comparison.csv \
      --output-json /path/to/results/evaluation/model_ranking_comparison.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


TARGET_KEYS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]

DIMENSION_NAMES = {
    "perspective_representation": "Representiveness",
    "informativeness": "Informativeness",
    "neutrality_balance": "Neutrality",
    "policy_approval": "Policy Approval",
}


def load_test_items(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_eval_results(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Accept both full dump with summary_statistics and detailed_results,
    # or a plain list of results.
    if isinstance(data, dict) and "detailed_results" in data:
        return data.get("detailed_results", [])
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported eval results format")


def build_triplet_to_models_map(full_augment_dir: Path) -> Dict[str, Tuple[str, str]]:
    """Scan case folders to map triplet_<id> -> (model_a, model_b).

    assigned_user_data.json stores entries keyed by ids like
    'triplet_<id>_comparison' with fields model_a/model_b.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    if not full_augment_dir.exists():
        raise FileNotFoundError(f"Full augment dir not found: {full_augment_dir}")

    for case_dir in full_augment_dir.iterdir():
        if not case_dir.is_dir():
            continue
        assigned_path = case_dir / "assigned_user_data.json"
        if not assigned_path.exists():
            continue
        try:
            data = json.loads(assigned_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Entries are nested; scan values for triplet_*_comparison entries
        if isinstance(data, dict):
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                if not isinstance(key, str):
                    continue
                if key.startswith("triplet_") and key.endswith("_comparison"):
                    triplet_key = key.replace("_comparison", "")  # e.g., triplet_1172
                    model_a = val.get("model_a") or val.get("modelA")
                    model_b = val.get("model_b") or val.get("modelB")
                    if model_a and model_b:
                        mapping[triplet_key] = (str(model_a), str(model_b))
    return mapping


def resolve_model_for_item(item: Dict[str, Any], triplet_to_models: Dict[str, Tuple[str, str]]) -> Optional[str]:
    item_id = item.get("id", "")
    # Extract base triplet key (e.g., triplet_1172)
    triplet_key: Optional[str] = None
    if isinstance(item_id, str) and item_id.startswith("triplet_"):
        # up to _rating or _comparison or _question
        parts = item_id.split("_")
        if len(parts) >= 2:
            triplet_key = f"{parts[0]}_{parts[1]}"  # triplet_<id>
    if not triplet_key:
        return None

    pair = triplet_to_models.get(triplet_key)
    if not pair:
        return None
    model_a, model_b = pair

    # Determine A/B from id suffix
    # e.g., triplet_1172_rating_B -> model_b
    if item_id.endswith("_A") or "_A_" in item_id:
        return model_a
    if item_id.endswith("_B") or "_B_" in item_id:
        return model_b
    # Fallback: if it contains "_rating_A/B"
    if "_rating_A" in item_id:
        return model_a
    if "_rating_B" in item_id:
        return model_b
    return None


def extract_human_scores(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    # Search in common containers
    for key in ("scores", "rating_scores", "labels", "targets"):
        obj = item.get(key)
        if isinstance(obj, dict):
            try:
                return {k: float(obj[k]) for k in TARGET_KEYS if k in obj}
            except (TypeError, ValueError):
                pass
    # Or direct keys
    out: Dict[str, float] = {}
    for k in TARGET_KEYS:
        if k in item:
            try:
                out[k] = float(item[k])
            except (TypeError, ValueError):
                continue
    return out if len(out) == len(TARGET_KEYS) else None


def extract_pred_scores(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if result.get("status") != "success":
        return None
    preds = result.get("predictions", {})
    if not isinstance(preds, dict):
        return None
    try:
        return {k: float(preds.get(k)) for k in TARGET_KEYS}
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare model-wise rankings between human eval and predicted eval")
    ap.add_argument("--test-jsonl", required=True, help="Path to test.jsonl")
    ap.add_argument("--eval-results", required=True, help="Path to my_model_test_results.json")
    ap.add_argument("--full-augment-dir", required=True, help="Path to annotation/.../full_augment directory")
    ap.add_argument("--output-csv", required=False, help="Output CSV path")
    ap.add_argument("--output-json", required=False, help="Output JSON summary path")
    args = ap.parse_args()

    test_path = Path(args.test_jsonl)
    eval_path = Path(args.eval_results)
    full_aug = Path(args.full_augment_dir)

    test_items = load_test_items(test_path)
    eval_results = load_eval_results(eval_path)
    if len(eval_results) < len(test_items):
        print(f"Warning: eval results ({len(eval_results)}) fewer than test items ({len(test_items)}); truncating alignment")
    n = min(len(test_items), len(eval_results))

    triplet_to_models = build_triplet_to_models_map(full_aug)
    if not triplet_to_models:
        print("Warning: no triplet -> (model_a, model_b) mappings found")

    # Aggregate per-model
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        item = test_items[i]
        result = eval_results[i]

        model_name = resolve_model_for_item(item, triplet_to_models)
        if not model_name:
            continue

        human = extract_human_scores(item)
        pred = extract_pred_scores(result)
        if human is None or pred is None:
            continue

        row = {
            "model": model_name,
        }
        for k in TARGET_KEYS:
            row[f"human_{k}"] = human.get(k)
            row[f"pred_{k}"] = pred.get(k)
        row["human_mean"] = float(np.mean([human[k] for k in TARGET_KEYS]))
        row["pred_mean"] = float(np.mean([pred[k] for k in TARGET_KEYS]))
        rows.append(row)

    if not rows:
        print("No aligned rows found; check mappings and inputs")
        return 1

    df = pd.DataFrame(rows)
    grouped = df.groupby("model")

    agg_parts = {}
    for prefix, colname in (("human_", "human_mean"), ("pred_", "pred_mean")):
        metrics = {f"{prefix}{k}": "mean" for k in TARGET_KEYS}
        metrics[colname] = "mean"
        part = grouped.agg(metrics)
        agg_parts[prefix] = part

    # Merge aggregated parts
    agg_df = agg_parts["human_"].join(agg_parts["pred_"], how="outer")

    # Ranks (1 = best highest mean)
    agg_df["human_rank"] = agg_df["human_mean"].rank(ascending=False, method="average")
    agg_df["pred_rank"] = agg_df["pred_mean"].rank(ascending=False, method="average")

    # Per-dimension ranks
    for dim in TARGET_KEYS:
        human_col = f"human_{dim}"
        pred_col = f"pred_{dim}"
        if human_col in agg_df.columns:
            agg_df[f"human_rank_{dim}"] = agg_df[human_col].rank(ascending=False, method="average")
        if pred_col in agg_df.columns:
            agg_df[f"pred_rank_{dim}"] = agg_df[pred_col].rank(ascending=False, method="average")

    # Rank correlations
    # Align to common index without NaNs
    common = agg_df.dropna(subset=["human_rank", "pred_rank"])  # models with both
    if len(common) >= 2:
        spear_r, spear_p = spearmanr(common["human_rank"], common["pred_rank"])  # type: ignore
        kend_r, kend_p = kendalltau(common["human_rank"], common["pred_rank"])  # type: ignore
    else:
        spear_r = spear_p = kend_r = kend_p = np.nan

    # Output
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        agg_df.sort_values(by=["human_mean"], ascending=False).to_csv(out_csv)

    summary = {
        "num_models": int(len(agg_df)),
        "spearman_rank_corr": None if np.isnan(spear_r) else float(spear_r),
        "spearman_p": None if np.isnan(spear_p) else float(spear_p),
        "kendall_tau": None if np.isnan(kend_r) else float(kend_r),
        "kendall_p": None if np.isnan(kend_p) else float(kend_p),
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "by_model": agg_df.reset_index().to_dict(orient="records")}, f, ensure_ascii=False, indent=2)

    # Print concise report
    print("\nModel-wise mean scores and ranks (sorted by human_mean):")
    display_cols = [
        "human_mean", "pred_mean", "human_rank", "pred_rank",
        "human_perspective_representation", "pred_perspective_representation",
        "human_informativeness", "pred_informativeness",
        "human_neutrality_balance", "pred_neutrality_balance",
        "human_policy_approval", "pred_policy_approval",
    ]
    # Include per-dimension ranks if present
    for dim in TARGET_KEYS:
        hr = f"human_rank_{dim}"
        pr = f"pred_rank_{dim}"
        if hr in agg_df.columns and pr in agg_df.columns:
            display_cols.extend([hr, pr])
    print(agg_df.sort_values(by=["human_mean"], ascending=False)[display_cols].round(4))

    print("\nRank agreement:")
    print(f"  Spearman rho: {summary['spearman_rank_corr']}")
    print(f"  Spearman p:   {summary['spearman_p']}")
    print(f"  Kendall tau:  {summary['kendall_tau']}")
    print(f"  Kendall p:    {summary['kendall_p']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


