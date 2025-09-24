#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# Ensure project imports work
PROJECT_DIR = Path("/ibex/project/c2328/LLMs-Scalable-Deliberation").resolve()
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))

from llm_evaluation.evaluator import SummaryEvaluator, DebertaEvaluator  # type: ignore


def find_size_dirs(model_dir: Path) -> List[Path]:
    size_dirs = []
    for p in sorted(model_dir.iterdir()):
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            size_dirs.append(p)
    return size_dirs


essential_json_fields = [
    ("metadata.dataset_name", list),
    ("metadata.question", str),
    ("summaries.main_points", str),
    ("comment_indices", list),
]


def get_nested(d: dict, path: str):
    cur = d
    for key in path.split('.'):
        if key not in cur:
            return None
        cur = cur[key]
    return cur


def select_topic_files(size_dir: Path, limit: int) -> List[Path]:
    candidates = sorted(size_dir.glob("*_summary_*.json"))
    return candidates[:limit]


def maybe_load_comments(comments_root: Path, dataset_name: str, indices: List[int]) -> Optional[List[str]]:
    # Attempt to find a JSONL or JSON with comments. Try common patterns.
    potential = [
        comments_root / f"{dataset_name}.jsonl",
        comments_root / f"{dataset_name}.json",
        comments_root / dataset_name / "comments.jsonl",
        comments_root / dataset_name / "comments.json",
    ]
    for path in potential:
        if path.exists():
            try:
                comments: List[str] = []
                if path.suffix == ".jsonl":
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            # Try common keys
                            text = obj.get("comment") or obj.get("text") or obj.get("content")
                            if isinstance(text, str):
                                comments.append(text)
                elif path.suffix == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for obj in data:
                            if not isinstance(obj, dict):
                                continue
                            text = obj.get("comment") or obj.get("text") or obj.get("content")
                            if isinstance(text, str):
                                comments.append(text)
                    elif isinstance(data, dict):
                        arr = data.get("comments") or data.get("items")
                        if isinstance(arr, list):
                            for obj in arr:
                                if not isinstance(obj, dict):
                                    continue
                                text = obj.get("comment") or obj.get("text") or obj.get("content")
                                if isinstance(text, str):
                                    comments.append(text)
                if comments:
                    # Map indices to comments if possible
                    selected = []
                    for idx in indices:
                        if 0 <= idx < len(comments):
                            selected.append(comments[idx])
                    return selected if selected else comments
            except Exception:
                continue
    return None


def synthesize_comments(n: int) -> List[str]:
    base = (
        "This is a placeholder opinion used for timing only. "
        "It has a moderate length to approximate real comments."
    )
    return [base for _ in range(max(n, 1))]


def measure_summary_evaluator(summary_text: str, comments: List[str], model_name: str, max_items: int) -> Tuple[float, int]:
    evaluator = SummaryEvaluator(model=model_name, temperature=1.0, verbose=False)
    num = min(len(comments), max_items)
    if num == 0:
        return 0.0, 0
    # Warmup single call (optional)
    try:
        _ = evaluator.evaluate_comment_representation(summary_text, comments[0])
    except Exception:
        pass
    start = time.perf_counter()
    done = 0
    for i in tqdm(range(num), desc=f"API {model_name}", leave=False):
        try:
            _ = evaluator.evaluate_comment_representation(summary_text, comments[i])
        except Exception:
            pass
        finally:
            done += 1
    elapsed = time.perf_counter() - start
    return elapsed, done


def measure_deberta_evaluator(question: str, summary_text: str, comments: List[str], model_path: Path, device: str, max_items: int) -> Tuple[float, int, Dict[str, Optional[float]]]:
    evaluator = DebertaEvaluator(model_path=str(model_path), device=device)
    num = min(len(comments), max_items)
    if num == 0:
        return 0.0, 0, {"perspective_representation": None, "informativeness": None, "neutrality_balance": None, "policy_approval": None}
    # Warmup
    try:
        _ = evaluator.evaluate_single(question, comments[0], summary_text)
    except Exception:
        pass
    start = time.perf_counter()
    done = 0
    sums: Dict[str, float] = {"perspective_representation": 0.0, "informativeness": 0.0, "neutrality_balance": 0.0, "policy_approval": 0.0}
    count_ok = 0
    for i in tqdm(range(num), desc="REG DeBERTa", leave=False):
        try:
            res = evaluator.evaluate_single(question, comments[i], summary_text)
            preds = res.get("predictions", {})
            if all(k in preds and isinstance(preds[k], (int, float)) for k in sums.keys()):
                for k in sums.keys():
                    sums[k] += float(preds[k])
                count_ok += 1
        except Exception:
            pass
        finally:
            done += 1
    elapsed = time.perf_counter() - start
    avgs: Dict[str, Optional[float]] = {}
    for k, v in sums.items():
        avgs[k] = (v / count_ok) if count_ok > 0 else None
    return elapsed, done, avgs


def main():
    parser = argparse.ArgumentParser(description="Measure per-item and total time; shows progress bars only.")
    parser.add_argument("--summaries-root", type=str, default=str(PROJECT_DIR / "results/summary_model_for_evaluation"), help="Root directory of summaries.")
    parser.add_argument("--summary-model", type=str, default="gpt-5-mini", help="Summary model subdirectory to use.")
    parser.add_argument("--comments-root", type=str, default=str(PROJECT_DIR / "datasets/annotation_V0_V1_dataset"), help="Root directory containing raw comments datasets.")
    parser.add_argument(
        "--api-models",
        type=str,
        default="gpt-5-mini,web-rev-claude-3-7-sonnet-20250219,gpt-4o-mini,grok-4-latest,qwen3-235b-a22b,qwen3-32b,qwen3-30b-a3b,qwen3-14b,qwen3-8b,qwen3-0.6b",
        help="Comma-separated API evaluator models to benchmark.",
    )
    parser.add_argument("--regression-model-path", type=str, default=str(PROJECT_DIR / "checkpoints/deberta_regression_base_final_v6_pair_split_noactivation"), help="Path to trained DeBERTa regression model.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for regression evaluator.")
    parser.add_argument("--max-items-per-size", type=int, default=3, help="Max number of items (comments) to time per summary.")
    parser.add_argument("--summaries-per-size", type=int, default=3, help="Number of summaries (topics) to evaluate per comment number.")
    parser.add_argument("--plot-total", type=str, default=str(PROJECT_DIR / "results/plots/total_time_vs_comments.pdf"), help="Output path for total time plot.")
    parser.add_argument("--plot-per-item", type=str, default=str(PROJECT_DIR / "results/plots/per_item_time_vs_comments.pdf"), help="Output path for per-item time plot.")
    parser.add_argument("--csv-out", type=str, default=str(PROJECT_DIR / "results/plots/eval_time_vs_comments.csv"), help="CSV file to save timing results.")

    args = parser.parse_args()

    summaries_root = Path(args.summaries_root).resolve()
    model_dir = summaries_root / args.summary_model
    comments_root = Path(args.comments_root).resolve()
    reg_model_path = Path(args.regression_model_path).resolve()

    api_models = [m.strip() for m in args.api_models.split(',') if m.strip()]

    if not model_dir.exists():
        print(f"Summaries directory not found: {model_dir}")
        sys.exit(1)

    size_dirs = find_size_dirs(model_dir)
    if not size_dirs:
        print(f"No comment-number directories found under: {model_dir}")
        sys.exit(1)

    # Storage: per model -> lists aligned with x_comment_nums
    x_comment_nums: List[int] = []
    api_per_item: Dict[str, List[float]] = {m: [] for m in api_models}
    api_total: Dict[str, List[float]] = {m: [] for m in api_models}
    reg_per_item: List[float] = []
    reg_total: List[float] = []

    # Prepare CSV rows
    csv_rows: List[Dict[str, object]] = []

    for size_dir in tqdm(size_dirs, desc="Comment sizes"):
        try:
            comment_num = int(size_dir.name)
        except ValueError:
            continue

        topic_files = select_topic_files(size_dir, limit=args.summaries_per_size)
        if not topic_files:
            continue

        # accumulators for aggregation
        api_model_to_per_item: Dict[str, List[float]] = {m: [] for m in api_models}
        api_model_to_items: Dict[str, List[int]] = {m: [] for m in api_models}
        reg_per_item_list: List[float] = []
        reg_items_list: List[int] = []

        for topic_file in topic_files:
            try:
                data = json.loads(topic_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            dataset_name = get_nested(data, "metadata.dataset_name") or ""
            question = get_nested(data, "metadata.question") or ""
            summary_text = get_nested(data, "summaries.main_points") or ""
            indices = get_nested(data, "comment_indices") or []

            if not question or not summary_text:
                continue

            # Load comments if available, else synthesize
            comments = maybe_load_comments(comments_root, dataset_name, indices)
            if not comments:
                comments = synthesize_comments(max(1, comment_num))

            # API models (per summary)
            for model_name in api_models:
                api_elapsed, api_count = measure_summary_evaluator(
                    summary_text=summary_text,
                    comments=comments,
                    model_name=model_name,
                    max_items=args.max_items_per_size,
                )
                api_avg = (api_elapsed / api_count) if api_count > 0 else 0.0
                total_time = api_avg * (api_count * comment_num)

                api_model_to_per_item[model_name].append(api_avg)
                api_model_to_items[model_name].append(api_count)

                # per-summary CSV row
                csv_rows.append({
                    "level": "summary",
                    "topic_file": str(topic_file.relative_to(summaries_root)),
                    "evaluator": "api",
                    "model": model_name,
                    "summary_model": args.summary_model,
                    "comment_num": comment_num,
                    "items": api_count,
                    "per_item_seconds": round(api_avg, 6),
                    "total_seconds": round(total_time, 6),
                    "perspective_representation": None,
                    "informativeness": None,
                    "neutrality_balance": None,
                    "policy_approval": None,
                })

            # Regression baseline (per summary)
            reg_elapsed, reg_count, reg_avg_preds = measure_deberta_evaluator(
                question=question,
                summary_text=summary_text,
                comments=comments,
                model_path=reg_model_path,
                device=args.device,
                max_items=args.max_items_per_size,
            )
            reg_avg = (reg_elapsed / reg_count) if reg_count > 0 else 0.0
            reg_total_time = reg_avg * (reg_count * comment_num)

            reg_per_item_list.append(reg_avg)
            reg_items_list.append(reg_count)

            csv_rows.append({
                "level": "summary",
                "topic_file": str(topic_file.relative_to(summaries_root)),
                "evaluator": "regression",
                "model": "deberta",
                "summary_model": args.summary_model,
                "comment_num": comment_num,
                "items": reg_count,
                "per_item_seconds": round(reg_avg, 6),
                "total_seconds": round(reg_total_time, 6),
                "perspective_representation": None if reg_avg_preds.get("perspective_representation") is None else round(reg_avg_preds.get("perspective_representation"), 6),
                "informativeness": None if reg_avg_preds.get("informativeness") is None else round(reg_avg_preds.get("informativeness"), 6),
                "neutrality_balance": None if reg_avg_preds.get("neutrality_balance") is None else round(reg_avg_preds.get("neutrality_balance"), 6),
                "policy_approval": None if reg_avg_preds.get("policy_approval") is None else round(reg_avg_preds.get("policy_approval"), 6),
            })

        # After summaries -> aggregate per size
        x_comment_nums.append(comment_num)

        # Aggregate API models
        for model_name in api_models:
            per_item_list = [v for v in api_model_to_per_item[model_name] if v is not None]
            items_list = [i for i in api_model_to_items[model_name] if i is not None]
            if per_item_list and items_list:
                avg_per_item = sum(per_item_list) / len(per_item_list)
                avg_items = sum(items_list) / len(items_list)
                total_time = avg_per_item * (avg_items * comment_num)
            else:
                avg_per_item = 0.0
                avg_items = 0.0
                total_time = 0.0

            api_per_item[model_name].append(avg_per_item)
            api_total[model_name].append(total_time)

            csv_rows.append({
                "level": "aggregate",
                "topic_file": None,
                "evaluator": "api",
                "model": model_name,
                "summary_model": args.summary_model,
                "comment_num": comment_num,
                "items": round(avg_items, 3),
                "per_item_seconds": round(avg_per_item, 6),
                "total_seconds": round(total_time, 6),
                "perspective_representation": None,
                "informativeness": None,
                "neutrality_balance": None,
                "policy_approval": None,
            })

        # Aggregate regression
        if reg_per_item_list and reg_items_list:
            reg_avg_per_item = sum(reg_per_item_list) / len(reg_per_item_list)
            reg_avg_items = sum(reg_items_list) / len(reg_items_list)
            reg_total_time = reg_avg_per_item * (reg_avg_items * comment_num)
        else:
            reg_avg_per_item = 0.0
            reg_avg_items = 0.0
            reg_total_time = 0.0

        reg_per_item.append(reg_avg_per_item)
        reg_total.append(reg_total_time)

        csv_rows.append({
            "level": "aggregate",
            "topic_file": None,
            "evaluator": "regression",
            "model": "deberta",
            "summary_model": args.summary_model,
            "comment_num": comment_num,
            "items": round(reg_avg_items, 3),
            "per_item_seconds": round(reg_avg_per_item, 6),
            "total_seconds": round(reg_total_time, 6),
            "perspective_representation": None,
            "informativeness": None,
            "neutrality_balance": None,
            "policy_approval": None,
        })

    # Sort by x and reorder arrays for plotting
    order = sorted(range(len(x_comment_nums)), key=lambda i: x_comment_nums[i])
    xs = [x_comment_nums[i] for i in order]

    api_per_item_sorted: Dict[str, List[float]] = {m: [api_per_item[m][i] for i in order] for m in api_models}
    api_total_sorted: Dict[str, List[float]] = {m: [api_total[m][i] for i in order] for m in api_models}
    reg_per_item_sorted = [reg_per_item[i] for i in order]
    reg_total_sorted = [reg_total[i] for i in order]

    # Plot total time
    total_path = Path(args.plot_total)
    total_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for m in api_models:
        plt.plot(xs, api_total_sorted[m], marker='o', label=f'API {m}')
    plt.plot(xs, reg_total_sorted, marker='s', label='Regression (DeBERTa)')
    plt.xlabel('Comment number')
    plt.ylabel('Estimated total time (seconds)')
    plt.title(f'Total evaluation time vs comment number\nSummary model: {args.summary_model}, items per summary: {args.max_items_per_size}, summaries per size: {args.summaries_per_size}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(total_path)
    print(f"Saved total time plot: {total_path}")

    # Plot per-item time
    per_item_path = Path(args.plot_per_item)
    per_item_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for m in api_models:
        plt.plot(xs, api_per_item_sorted[m], marker='o', label=f'API {m}')
    plt.plot(xs, reg_per_item_sorted, marker='s', label='Regression (DeBERTa)')
    plt.xlabel('Comment number')
    plt.ylabel('Average time per item (seconds)')
    plt.title(f'Average per-item time vs comment number\nSummary model: {args.summary_model}, items per summary: {args.max_items_per_size}, summaries per size: {args.summaries_per_size}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(per_item_path)
    print(f"Saved per-item time plot: {per_item_path}")

    # Write CSV
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "level",
            "topic_file",
            "evaluator",
            "model",
            "summary_model",
            "comment_num",
            "items",
            "per_item_seconds",
            "total_seconds",
            "perspective_representation",
            "informativeness",
            "neutrality_balance",
            "policy_approval",
        ])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
