#!/usr/bin/env python3
"""
Construct a fine-tuning dataset from summary and evaluation results.

Each example:
- input: "<summary>...</summary>\n<comment>...</comment>\nHow well is the comment represented in the summary?"
- output: "1".."5"
- metadata: { dataset_name, evaluation_model, summary_model, comment_index }

The scores are sourced from evaluation files under results/summary/210 for the
following models by default: gpt-5-nano, gemini-2.5-flash-lite, web-rev-claude-3-7-sonnet-20250219.

Usage:
  python src/finetuning/data_construct.py \
    --results-dir /ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary/210 \
    --out-file /ibex/project/c2328/LLMs-Scalable-Deliberation/results/summary/210/finetuning/constructed.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_SUMMARY_MODELS = [
    "gpt-5-nano",
    "gemini-2.5-flash-lite",
    "web-rev-claude-3-7-sonnet-20250219",
]

DEFAULT_EVAL_MODELS = [
    "gpt-5-nano",
    "gemini-2.5-flash-lite",
    "web-rev-claude-3-7-sonnet-20250219",
]


def read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def build_example(summary_text: str, comment_text: str, score: int, metadata: Dict) -> Dict:
    prompt = (
        f"<summary>{summary_text}</summary>\n"
        f"<comment>{comment_text}</comment>\n"
        f"How well is the comment represented in the summary?"
    )
    return {
        "input": prompt,
        "output": str(score),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct fine-tuning dataset from evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "results" / "summary" / "210"),
        help="Directory like results/summary/210 containing model subfolders",
    )
    parser.add_argument(
        "--summary-models",
        type=str,
        default=",".join(DEFAULT_SUMMARY_MODELS),
        help="Comma-separated list of summary model names to include",
    )
    parser.add_argument(
        "--eval-models",
        type=str,
        default=",".join(DEFAULT_EVAL_MODELS),
        help="Comma-separated list of evaluation model names to include",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "results" / "summary" / "210" / "finetuning" / "constructed.jsonl"),
        help="Path to write JSONL fine-tuning dataset",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_path = Path(args.out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_models = [m.strip() for m in args.summary_models.split(",") if m.strip()]
    eval_models = [m.strip() for m in args.eval_models.split(",") if m.strip()]

    total_written = 0

    for summary_model in summary_models:
        model_dir = results_dir / summary_model
        if not model_dir.exists() or not model_dir.is_dir():
            continue

        # Iterate topics within this summary model
        for topic_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            dataset_name = topic_dir.name
            eval_file = topic_dir / f"eva_summary_{dataset_name}.json"
            sum_file = topic_dir / f"summary_{dataset_name}.json"

            eval_data = read_json(eval_file)
            sum_data = read_json(sum_file)
            if eval_data is None or sum_data is None:
                continue

            # Summary text (we use main_points)
            summary_text = sum_data.get("summaries", {}).get("main_points")
            if not summary_text:
                # Fallback to topic_modeling or custom_analysis if needed
                summary_text = sum_data.get("summaries", {}).get("topic_modeling") or sum_data.get("summaries", {}).get("custom_analysis") or ""

            evaluations = eval_data.get("evaluations", {})
            for eval_model, payload in evaluations.items():
                if eval_model not in eval_models:
                    continue
                results = payload.get("evaluation_data", {}).get("evaluation_results", [])
                for item in results:
                    score_val = item.get("score", item.get("extracted_score"))
                    comment_text = item.get("comment")
                    comment_index = item.get("comment_index")
                    if score_val is None or comment_text is None or comment_index is None:
                        continue
                    try:
                        score_int = int(float(score_val))
                    except Exception:
                        # Best-effort parse
                        import re
                        m = re.findall(r"[-+]?[0-9]+", str(score_val))
                        if not m:
                            continue
                        score_int = int(m[-1])
                    if score_int < 1 or score_int > 5:
                        continue

                    meta = {
                        "dataset_name": dataset_name,
                        "evaluation_model": eval_model,
                        "summary_model": summary_model,
                        "comment_index": comment_index,
                    }
                    example = build_example(summary_text=summary_text, comment_text=comment_text, score=score_int, metadata=meta)

                    with out_path.open("a", encoding="utf-8") as w:
                        w.write(json.dumps(example, ensure_ascii=False) + "\n")
                    total_written += 1

    print(f"Wrote {total_written} examples to {out_path}")


if __name__ == "__main__":
    main()
