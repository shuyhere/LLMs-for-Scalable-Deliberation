#!/usr/bin/env python3
"""
Convert supervised annotation datasets (rating/comparison) into Alpaca format.

Input JSONL entries expected fields:
- id, prompt, labels (a JSON object of gold labels)

Output Alpaca JSONL:
{ "instruction": <prompt>, "input": "", "output": <labels as JSON string> }
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def convert_jsonl_to_alpaca(src_path: Path, dst_path: Path):
    count = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = rec.get("prompt", "")
            labels_obj = rec.get("labels")
            if labels_obj is None:
                # Some datasets may store target under a different key
                labels_obj = rec.get("output")
            # Ensure output is a compact JSON string
            output_str = json.dumps(labels_obj, ensure_ascii=False)

            alpaca = {
                "instruction": prompt,
                "input": "",
                "output": output_str,
            }
            fout.write(json.dumps(alpaca, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} Alpaca entries to {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert supervised datasets to Alpaca format")
    parser.add_argument(
        "--rating-src",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format/rating_supervised.jsonl",
        help="Source rating supervised JSONL",
    )
    parser.add_argument(
        "--comparison-src",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format/comparison_supervised.jsonl",
        help="Source comparison supervised JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/alpaca",
        help="Directory to write Alpaca JSONL files",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    rating_src = Path(args.rating_src)
    comparison_src = Path(args.comparison_src)

    convert_jsonl_to_alpaca(rating_src, output_dir / "rating_alpaca.jsonl")
    convert_jsonl_to_alpaca(comparison_src, output_dir / "comparison_alpaca.jsonl")


if __name__ == "__main__":
    main()


