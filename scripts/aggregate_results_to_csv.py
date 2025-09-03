#!/usr/bin/env python3
"""
Aggregate summarization JSON results into CSVs (simple and detail) with stats.

Input directory structure (default: results/human_judgement):
  results/human_judgement/<model>/<num_samples>/<dataset_name>_summary_<sample_id>.json

Outputs (default: results/aggregated/<timestamp>/):
  - summaries_simple.csv   # comments column includes only numbered indices
  - summaries_detail.csv   # comments column includes numbered full text
  - stats.json             # basic aggregation statistics
"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Record:
    unique_id: str
    topic: str
    question: str
    summary: str
    model: str
    comment_count: int
    comments_numbered: str  # e.g., "1,2,3,4" (simple) OR "1: text | 2: text" (detail)
    num_samples_group: int  # directory level (e.g., 10, 30, ...)
    sample_id: Optional[int]  # from filename suffix _N
    dataset_name: str  # without suffixes
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate summarization results to CSV")
    parser.add_argument(
        "--input-dir",
        default=str(Path("results") / "human_judgement"),
        help="Root directory containing model/num_samples result folders",
    )
    parser.add_argument(
        "--datasets-dir",
        default=str(Path("datasets") / "cleaned_new_dataset"),
        help="Directory containing source dataset JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("results") / "aggregated" / datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Output directory to write CSVs and stats",
    )
    parser.add_argument(
        "--delimiter",
        default=" | ",
        help="Delimiter for joining detailed comments (detail CSV)",
    )
    return parser.parse_args()


def safe_read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_filename_parts(json_path: Path) -> Tuple[str, Optional[int]]:
    """
    Extract dataset_name and sample_id from filename like:
      <dataset>_summary_<sample_id>.json OR <dataset>_summary.json
    """
    stem = json_path.stem
    if stem.endswith("_summary"):
        return stem[:-len("_summary")], None
    if "_summary_" in stem:
        base, sid = stem.split("_summary_", 1)
        try:
            return base, int(sid)
        except ValueError:
            return base, None
    # Fallback: treat the whole stem as dataset name
    return stem, None


def make_unique_id(model: str, num_samples: int, sample_id: Optional[int], dataset_name: str, source_path: str) -> str:
    base = f"{model}|{num_samples}|{sample_id or 1}|{dataset_name}|{source_path}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def load_dataset_comments(datasets_dir: Path, dataset_name: str) -> List[str]:
    """Load comments text from the original dataset JSON by dataset name."""
    data_path = datasets_dir / f"{dataset_name}.json"
    data = safe_read_json(data_path)
    if not data:
        return []
    comments: List[str] = []
    for obj in data.get("comments", []):
        if isinstance(obj, dict):
            text = obj.get("comment", "")
        else:
            text = str(obj)
        text = text.strip()
        if text:
            comments.append(text)
    return comments


def collect_records(input_dir: Path, delimiter: str, datasets_dir: Path) -> List[Record]:
    records: List[Record] = []
    for model_dir in sorted(input_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for ns_dir in sorted(model_dir.iterdir()):
            if not ns_dir.is_dir():
                continue
            try:
                num_samples_group = int(ns_dir.name)
            except ValueError:
                continue
            for json_path in sorted(ns_dir.glob("*_summary*.json")):
                data = safe_read_json(json_path)
                if not data:
                    continue
                # Extract identifiers from filename and within JSON
                dataset_from_file, sample_id = extract_filename_parts(json_path)
                metadata = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}
                summaries = data.get("summaries", {}) if isinstance(data.get("summaries", {}), dict) else {}

                dataset_name = metadata.get("dataset_name") or dataset_from_file
                question = (metadata.get("question") or "").strip()
                # Use main_points field for summary as requested
                summary_val = summaries.get("main_points", "")
                if isinstance(summary_val, str):
                    summary_val = summary_val.strip()
                else:
                    summary_val = str(summary_val)

                model_val = metadata.get("summary_model") or model
                num_samples_val = int(metadata.get("num_samples") or num_samples_group)

                # Build comments via indices from the original dataset
                comment_indices = data.get("comment_indices", [])
                if not isinstance(comment_indices, list):
                    comment_indices = []
                original_comments = load_dataset_comments(datasets_dir, dataset_name)
                indexed: List[Tuple[int, str]] = []
                for idx in comment_indices:
                    try:
                        indexed.append((int(idx), original_comments[int(idx)]))
                    except Exception:
                        continue
                comment_count = len(indexed)
                comments_simple = ",".join(str(i) for (i, _) in indexed) if indexed else ""
                comments_detail = delimiter.join(f"{i}: {t}" for (i, t) in indexed) if indexed else ""

                unique_id = make_unique_id(model_val, num_samples_val, sample_id, dataset_name, str(json_path))

                record_simple = Record(
                    unique_id=unique_id,
                    topic=dataset_name,
                    question=question,
                    summary=summary_val,
                    model=model_val,
                    comment_count=comment_count,
                    comments_numbered=comments_simple,
                    num_samples_group=num_samples_group,
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    source_path=str(json_path),
                )

                record_detail = Record(
                    unique_id=unique_id,
                    topic=dataset_name,
                    question=question,
                    summary=summary_val,
                    model=model_val,
                    comment_count=comment_count,
                    comments_numbered=comments_detail,
                    num_samples_group=num_samples_group,
                    sample_id=sample_id,
                    dataset_name=dataset_name,
                    source_path=str(json_path),
                )

                records.append((record_simple, record_detail))
    # Flatten to parallel lists to preserve pairing order
    flat_records: List[Record] = []
    for pair in records:
        flat_records.extend(list(pair))
    return flat_records


def split_simple_detail(records: List[Record]) -> Tuple[List[Record], List[Record]]:
    simple: List[Record] = []
    detail: List[Record] = []
    seen: set = set()
    # Records come in pairs (simple, detail) back-to-back
    for i, rec in enumerate(records):
        key = (rec.unique_id, rec.source_path)
        if key in seen:
            detail.append(rec)
        else:
            simple.append(rec)
            seen.add(key)
    return simple, detail


def write_csv(path: Path, rows: List[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "id",
        "topic",
        "question",
        "summary",
        "model",
        "comment_num",
        "comments",
        "num_samples_group",
        "sample_id",
        "dataset_name",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow([
                r.unique_id,
                r.topic,
                r.question,
                r.summary,
                r.model,
                r.comment_count,
                r.comments_numbered,
                r.num_samples_group,
                r.sample_id if r.sample_id is not None else "",
                r.dataset_name,
                r.source_path,
            ])


def compute_stats(simple_rows: List[Record]) -> Dict:
    total = len(simple_rows)
    by_model: Dict[str, int] = {}
    by_dataset: Dict[str, int] = {}
    by_num_samples: Dict[int, int] = {}
    for r in simple_rows:
        by_model[r.model] = by_model.get(r.model, 0) + 1
        by_dataset[r.dataset_name] = by_dataset.get(r.dataset_name, 0) + 1
        by_num_samples[r.num_samples_group] = by_num_samples.get(r.num_samples_group, 0) + 1
    return {
        "total_rows": total,
        "by_model": by_model,
        "by_dataset": by_dataset,
        "by_num_samples": by_num_samples,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    datasets_dir = Path(args.datasets_dir)

    all_records = collect_records(input_dir, args.delimiter, datasets_dir)
    simple_rows, detail_rows = split_simple_detail(all_records)

    write_csv(output_dir / "summaries_simple.csv", simple_rows)
    write_csv(output_dir / "summaries_detail.csv", detail_rows)

    stats = compute_stats(simple_rows)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(simple_rows)} simple rows and {len(detail_rows)} detail rows to {output_dir}")
    print("Stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


