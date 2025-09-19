#!/usr/bin/env python3
"""
Create supervised datasets for training two annotation models:
- Rating model: single-summary input using get_rating_prompt; labels are 4 ratings (1-5)
- Comparison model: A/B summaries using get_comparison_prompt; labels are 4 choices (1 or 2)

Key requirements:
- Do NOT use cleaned summaries; use the original summary texts from assigned_user_data.json
- Build prompts that match src/utils/prompts/evaluation.py HumanAnnotationPrompt methods
- Read data similar to create_rl_datasets.py patterns
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

# We intentionally avoid importing prompt classes to prevent unintended cleaning; we reconstruct templates.


RATING_QUESTIONS = [
    ("perspective_representation", "To what extent is the annotator's opinion represented in this response?"),
    ("informativeness", "How informative is this summary?"),
    ("neutrality_balance", "Do you think this summary presents a neutral and balanced view of the issue?"),
    ("policy_approval", "Would the annotator approve of this summary being used by the policy makers to make decisions relevant to the issue?")
]

COMPARISON_QUESTIONS = [
    ("perspective_representation", "Which summary is more representative of the annotator's opinion?"),
    ("informativeness", "Which summary is more informative?"),
    ("neutrality_balance", "Which summary presents a more neutral and balanced view of the issue?"),
    ("policy_approval", "Which summary would the annotator prefer of being used by the policy makers to make decisions relevant to the issue?")
]


def build_rating_prompt(question: str, annotator_answer: str, summary_raw: str) -> str:
    perspective_text = ""
    if annotator_answer:
        perspective_text = f"""

One annotator's opinion on this question is:
{annotator_answer}
"""

    return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Below is a summary of all people's opinions on the issue.

{summary_raw}

Please evaluate this summary on the following 4 criteria using a 1-5 scale:

1. **To what extent is the annotator's opinion represented in this response?**
   (1 = Not at all, 2 = Slightly, 3 = Moderately, 4 = Well, 5 = Very well)

2. **How informative is this summary?**
   (1 = Not informative, 2 = Slightly informative, 3 = Moderately informative, 4 = Very informative, 5 = Extremely informative)

3. **Do you think this summary presents a neutral and balanced view of the issue?**
   (1 = Very biased, 2 = Somewhat biased, 3 = Neutral, 4 = Fairly balanced, 5 = Very balanced)

4. **Would the annotator approve of this summary being used by the policy makers to make decisions relevant to the issue?**
   (1 = Strongly disapprove, 2 = Disapprove, 3 = Neutral, 4 = Approve, 5 = Strongly approve)

Please provide your evaluation in the following JSON format:
```json
{{
    "perspective_representation": <1-5>,
    "informativeness": <1-5>,
    "neutrality_balance": <1-5>,
    "policy_approval": <1-5>
}}
```

Important: Respond ONLY with the JSON object, no additional text."""


def build_comparison_prompt(question: str, annotator_answer: str, summary_a_raw: str, summary_b_raw: str) -> str:
    perspective_text = ""
    if annotator_answer:
        perspective_text = f"""

One annotator's opinion on this question is:
{annotator_answer}
"""

    return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Two summaries of all people's opinions are shown below. Read carefully and answer according to your prior opinion.

Summary A:
{summary_a_raw}

Summary B:
{summary_b_raw}

Please compare these summaries on the following 4 criteria. For each criterion, choose which summary is better (1 for Summary A, 2 for Summary B):

1. **Which summary is more representative of the annotator's opinion?**
2. **Which summary is more informative?**
3. **Which summary presents a more neutral and balanced view of the issue?**
4. **Which summary would the annotator prefer of being used by the policy makers to make decisions relevant to the issue?**

Please provide your evaluation in the following JSON format:
```json
{{
    "perspective_representation": <1 or 2>,
    "informativeness": <1 or 2>,
    "neutrality_balance": <1 or 2>,
    "policy_approval": <1 or 2>
}}
```

Important: 
- Use 1 for Summary A, 2 for Summary B
- Respond ONLY with the JSON object, no additional text."""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_annotator_answer(question_ann: Dict[str, Any]) -> str:
    anns = question_ann.get("label_annotations", {})
    if "answer" in anns and isinstance(anns["answer"], dict):
        return anns["answer"].get("text_box", "")
    return ""


def extract_rating_labels(label_annotations: Dict[str, Any]) -> Dict[str, int]:
    key_map = {
        "To what extent is your perspective represented in this response?": "perspective_representation",
        "How informative is this summary?": "informativeness",
        "Do you think this summary presents a neutral and balanced view of the issue?": "neutrality_balance",
        "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?": "policy_approval",
    }
    result: Dict[str, int] = {}
    for human_q, target_key in key_map.items():
        if human_q in label_annotations:
            scales = label_annotations[human_q]
            # scales may be like {"scale_5": "5"} or {"scale_4": "4"}
            for v in scales.values():
                try:
                    result[target_key] = int(v)
                except Exception:
                    pass
                break
    return result


def extract_comparison_labels(label_annotations: Dict[str, Any]) -> Dict[str, int]:
    key_map = {
        "Which summary is more representative of your perspective?": "perspective_representation",
        "Which summary is more informative?": "informativeness",
        "Which summary presents a more neutral and balanced view of the issue?": "neutrality_balance",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?": "policy_approval",
    }
    result: Dict[str, int] = {}
    for human_q, target_key in key_map.items():
        if human_q in label_annotations:
            scales = label_annotations[human_q]
            # scales may be like {"scale_1": "1"} or {"scale_2": "2"}
            for v in scales.values():
                try:
                    result[target_key] = int(v)
                except Exception:
                    pass
                break
    return result


def process_annotation_dir(case_dir: Path, rating_csv_map: Dict[str, str] | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rating_entries: List[Dict[str, Any]] = []
    comparison_entries: List[Dict[str, Any]] = []

    assigned_path = case_dir / "assigned_user_data.json"
    ann_path = case_dir / "annotated_instances.jsonl"
    if not assigned_path.exists() or not ann_path.exists():
        return rating_entries, comparison_entries

    try:
        assigned = json.loads(assigned_path.read_text(encoding="utf-8"))
    except Exception:
        assigned = {}

    ann_records = read_jsonl(ann_path)

    # Index by id
    rec_by_id: Dict[str, Dict[str, Any]] = {rec.get("id", ""): rec for rec in ann_records}

    # Discover triplet ids present
    triplet_ids: List[str] = []
    for rec_id in rec_by_id.keys():
        if rec_id.startswith("triplet_") and (rec_id.endswith("_rating") or rec_id.endswith("_comparison") or rec_id.endswith("_question")):
            tid = rec_id.split("_")[1]
            if tid not in triplet_ids:
                triplet_ids.append(tid)

    for tid in triplet_ids:
        q_id = f"triplet_{tid}_question"
        r_id = f"triplet_{tid}_rating"
        c_id = f"triplet_{tid}_comparison"

        q_rec = rec_by_id.get(q_id, {})
        r_rec = rec_by_id.get(r_id, {})
        c_rec = rec_by_id.get(c_id, {})

        # Question text from assigned file if available
        q_info = assigned.get(q_id, {})
        question_text = q_info.get("question", "") or q_rec.get("question", "")
        if question_text.startswith("[Question]"):
            question_text = question_text.replace("[Question]", "").strip()

        annotator_answer = get_annotator_answer(q_rec)

        # Rating entry
        if r_rec:
            rating_info = assigned.get(r_id, {})
            # Prefer pulling rating summary from CSV using the corresponding summary_*_id selected by the rating model
            summary_raw = ""
            selected_summary_id = None
            rating_model = rating_info.get("model", "")
            comp_info_for_rating = assigned.get(c_id, {})
            model_a = comp_info_for_rating.get("model_a", "")
            model_b = comp_info_for_rating.get("model_b", "")
            if rating_model and comp_info_for_rating:
                if rating_model == model_a:
                    selected_summary_id = comp_info_for_rating.get("summary_a_id", "")
                elif rating_model == model_b:
                    selected_summary_id = comp_info_for_rating.get("summary_b_id", "")
            # Lookup in CSV map
            if rating_csv_map and selected_summary_id and selected_summary_id in rating_csv_map:
                summary_raw = rating_csv_map[selected_summary_id]
            # Fallbacks: use assigned raw HTML text or displayed text if CSV not found
            if not summary_raw:
                summary_raw = rating_info.get("text", "") or rating_info.get("displayed_text", "") or r_rec.get("displayed_text", "")

            prompt = build_rating_prompt(question_text, annotator_answer, summary_raw)
            labels = extract_rating_labels(r_rec.get("label_annotations", {}))
            if labels:
                rating_entries.append({
                    "id": r_id,
                    "prompt": prompt,
                    "question": question_text,
                    "annotator_answer": annotator_answer,
                    "summary_raw": summary_raw,
                    "labels": labels,
                })

        # Comparison entry
        if c_rec:
            comp_info = assigned.get(c_id, {})
            sum_a = comp_info.get("summary_a_text", "")
            sum_b = comp_info.get("summary_b_text", "")
            # As strict fallback, use displayed_html from rec, but we prefer raw texts from assigned file
            if not sum_a or not sum_b:
                sum_a = sum_a or ""
                sum_b = sum_b or ""

            prompt_c = build_comparison_prompt(question_text, annotator_answer, sum_a, sum_b)
            labels_c = extract_comparison_labels(c_rec.get("label_annotations", {}))
            if labels_c:
                comparison_entries.append({
                    "id": c_id,
                    "prompt": prompt_c,
                    "question": question_text,
                    "annotator_answer": annotator_answer,
                    "summary_a_raw": sum_a,
                    "summary_b_raw": sum_b,
                    "labels": labels_c,
                    "summary_a_id": comp_info.get("summary_a_id", ""),
                    "summary_b_id": comp_info.get("summary_b_id", ""),
                    "model_a": comp_info.get("model_a", ""),
                    "model_b": comp_info.get("model_b", ""),
                })

    return rating_entries, comparison_entries


def scan_all(annotation_root: Path, rating_csv_map: Dict[str, str] | None, max_dirs: int | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rating_all: List[Dict[str, Any]] = []
    comparison_all: List[Dict[str, Any]] = []

    case_dirs = [p for p in annotation_root.glob("*/") if p.is_dir()]
    if max_dirs is not None:
        case_dirs = case_dirs[:max_dirs]

    for case_dir in case_dirs:
        r_entries, c_entries = process_annotation_dir(case_dir, rating_csv_map)
        if r_entries:
            rating_all.extend(r_entries)
        if c_entries:
            comparison_all.extend(c_entries)

    return rating_all, comparison_all


def save_jsonl(path: Path, entries: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create supervised datasets for rating and comparison annotation models")
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full",
        help="Directory containing per-case assigned_user_data.json and annotated_instances.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format",
        help="Output directory for supervised datasets",
    )
    parser.add_argument("--max-dirs", type=int, default=None, help="Max case directories to process (for testing)")
    parser.add_argument(
        "--rating-summary-csv",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_detail.csv",
        help="CSV file containing raw summaries; will be used for rating summaries",
    )

    args = parser.parse_args()

    annotation_root = Path(args.annotation_dir)
    output_root = Path(args.output_dir)
    if not annotation_root.exists():
        print(f"Annotation directory {annotation_root} does not exist")
        return

    # Build map from summary id -> summary text from CSV
    rating_csv_map: Dict[str, str] = {}
    csv_path = Path(args.rating_summary_csv)
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("id", "").strip()
                text = row.get("summary", "")
                if sid:
                    rating_csv_map[sid] = text

    rating_all, comparison_all = scan_all(annotation_root, rating_csv_map, args.max_dirs)

    # Save
    save_jsonl(output_root / "rating_supervised.jsonl", rating_all)
    save_jsonl(output_root / "comparison_supervised.jsonl", comparison_all)

    print(f"Saved rating entries: {len(rating_all)} -> {output_root / 'rating_supervised.jsonl'}")
    print(f"Saved comparison entries: {len(comparison_all)} -> {output_root / 'comparison_supervised.jsonl'}")


if __name__ == "__main__":
    main()


