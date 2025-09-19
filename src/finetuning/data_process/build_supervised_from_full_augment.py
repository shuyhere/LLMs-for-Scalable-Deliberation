#!/usr/bin/env python3
"""
Build supervised datasets from annotation_output/full_augment.

This script does NOT modify legacy scripts. It reads per-case annotated_instances.jsonl
and optionally assigned_user_data.json if present, parses HTML to extract question and
summaries when needed, normalizes label keys, and writes rating/comparison JSONL files.
"""

import argparse
import csv
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def strip_html_tags(raw_html: str) -> str:
    text = html_lib.unescape(raw_html)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_question_from_html(displayed_html: str) -> str:
    m = re.search(r"<h4[^>]*>(.*?)</h4>", displayed_html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        q = strip_html_tags(m.group(1))
        q = q.replace("[Question]", "").strip()
        return q
    return strip_html_tags(displayed_html)


def extract_summaries_from_comparison_html(displayed_html: str) -> Tuple[str, str]:
    a_match = re.search(r"<h4[^>]*>\s*Summary\s*A\s*</h4>", displayed_html, flags=re.IGNORECASE)
    b_match = re.search(r"<h4[^>]*>\s*Summary\s*B\s*</h4>", displayed_html, flags=re.IGNORECASE)
    if not a_match or not b_match:
        text_all = strip_html_tags(displayed_html)
        return text_all, ""
    a_start = a_match.end()
    b_start = b_match.end()
    a_section = displayed_html[a_start:b_match.start()]
    b_section = displayed_html[b_start:]
    sum_a = strip_html_tags(a_section)
    sum_b = strip_html_tags(b_section)
    return sum_a, sum_b


def extract_summary_from_rating_html(displayed_html: str) -> str:
    return strip_html_tags(displayed_html)


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
    la_norm = {k.strip(): v for k, v in label_annotations.items()}
    for human_q, target_key in key_map.items():
        if human_q in la_norm:
            scales = la_norm[human_q]
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
    la_norm = {k.strip(): v for k, v in label_annotations.items()}
    for human_q, target_key in key_map.items():
        if human_q in la_norm:
            scales = la_norm[human_q]
            for v in scales.values():
                try:
                    result[target_key] = int(v)
                except Exception:
                    pass
                break
    return result


def update_label_catalog(
    catalog: Dict[str, Dict[str, Dict[str, List[str]]]],
    rec: Dict[str, Any]
) -> None:
    # catalog structure:
    # {
    #   "rating": {
    #       human_question_text: { numeric_str: [unique option descriptions...] }
    #   },
    #   "comparison": {
    #       human_question_text: { numeric_str: [unique option descriptions...] }
    #   }
    # }
    la = rec.get("label_annotations", {})
    if not isinstance(la, dict):
        return
    is_comparison = rec.get("id", "").endswith("_comparison")
    section = "comparison" if is_comparison else ("rating" if rec.get("id", "").endswith("_rating") else None)
    if not section:
        return
    if section not in catalog:
        catalog[section] = {}
    for human_q, mapping in la.items():
        if not isinstance(mapping, dict):
            continue
        human_q_norm = human_q.strip()
        if human_q_norm not in catalog[section]:
            catalog[section][human_q_norm] = {}
        for desc, num in mapping.items():
            # desc is the human option description, num is numeric as string
            num_str = str(num).strip()
            arr = catalog[section][human_q_norm].setdefault(num_str, [])
            if desc not in arr:
                arr.append(desc)


def _format_rating_options(options_by_score: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    for i in range(1, 6):
        k = str(i)
        if k in options_by_score and options_by_score[k]:
            desc_list = sorted(options_by_score[k])
            desc = " | ".join(desc_list)
            lines.append(f"{i}: {desc}")
    return "\n".join(lines)


def _format_comparison_options(options_by_score: Dict[str, List[str]]) -> str:
    # Comparison uses 1-5 mapped to A/B/Both phrased descriptions
    lines: List[str] = []
    for i in range(1, 6):
        k = str(i)
        if k in options_by_score and options_by_score[k]:
            desc_list = sorted(options_by_score[k])
            desc = " | ".join(desc_list)
            lines.append(f"{i}: {desc}")
    return "\n".join(lines)


def build_rating_prompt(
    question: str,
    annotator_answer: str,
    summary_raw: str,
    label_catalog: Dict[str, Dict[str, Dict[str, List[str]]]] | None,
) -> str:
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

Please evaluate this summary on the following 4 criteria using a 1-5 scale. For each criterion, choose one of the exact options shown:

1. **To what extent is the annotator's opinion represented in this response?**
Options:
{_format_rating_options((label_catalog or {}).get('rating', {}).get('To what extent is your perspective represented in this response?', {}))}

2. **How informative is this summary?**
Options:
{_format_rating_options((label_catalog or {}).get('rating', {}).get('How informative is this summary?', {}))}

3. **Do you think this summary presents a neutral and balanced view of the issue?**
Options:
{_format_rating_options((label_catalog or {}).get('rating', {}).get('Do you think this summary presents a neutral and balanced view of the issue?', {}))}

4. **Would the annotator approve of this summary being used by the policy makers to make decisions relevant to the issue?**
Options:
{_format_rating_options((label_catalog or {}).get('rating', {}).get('Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?', {}))}

Please provide your evaluation in the following JSON format:
```json
{{
    "Representiveness": <1-5>,
    "Informativeness": <1-5>,
    "Neutrality": <1-5>,
    "Policy Approval": <1-5>
}}
```

Important: Select exactly one option for each criterion."""


def build_comparison_prompt(
    question: str,
    annotator_answer: str,
    summary_a_raw: str,
    summary_b_raw: str,
    label_catalog: Dict[str, Dict[str, Dict[str, List[str]]]] | None,
) -> str:
    perspective_text = ""
    if annotator_answer:
        perspective_text = f"""

One annotator's opinion on this question is:
{annotator_answer}
"""

    return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Two summaries of opinions are shown below. Read carefully and answer according to the opinion of the annotator.

Summary A:
{summary_a_raw}

Summary B:
{summary_b_raw}

Please compare these summaries on the following 4 criteria. For each criterion, choose one of the exact options shown:

1. **Which summary is more representative of the annotator's opinion?**
Options:
{_format_comparison_options((label_catalog or {}).get('comparison', {}).get('Which summary is more representative of your perspective?', {}))}

2. **Which summary is more informative?**
Options:
{_format_comparison_options((label_catalog or {}).get('comparison', {}).get('Which summary is more informative?', {}))}

3. **Which summary presents a more neutral and balanced view of the issue?**
Options:
{_format_comparison_options((label_catalog or {}).get('comparison', {}).get('Which summary presents a more neutral and balanced view of the issue?', {}))}

4. **Which summary would the annotator prefer of being used by the policy makers to make decisions relevant to the issue?**
Options:
{_format_comparison_options((label_catalog or {}).get('comparison', {}).get('Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?', {}))}

Please provide your evaluation in the following JSON format:
```json
{{
    "Representiveness": <1-5>,
    "Informativeness": <1-5>,
    "Neutrality": <1-5>,
    "Policy Approval": <1-5>
}}
```

Important: Select exactly one option for each criterion."""


def process_annotation_dir(
    case_dir: Path,
    rating_csv_map: Dict[str, str] | None = None,
    label_catalog: Dict[str, Dict[str, Dict[str, List[str]]]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rating_entries: List[Dict[str, Any]] = []
    comparison_entries: List[Dict[str, Any]] = []

    assigned_path = case_dir / "assigned_user_data.json"
    ann_path = case_dir / "annotated_instances.jsonl"
    if not ann_path.exists():
        return rating_entries, comparison_entries

    assigned: Dict[str, Any] = {}
    if assigned_path.exists():
        try:
            assigned = json.loads(assigned_path.read_text(encoding="utf-8"))
        except Exception:
            assigned = {}

    ann_records = read_jsonl(ann_path)
    rec_by_id: Dict[str, Dict[str, Any]] = {rec.get("id", ""): rec for rec in ann_records}

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

        q_info = assigned.get(q_id, {})
        question_text = q_info.get("question", "") or q_rec.get("question", "")
        if not question_text:
            q_html = q_rec.get("displayed_text", "")
            if q_html:
                question_text = extract_question_from_html(q_html)
        if question_text.startswith("[Question]"):
            question_text = question_text.replace("[Question]", "").strip()

        annotator_answer = get_annotator_answer(q_rec)

        if r_rec:
            rating_info = assigned.get(r_id, {})
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
            if rating_csv_map and selected_summary_id and selected_summary_id in rating_csv_map:
                summary_raw = rating_csv_map[selected_summary_id]
            if not summary_raw:
                html_text = rating_info.get("text", "") or rating_info.get("displayed_text", "") or r_rec.get("displayed_text", "")
                if html_text:
                    summary_raw = extract_summary_from_rating_html(html_text)

            prompt = build_rating_prompt(question_text, annotator_answer, summary_raw, label_catalog)
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

        if c_rec:
            comp_info = assigned.get(c_id, {})
            sum_a = comp_info.get("summary_a_text", "")
            sum_b = comp_info.get("summary_b_text", "")
            if (not sum_a or not sum_b) and c_rec.get("displayed_text"):
                parsed_a, parsed_b = extract_summaries_from_comparison_html(c_rec.get("displayed_text", ""))
                sum_a = sum_a or parsed_a
                sum_b = sum_b or parsed_b

            prompt_c = build_comparison_prompt(question_text, annotator_answer, sum_a, sum_b, label_catalog)
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
    label_catalog: Dict[str, Dict[str, Dict[str, List[str]]]] = {"rating": {}, "comparison": {}}

    case_dirs = [p for p in annotation_root.glob("*/") if p.is_dir()]
    if max_dirs is not None:
        case_dirs = case_dirs[:max_dirs]

    for case_dir in case_dirs:
        # Update label catalog first
        ann_path = case_dir / "annotated_instances.jsonl"
        if ann_path.exists():
            for rec in read_jsonl(ann_path):
                update_label_catalog(label_catalog, rec)

        r_entries, c_entries = process_annotation_dir(case_dir, rating_csv_map, label_catalog)
        if r_entries:
            rating_all.extend(r_entries)
        if c_entries:
            comparison_all.extend(c_entries)

    # Attach catalog to outputs by returning via global attributes on the function (lightweight)
    scan_all.label_catalog = label_catalog  # type: ignore[attr-defined]
    return rating_all, comparison_all


def save_jsonl(path: Path, entries: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create supervised datasets (rating/comparison) from full_augment annotations")
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment",
        help="Directory containing per-case annotated_instances.jsonl (and optionally assigned_user_data.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format_full_augment",
        help="Output directory for supervised datasets",
    )
    parser.add_argument("--max-dirs", type=int, default=None, help="Max case directories to process (for testing)")
    parser.add_argument(
        "--rating-summary-csv",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/data_files/raw/summaries_V0903_for_humanstudy_detail.csv",
        help="CSV file containing raw summaries; used to resolve rating selected summary text",
    )
    parser.add_argument(
        "--labels-report",
        type=str,
        default="",
        help="Optional path to save a JSON report of unique label options per question",
    )

    args = parser.parse_args()

    annotation_root = Path(args.annotation_dir)
    output_root = Path(args.output_dir)
    if not annotation_root.exists():
        print(f"Annotation directory {annotation_root} does not exist")
        return

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

    # Decide the canonical label catalog to render options from.
    label_catalog = getattr(scan_all, "label_catalog", {"rating": {}, "comparison": {}})
    # If a labels_report path is provided and exists, prefer loading it to ensure completeness.
    if args.labels_report:
        report_path = Path(args.labels_report)
        if report_path.exists():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "rating" in loaded and "comparison" in loaded:
                    label_catalog = loaded
            except Exception:
                pass

    # Rebuild prompts using the final label_catalog to guarantee full options are shown
    for e in rating_all:
        e["prompt"] = build_rating_prompt(
            e.get("question", ""),
            e.get("annotator_answer", ""),
            e.get("summary_raw", ""),
            label_catalog,
        )
    for e in comparison_all:
        e["prompt"] = build_comparison_prompt(
            e.get("question", ""),
            e.get("annotator_answer", ""),
            e.get("summary_a_raw", ""),
            e.get("summary_b_raw", ""),
            label_catalog,
        )

    save_jsonl(output_root / "rating_supervised.jsonl", rating_all)
    save_jsonl(output_root / "comparison_supervised.jsonl", comparison_all)

    print(f"Saved rating entries: {len(rating_all)} -> {output_root / 'rating_supervised.jsonl'}")
    print(f"Saved comparison entries: {len(comparison_all)} -> {output_root / 'comparison_supervised.jsonl'}")

    # Optionally write labels report and validate completeness
    if args.labels_report:
        report_path = Path(args.labels_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(label_catalog, ensure_ascii=False, indent=2))
        print(f"Saved labels report -> {report_path}")

    # Print quick validation to stdout
    def validate_rating_question(human_q: str, mapping: Dict[str, List[str]]):
        missing = [str(i) for i in range(1, 6) if str(i) not in mapping]
        if missing:
            print(f"[WARN] Rating question missing scores {missing}: {human_q}")
    for human_q, mapping in label_catalog.get("rating", {}).items():
        validate_rating_question(human_q, mapping)

    def validate_comparison_question(human_q: str, mapping: Dict[str, List[str]]):
        # Expect numeric values 1 or 2 present
        missing = [k for k in ["1", "2"] if k not in mapping]
        if missing:
            print(f"[WARN] Comparison question missing choices {missing}: {human_q}")
    for human_q, mapping in label_catalog.get("comparison", {}).items():
        validate_comparison_question(human_q, mapping)


if __name__ == "__main__":
    main()


