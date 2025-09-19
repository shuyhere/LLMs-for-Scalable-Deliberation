#!/usr/bin/env python3
"""
Construct per-summary rating data in the format: (question, comment, summary, scores)

Data source: annotation/summary-rating/annotation_output/full_augment

For each triplet, we generate TWO rows:
- Row 1: The summary that was rated in the single-summary rating task, using the given scores.
- Row 2: The other summary, whose scores are derived by combining Row 1 scores with the comparison choices.

Comparison mapping policy (per criterion):
- If A is better than B: B = A - 1 (slightly) or B = A - 2 (much better)
- If B is better than A: B = A + 1 (slightly) or B = A + 2 (much better)
- If both about the same: B = A
Implementation detail: we map choices to deltas with respect to (A - B):
  choice 1 -> +2, 2 -> +1, 3 -> 0, 4 -> -1, 5 -> -2.
When the rated summary is B (not A), we invert the delta sign to maintain the same rule.
Note: We do NOT clamp derived scores; they may go below 1 or above 5 per your spec.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


RATING_Q_TEXTS = {
    "perspective_representation": "To what extent is your perspective represented in this response?",
    "informativeness": "How informative is this summary?",
    "neutrality_balance": "Do you think this summary presents a neutral and balanced view of the issue?",
    "policy_approval": "Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?",
}

COMPARISON_Q_TEXTS = {
    "perspective_representation": "Which summary is more representative of your perspective?",
    "informativeness": "Which summary is more informative?",
    "neutrality_balance": "Which summary presents a more neutral and balanced view of the issue?",
    "policy_approval": "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
}


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


def extract_annotator_answer(q_rec: Dict[str, Any]) -> str:
    anns = q_rec.get("label_annotations", {})
    if "answer" in anns and isinstance(anns["answer"], dict):
        return anns["answer"].get("text_box", "")
    return ""


def extract_rating_labels(label_annotations: Dict[str, Any]) -> Dict[str, int]:
    key_map = {
        RATING_Q_TEXTS["perspective_representation"]: "perspective_representation",
        RATING_Q_TEXTS["informativeness"]: "informativeness",
        RATING_Q_TEXTS["neutrality_balance"]: "neutrality_balance",
        RATING_Q_TEXTS["policy_approval"]: "policy_approval",
    }
    result: Dict[str, int] = {}
    la_norm = {k.strip(): v for k, v in label_annotations.items()}
    for human_q, key in key_map.items():
        if human_q in la_norm and isinstance(la_norm[human_q], dict):
            for v in la_norm[human_q].values():
                try:
                    result[key] = int(v)
                except Exception:
                    pass
                break
    return result


def extract_comparison_labels(label_annotations: Dict[str, Any]) -> Dict[str, int]:
    key_map = {
        COMPARISON_Q_TEXTS["perspective_representation"]: "perspective_representation",
        COMPARISON_Q_TEXTS["informativeness"]: "informativeness",
        COMPARISON_Q_TEXTS["neutrality_balance"]: "neutrality_balance",
        COMPARISON_Q_TEXTS["policy_approval"]: "policy_approval",
    }
    result: Dict[str, int] = {}
    la_norm = {k.strip(): v for k, v in label_annotations.items()}
    for human_q, key in key_map.items():
        if human_q in la_norm and isinstance(la_norm[human_q], dict):
            for v in la_norm[human_q].values():
                try:
                    result[key] = int(v)
                except Exception:
                    pass
                break
    return result


def derive_other_scores(
    base_scores: Dict[str, int],
    comp_labels: Dict[str, int],
    base_is_a: bool,
) -> Dict[str, int]:
    # Delatas when perspective is A vs B
    delta_map = {1: 2, 2: 1, 3: 0, 4: -1, 5: -2}
    derived: Dict[str, int] = {}
    for key, base_val in base_scores.items():
        choice = comp_labels.get(key, 3)
        delta = delta_map.get(choice, 0)
        # If base refers to B, invert the direction
        if not base_is_a:
            delta = -delta
        val = base_val - delta  # other = base - delta (since delta is A - B); can go beyond [1,5]
        derived[key] = val
    return derived


def process_case_dir(case_dir: Path) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    assigned_path = case_dir / "assigned_user_data.json"
    ann_path = case_dir / "annotated_instances.jsonl"
    if not ann_path.exists():
        return out_rows

    assigned = {}
    if assigned_path.exists():
        try:
            assigned = json.loads(assigned_path.read_text(encoding="utf-8"))
        except Exception:
            assigned = {}

    ann = read_jsonl(ann_path)
    rec_by_id: Dict[str, Dict[str, Any]] = {r.get("id", ""): r for r in ann}

    # discover triplets
    triplet_ids: List[str] = []
    for rid in rec_by_id.keys():
        if rid.startswith("triplet_") and (rid.endswith("_rating") or rid.endswith("_comparison") or rid.endswith("_question")):
            t = rid.split("_")[1]
            if t not in triplet_ids:
                triplet_ids.append(t)

    for tid in triplet_ids:
        q_id = f"triplet_{tid}_question"
        r_id = f"triplet_{tid}_rating"
        c_id = f"triplet_{tid}_comparison"

        q_rec = rec_by_id.get(q_id, {})
        r_rec = rec_by_id.get(r_id, {})
        c_rec = rec_by_id.get(c_id, {})
        if not r_rec or not c_rec:
            continue

        # Question and annotator comment
        q_info = assigned.get(q_id, {})
        question = q_info.get("question", "") or q_rec.get("question", "")
        if question.startswith("[Question]"):
            question = question.replace("[Question]", "").strip()
        comment = extract_annotator_answer(q_rec)

        # Summaries and model alignment
        comp_info = assigned.get(c_id, {})
        sum_a = comp_info.get("summary_a_text", "")
        sum_b = comp_info.get("summary_b_text", "")
        model_a = comp_info.get("model_a", "")
        model_b = comp_info.get("model_b", "")
        # Must have both summaries
        if not sum_a or not sum_b:
            continue

        rating_info = assigned.get(r_id, {})
        rated_model = rating_info.get("model", "")
        rated_is_a = None
        if rated_model and model_a and model_b:
            if rated_model == model_a:
                rated_is_a = True
            elif rated_model == model_b:
                rated_is_a = False
        # Fallback: infer via selected summary id if available
        if rated_is_a is None:
            selected_summary_id = None
            if isinstance(comp_info, dict):
                a_id = comp_info.get("summary_a_id", "")
                b_id = comp_info.get("summary_b_id", "")
                # Try to find which summary id was rated
                selected_summary_id = rating_info.get("selected_summary_id", "") or rating_info.get("summary_id", "")
                # Some pipelines store the chosen id under rating_info->text_id
                selected_summary_id = selected_summary_id or rating_info.get("text_id", "")
                if selected_summary_id:
                    if a_id and selected_summary_id == a_id:
                        rated_is_a = True
                    elif b_id and selected_summary_id == b_id:
                        rated_is_a = False

        # Extract labels
        rating_scores = extract_rating_labels(r_rec.get("label_annotations", {}))
        comp_labels = extract_comparison_labels(c_rec.get("label_annotations", {}))
        if not rating_scores or not comp_labels:
            continue

        # If we cannot determine which summary was rated, skip
        if rated_is_a is None:
            continue

        # Build row for rated summary
        rated_summary = sum_a if rated_is_a else sum_b
        out_rows.append({
            "id": r_id + ("_A" if rated_is_a else "_B"),
            "question": question,
            "comment": comment,
            "summary": rated_summary,
            "scores": rating_scores,
            "which": "A" if rated_is_a else "B",
            "user_dir": case_dir.name,
        })

        # Derive other summary scores
        other_scores = derive_other_scores(rating_scores, comp_labels, base_is_a=rated_is_a)
        other_summary = sum_b if rated_is_a else sum_a
        out_rows.append({
            "id": r_id + ("_B" if rated_is_a else "_A"),
            "question": question,
            "comment": comment,
            "summary": other_summary,
            "scores": other_scores,
            "which": "B" if rated_is_a else "A",
            "user_dir": case_dir.name,
        })

    return out_rows


def scan_all(annotation_root: Path, max_dirs: int | None = None) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    case_dirs = [p for p in annotation_root.glob("*/") if p.is_dir()]
    if max_dirs is not None:
        case_dirs = case_dirs[:max_dirs]
    for d in case_dirs:
        rows = process_case_dir(d)
        if rows:
            all_rows.extend(rows)
    return all_rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build comment-summary rating dataset from full_augment annotations")
    parser.add_argument("--annotation-dir", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment",
                        help="Directory of case folders with assigned_user_data.json and annotated_instances.jsonl")
    parser.add_argument("--output", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/comment_summary_ratings.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--max-dirs", type=int, default=None, help="Limit number of case dirs (debug)")

    args = parser.parse_args()

    root = Path(args.annotation_dir)
    if not root.exists():
        print(f"Annotation directory {root} does not exist")
        return

    rows = scan_all(root, args.max_dirs)
    save_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()


