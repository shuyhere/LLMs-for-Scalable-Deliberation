#!/usr/bin/env python3
"""
Create RL datasets (chosen/rejected) from full_augment annotations for reward model training.

Rules:
- Keep strong and slight preference pairs; drop only "both about the same".
- If the choice indicates "much more" (far more), set sample weight = 2; if "slightly", weight = 1.

Choice mapping (from human labels 1..5):
  1 = A is much better      -> chosen=A, rejected=B, weight=2
  2 = A is slightly better  -> chosen=A, rejected=B, weight=1
  3 = Both about the same   -> DROP
  4 = B is slightly better  -> chosen=B, rejected=A, weight=1
  5 = B is much better      -> chosen=B, rejected=A, weight=2

Dimensions handled: perspective, informativeness, neutrality, policy
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DIMENSIONS = {
    "perspective": "Which summary is more representative of your perspective?",
    "informativeness": "Which summary is more informative?",
    "neutrality": "Which summary presents a more neutral and balanced view of the issue?",
    "policy": "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?",
}

QUESTION_TEMPLATES = {
    "perspective": "Which summary is more representative of the annotator's opinion?",
    "informativeness": "Which summary is more informative?",
    "neutrality": "Which summary presents a more neutral and balanced view of the issue?",
    "policy": "Which summary would the annotator prefer of being used by the policy makers to make decisions relevant to the issue?",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def extract_answer(q_rec: Dict[str, Any]) -> str:
    anns = q_rec.get("label_annotations", {})
    if "answer" in anns and isinstance(anns["answer"], dict):
        return anns["answer"].get("text_box", "")
    return ""


def process_case_dir(case_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    assigned_path = case_dir / "assigned_user_data.json"
    ann_path = case_dir / "annotated_instances.jsonl"
    if not ann_path.exists() or not assigned_path.exists():
        return out

    try:
        assigned = json.loads(assigned_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    ann = read_jsonl(ann_path)
    by_id: Dict[str, Dict[str, Any]] = {r.get("id", ""): r for r in ann}

    # discover triplets
    tids: List[str] = []
    for rid in by_id.keys():
        if rid.startswith("triplet_") and (rid.endswith("_comparison") or rid.endswith("_question")):
            t = rid.split("_")[1]
            if t not in tids:
                tids.append(t)

    for t in tids:
        q_id = f"triplet_{t}_question"
        c_id = f"triplet_{t}_comparison"
        q_rec = by_id.get(q_id)
        c_rec = by_id.get(c_id)
        if not q_rec or not c_rec:
            continue

        # question text and annotator answer
        q_info = assigned.get(q_id, {})
        question = q_info.get("question", "") or q_rec.get("question", "")
        if question.startswith("[Question]"):
            question = question.replace("[Question]", "").strip()
        annotator_answer = extract_answer(q_rec)

        # summaries from assigned
        comp_info = assigned.get(c_id, {})
        sum_a = comp_info.get("summary_a_text", "")
        sum_b = comp_info.get("summary_b_text", "")
        model_a = comp_info.get("model_a", "")
        model_b = comp_info.get("model_b", "")
        a_id = comp_info.get("summary_a_id", "")
        b_id = comp_info.get("summary_b_id", "")
        if not sum_a or not sum_b:
            continue

        la = c_rec.get("label_annotations", {})
        la = {k.strip(): v for k, v in la.items()}  # normalize keys

        for dim, human_q in DIMENSIONS.items():
            if human_q not in la or not isinstance(la[human_q], dict):
                continue
            # extract single value (e.g., {"A is slightly more informative": "2"})
            choice = None
            for v in la[human_q].values():
                choice = v
                break
            if choice is None:
                continue
            try:
                ch = int(choice)
            except Exception:
                continue

            # map preference & weight
            if ch == 3:
                continue  # drop equal
            if ch == 1:
                chosen, rejected, weight = sum_a, sum_b, 2
            elif ch == 2:
                chosen, rejected, weight = sum_a, sum_b, 1
            elif ch == 4:
                chosen, rejected, weight = sum_b, sum_a, 1
            elif ch == 5:
                chosen, rejected, weight = sum_b, sum_a, 2
            else:
                continue

            prompt = f"We have made a deliberation with many annotators on the issue: {question}\n\n"
            if annotator_answer:
                prompt += f"One annotator's opinion on this question is:\n{annotator_answer}\n\n"
            prompt += QUESTION_TEMPLATES[dim]

            out.append({
                "id": f"{c_id}_{dim}",
                "prompt": prompt,
                "question": question,
                "annotator_answer": annotator_answer,
                "chosen": chosen,
                "rejected": rejected,
                "dimension": dim,
                "dimension_question": QUESTION_TEMPLATES[dim],
                "model_a": model_a,
                "model_b": model_b,
                "summary_a_id": a_id,
                "summary_b_id": b_id,
                "weight": weight,
            })

    return out


def scan_all(root: Path, max_dirs: int | None = None) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    case_dirs = [p for p in root.glob("*/") if p.is_dir()]
    if max_dirs is not None:
        case_dirs = case_dirs[:max_dirs]
    for d in case_dirs:
        rows = process_case_dir(d)
        if rows:
            all_rows.extend(rows)
    return all_rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create RL chosen/rejected dataset from full_augment annotations")
    parser.add_argument("--annotation-dir", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment")
    parser.add_argument("--output-dir", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/rl_datasets/full_augment_by_dim",
                        help="Directory to save per-dimension RL datasets")
    parser.add_argument("--max-dirs", type=int, default=None)

    args = parser.parse_args()

    root = Path(args.annotation_dir)
    if not root.exists():
        print(f"Annotation directory {root} does not exist")
        return

    rows = scan_all(root, args.max_dirs)
    # split by dimension
    by_dim: Dict[str, List[Dict[str, Any]]] = {k: [] for k in DIMENSIONS.keys()}
    for r in rows:
        dim = r.get("dimension", "")
        if dim in by_dim:
            by_dim[dim].append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for dim, items in by_dim.items():
        out_path = out_dir / f"{dim}_rl.jsonl"
        save_jsonl(out_path, items)
        print(f"Saved {len(items)} pairs for {dim} -> {out_path}")
        total += len(items)
    print(f"Saved total {total} pairs -> {out_dir}")


if __name__ == "__main__":
    main()


