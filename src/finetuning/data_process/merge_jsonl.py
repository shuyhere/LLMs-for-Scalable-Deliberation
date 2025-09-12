#!/usr/bin/env python3
import argparse
import json
import os
from typing import Iterator, Dict, Any, Optional, Tuple


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, iterator: Iterator[dict]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for obj in iterator:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")


SUMMARY_PREFIX_RAW = "Below is a summary of people's opinions on the issue."
SUMMARY_PREFIX_NL = SUMMARY_PREFIX_RAW + "\n"
SUMMARY_PREFIX_SP = SUMMARY_PREFIX_RAW + " "


def normalize_to_target_schema(obj: Dict[str, Any], *,
                               want_prefix: Optional[bool],
                               prefix_format: Optional[str]) -> Dict[str, Any]:
    """Normalize a record to match summary_rating_extracted train schema.

    Target fields:
    - triplet_key: str
    - displayed_text: str (summary text)
    - answer_text: str (opinion)
    - question: str (with prefix "[Question] ")
    - rating_scores: List[int] order: [representativeness, informativeness, neutrality, policy]
    - folder_id: str (fallback to source indicator if absent)
    - source: str
    """
    # Detect negatives-style schema
    if "summary" in obj and "opinion" in obj and "scoring" in obj:
        rating_instance_id = obj.get("rating_instance_id") or obj.get("id") or ""
        # triplet_key from rating_instance_id by removing trailing suffix like "_rating"
        triplet_key = rating_instance_id.replace("_rating", "") if rating_instance_id else ""

        question = obj.get("question", "")
        # Ensure exact prefix formatting: "[Question] " (with one space)
        if question.startswith("[Question]") and not question.startswith("[Question] "):
            question = "[Question] " + question[len("[Question]"):].lstrip()

        displayed_text = obj.get("summary", "")
        # Normalize prefix presence and formatting to match base
        if want_prefix is True:
            # Ensure presence and exact spacing/newline formatting
            if displayed_text.startswith(SUMMARY_PREFIX_NL):
                displayed_text = prefix_format + displayed_text[len(SUMMARY_PREFIX_NL):]
            elif displayed_text.startswith(SUMMARY_PREFIX_SP):
                if prefix_format == SUMMARY_PREFIX_NL:
                    displayed_text = prefix_format + displayed_text[len(SUMMARY_PREFIX_SP):]
                else:
                    # already space format
                    pass
            elif displayed_text.startswith(SUMMARY_PREFIX_RAW):
                # Rare: raw without space/newline after, normalize
                displayed_text = prefix_format + displayed_text[len(SUMMARY_PREFIX_RAW):].lstrip()
            else:
                # No prefix; prepend with chosen format
                displayed_text = prefix_format + displayed_text.lstrip()
        elif want_prefix is False:
            # Strip any variant of the prefix if present
            if displayed_text.startswith(SUMMARY_PREFIX_NL):
                displayed_text = displayed_text[len(SUMMARY_PREFIX_NL):]
            elif displayed_text.startswith(SUMMARY_PREFIX_SP):
                displayed_text = displayed_text[len(SUMMARY_PREFIX_SP):]
            elif displayed_text.startswith(SUMMARY_PREFIX_RAW):
                displayed_text = displayed_text[len(SUMMARY_PREFIX_RAW):].lstrip()

        scoring = obj.get("scoring", {}) or {}
        # Normalize keys possibly with different capitalization
        represent = scoring.get("representiveness") or scoring.get("representativeness") or scoring.get("Representativeness") or 0
        informative = scoring.get("informativeness") or scoring.get("Informativeness") or 0
        neutrality = scoring.get("Neutrality") or scoring.get("neutrality") or 0
        policy = scoring.get("Policy Approval") or scoring.get("policy") or scoring.get("policy_approval") or 0
        rating_scores = [represent, informative, neutrality, policy]

        return {
            "triplet_key": triplet_key,
            "displayed_text": displayed_text,
            "answer_text": obj.get("opinion", ""),
            "question": question,
            "rating_scores": rating_scores,
            "folder_id": obj.get("folder_id", "dataset_argument"),
            "source": obj.get("source", "dataset-argument"),
        }

    # Already in target schema; return as-is
    return obj


def detect_base_prefix_format(base_path: str, probe_limit: int = 200) -> Tuple[Optional[bool], Optional[str]]:
    """Detect whether base uses the summary prefix and whether it is followed by space or newline.

    Returns (want_prefix, prefix_format) where:
      - want_prefix: True/False/None (None means undecided)
      - prefix_format: SUMMARY_PREFIX_SP or SUMMARY_PREFIX_NL or None
    """
    for i, obj in enumerate(iter_jsonl(base_path)):
        if i >= probe_limit:
            break
        if not isinstance(obj, dict):
            continue
        txt = obj.get("displayed_text")
        if not isinstance(txt, str):
            continue
        if txt.startswith(SUMMARY_PREFIX_SP):
            return True, SUMMARY_PREFIX_SP
        if txt.startswith(SUMMARY_PREFIX_NL):
            return True, SUMMARY_PREFIX_NL
        if txt.startswith(SUMMARY_PREFIX_RAW):
            # Has raw prefix but unknown following char, prefer space for consistency
            return True, SUMMARY_PREFIX_SP
    # If no evidence of prefix in base, assume not wanted
    return False, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONL files, normalizing additional records to target schema.")
    parser.add_argument("--base", required=True, help="Path to base train.jsonl (kept first)")
    parser.add_argument("--add", required=True, help="Path to additional JSONL to append (e.g., negatives)")
    parser.add_argument("--output", required=False, help="Output path. If omitted, writes train_aug.jsonl next to base.")
    args = parser.parse_args()

    base_path = os.path.abspath(args.base)
    add_path = os.path.abspath(args.add)

    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Base file not found: {base_path}")
    if not os.path.isfile(add_path):
        raise FileNotFoundError(f"Add file not found: {add_path}")

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base_dir = os.path.dirname(base_path)
        output_path = os.path.join(base_dir, "train_aug.jsonl")

    want_prefix, prefix_format = detect_base_prefix_format(base_path)

    def chain_iter():
        for obj in iter_jsonl(base_path):
            # Keep base records unchanged to preserve exact original formatting
            yield obj
        for obj in iter_jsonl(add_path):
            yield normalize_to_target_schema(obj, want_prefix=want_prefix, prefix_format=prefix_format)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(output_path, chain_iter())

    print(f"Merged: {base_path} + {add_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()


