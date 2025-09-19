#!/usr/bin/env python3
"""
Validate that every (instance_id, user) pair in task_assignment.json (assigned)
exists in annotated_instances.csv.

Defaults target the full_augment outputs. You can override via CLI args.
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import pandas as pd


def load_task_assignment(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    assigned = data.get("assigned", {})
    unassigned = data.get("unassigned", {})
    return assigned, unassigned


def build_assigned_pairs(assigned: dict):
    """Return set of (instance_id, user) and summary stats."""
    pairs = set()
    users = set()
    inst_with_lists = 0
    slots = 0
    for inst_id, lst in assigned.items():
        if isinstance(lst, list):
            inst_with_lists += 1
            slots += len(lst)
            for u in lst:
                u_str = str(u)
                users.add(u_str)
                pairs.add((inst_id, u_str))
    return pairs, users, inst_with_lists, slots


def build_annotation_pairs(csv_path: str):
    df = pd.read_csv(csv_path, dtype={"user": str, "instance_id": str})
    # Only keep rows that have both user and instance_id
    df = df[df["user"].notna() & df["instance_id"].notna()].copy()
    ann_pairs = set(zip(df["instance_id"].astype(str), df["user"].astype(str)))
    return df, ann_pairs


def summarize_missing(missing_pairs):
    by_user = Counter(u for _, u in missing_pairs)
    by_inst = Counter(inst for inst, _ in missing_pairs)
    return by_user, by_inst


def main():
    parser = argparse.ArgumentParser(description="Check assigned vs annotated pairs")
    parser.add_argument(
        "--assignment",
        default="/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment/task_assignment.json",
        help="Path to task_assignment.json",
    )
    parser.add_argument(
        "--annotations",
        default="/home/ec2-user/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full_augment/annotated_instances.csv",
        help="Path to annotated_instances.csv",
    )
    args = parser.parse_args()

    print("Loading task assignment ...")
    assigned, unassigned = load_task_assignment(args.assignment)
    assigned_pairs, assigned_users, assigned_questions, assigned_slots = build_assigned_pairs(assigned)

    print("Loading annotations ...")
    df_ann, ann_pairs = build_annotation_pairs(args.annotations)

    # Compare
    missing = assigned_pairs - ann_pairs
    by_user, by_inst = summarize_missing(missing)

    print("\nSummary:")
    print({
        "assigned_questions": assigned_questions,
        "assigned_slots": assigned_slots,
        "assigned_unique_users": len(assigned_users),
        "annotated_rows": len(df_ann),
        "annotated_unique_pairs": len(ann_pairs),
        "missing_pairs": len(missing),
    })

    if missing:
        print("\nTop missing users (user -> missing count):")
        for user, cnt in by_user.most_common(10):
            print(f"  {user}: {cnt}")

        print("\nTop missing instances (instance_id -> missing count):")
        for inst, cnt in by_inst.most_common(10):
            print(f"  {inst}: {cnt}")

        print("\nSample missing pairs (instance_id, user):")
        for i, (inst, user) in enumerate(sorted(missing)[:20], 1):
            print(f"  {i:2d}. {inst} :: {user}")
    else:
        print("\n✅ All assigned pairs exist in annotated_instances.csv")


if __name__ == "__main__":
    main()


