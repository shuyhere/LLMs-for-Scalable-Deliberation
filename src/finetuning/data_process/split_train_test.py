#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Iterable, List, Tuple, Union


def read_json_or_jsonl(path: str) -> Tuple[List[dict], str]:
    """Read JSON (.json list) or JSON Lines (.jsonl) file and return list of dicts and detected format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        records: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records, "jsonl"
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data, "json"
            raise ValueError("JSON file must contain a top-level list of objects")
    else:
        raise ValueError("Unsupported file extension. Use .json or .jsonl")


def write_json(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, items: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_pair_id(item: dict) -> str:
    """Extract pair identifier from item. Try multiple possible fields."""
    # Try different possible fields for pair identification
    if "id" in item and "_rating_" in item["id"]:
        # Extract triplet ID from id field like "triplet_1306_rating_A"
        return item["id"].split("_rating_")[0]
    elif "triplet_id" in item:
        return item["triplet_id"]
    elif "user_dir" in item:
        return item["user_dir"]
    else:
        # Fallback: use question + comment as pair identifier
        question = item.get("question", "")
        comment = item.get("comment", "")
        return f"{question[:50]}_{comment[:50]}"


def group_items_by_pairs(items: List[dict]) -> dict:
    """Group items by their pair identifier."""
    pairs = {}
    for item in items:
        pair_id = extract_pair_id(item)
        if pair_id not in pairs:
            pairs[pair_id] = []
        pairs[pair_id].append(item)
    return pairs


def split_paired_items(items: List[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Split items while keeping pairs together and shuffling their order."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    
    rng = random.Random(seed)
    
    # Group items by pairs
    pairs = group_items_by_pairs(items)
    pair_ids = list(pairs.keys())
    
    # Shuffle pair order
    rng.shuffle(pair_ids)
    
    # Split pairs (not individual items)
    split_index = int(len(pair_ids) * train_ratio)
    train_pair_ids = set(pair_ids[:split_index])
    test_pair_ids = set(pair_ids[split_index:])
    
    # Collect items from train/test pairs
    train_items: List[dict] = []
    test_items: List[dict] = []
    
    # Shuffle the order of pairs to avoid consecutive pairs
    all_pairs = []
    for pair_id in pair_ids:
        pair_items = pairs[pair_id]
        # Shuffle items within each pair to avoid A always coming before B
        rng.shuffle(pair_items)
        all_pairs.append((pair_id, pair_items))
    
    # Shuffle the order of pairs again
    rng.shuffle(all_pairs)
    
    # Distribute pairs to train/test and shuffle the final order
    train_pairs = []
    test_pairs = []
    
    for pair_id, pair_items in all_pairs:
        if pair_id in train_pair_ids:
            train_pairs.extend(pair_items)
        else:
            test_pairs.extend(pair_items)
    
    # Final shuffle to mix up the order even more
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    
    train_items = train_pairs
    test_items = test_pairs
    
    return train_items, test_items


def split_items(items: List[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Main split function that handles both paired and unpaired data."""
    # Check if data appears to be paired (has common pair identifiers)
    pairs = group_items_by_pairs(items)
    
    # If most pairs have exactly 2 items, treat as paired data
    pair_sizes = [len(pair_items) for pair_items in pairs.values()]
    avg_pair_size = sum(pair_sizes) / len(pair_sizes) if pair_sizes else 0
    
    if avg_pair_size >= 1.8:  # Most pairs have 2 items
        print(f"Detected paired data: {len(pairs)} pairs, avg size {avg_pair_size:.1f}")
        return split_paired_items(items, train_ratio, seed)
    else:
        print("Detected unpaired data, using standard split")
        # Fallback to original method for unpaired data
        rng = random.Random(seed)
        indices = list(range(len(items)))
        rng.shuffle(indices)
        split_index = int(len(items) * train_ratio)
        train_idx = set(indices[:split_index])
        train_items: List[dict] = []
        test_items: List[dict] = []
        for i, obj in enumerate(items):
            (train_items if i in train_idx else test_items).append(obj)
        return train_items, test_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSON/JSONL dataset into 70/30 train/test.")
    parser.add_argument("input", type=str, help="Path to input .json or .jsonl file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio (default: 0.7)")
    parser.add_argument("--seed", type=int, default=6666, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory. If not set, a sibling folder named after the input basename will be created next to the input file.")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    items, fmt = read_json_or_jsonl(input_path)

    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = args.output_dir or os.path.join(base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    train_items, test_items = split_items(items, args.train_ratio, args.seed)

    if fmt == "jsonl":
        train_path = os.path.join(output_dir, "train.jsonl")
        test_path = os.path.join(output_dir, "test.jsonl")
        write_jsonl(train_path, train_items)
        write_jsonl(test_path, test_items)
    else:
        train_path = os.path.join(output_dir, "train.json")
        test_path = os.path.join(output_dir, "test.json")
        write_json(train_path, train_items)
        write_json(test_path, test_items)

    print(f"Input: {input_path}")
    print(f"Total: {len(items)} | Train: {len(train_items)} | Test: {len(test_items)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()


