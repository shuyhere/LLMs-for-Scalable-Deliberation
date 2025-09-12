#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import List


def read_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffle a JSONL file deterministically with a seed.")
    parser.add_argument("input", type=str, help="Path to input .jsonl file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Optional output path. If omitted, overwrite input in-place.")
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output) if args.output else in_path

    data = read_jsonl(in_path)
    rng = random.Random(args.seed)
    rng.shuffle(data)

    write_jsonl(out_path, data)
    print(f"Shuffled {len(data)} records.")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()


