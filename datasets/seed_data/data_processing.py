import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def find_question_column(headers: List[str]) -> Optional[str]:
    """Return the header that contains the question prompt.

    Heuristics:
    - Prefer a column that ends with "(text)" as exported by Deliberation.io.
    - Fallback to a column that contains a question mark and is not a known administrative field.
    - If none found, return None.
    """
    # Prefer explicit question text column
    for header in headers:
        if header.strip().endswith("(text)"):
            return header

    # Fallback: any header with a question mark that isn't obviously metadata
    admin_like = {
        "ID",
        "Session ID",
        "Vote",
        "Opinion Text",
        "Submitted At",
        "Is Flagged",
        "Flagged By",
        "Flagged At",
        "Flag Reason",
        "Access Type",
        "Group/Participant Name",
        "Participation Code",
        "Access Granted At",
        "Participation URL",
        "Page Visit Timestamps",
        "Module Status",
        "Full Response Metadata (JSON)",
    }
    for header in headers:
        if header in admin_like:
            continue
        if "?" in header:
            return header

    return None


def normalize_question(question_header: str) -> str:
    """Normalize the question string from the header text.

    - Strip trailing " (text)" suffix if present.
    - Collapse whitespace.
    """
    question = re.sub(r"\s*\(text\)\s*$", "", question_header).strip()
    question = re.sub(r"\s+", " ", question)
    return question


def extract_comment(row: Dict[str, str], question_col: Optional[str]) -> Optional[str]:
    """Extract the free-text comment from a CSV row.

    Priority:
    1) Use "Opinion Text" if present and non-empty
    2) Use the question column value if present
    Returns None if no content.
    """
    opinion = row.get("Opinion Text", "").strip()
    if opinion:
        return opinion
    if question_col:
        val = row.get(question_col, "").strip()
        if val:
            return val
    return None


def csv_to_json_structure(csv_path: Path) -> Optional[Dict]:
    """Convert one CSV file to the target JSON structure.

    Target schema:
    {
      "question": <str>,
      "comments": [ {"index": <int>, "comment": <str>}, ... ]
    }
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            return None

        question_col = find_question_column(headers)
        if not question_col:
            # If we cannot infer the question, skip this file
            return None

        question = normalize_question(question_col)

        comments = []
        idx = 0
        for row in reader:
            text = extract_comment(row, question_col)
            if not text:
                continue
            comments.append({"index": idx, "comment": text})
            idx += 1

        return {"question": question, "comments": comments}


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a CSV basename (without extension)."""
    # Preserve letters, digits, spaces, dashes, underscores; replace others with space
    safe = re.sub(r"[^\w\-\s]", " ", name, flags=re.UNICODE)
    # Collapse whitespace
    safe = re.sub(r"\s+", " ", safe).strip()
    # Replace spaces with single dash for consistency
    safe = safe.replace(" ", "-")
    return safe or "dataset"


def process_seed_data(seed_dir: Path, output_dir: Path) -> List[Path]:
    """Process all CSVs in seed_dir and write JSON files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for csv_path in sorted(seed_dir.glob("*.csv")):
        data = csv_to_json_structure(csv_path)
        if not data:
            continue

        base = csv_path.name
        if base.endswith("_responses.csv"):
            base = base[: -len("_responses.csv")]
        else:
            base = csv_path.stem
        out_name = sanitize_filename(base) + ".json"
        out_path = output_dir / out_name

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        written.append(out_path)

    return written


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    seed_dir = project_root / "datasets" / "seed_data"
    output_dir = project_root / "datasets" / "cleaned_new_dataset"

    written = process_seed_data(seed_dir, output_dir)
    print(f"Wrote {len(written)} files to {output_dir}")
    for p in written:
        print(f" - {p.name}")


if __name__ == "__main__":
    main()


