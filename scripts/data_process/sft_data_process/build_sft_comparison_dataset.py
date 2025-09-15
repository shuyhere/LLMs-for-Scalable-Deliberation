#!/usr/bin/env python3
"""
Build SFT dataset from summary-rating annotation data for comparison tasks.

This script extracts data from annotation files and creates SFT training data
with the following structure:
- question: The original question
- comment: User's comment/answer 
- summary_a: First summary text
- summary_b: Second summary text
- comparison_scores: Four-dimensional comparison scores [perspective, informativeness, neutrality, policy]

Input: Directory containing annotation files
Output: JSONL file with SFT training examples
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import re
import html as html_lib
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT dataset from summary-rating comparison annotations")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input directory containing annotation files",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--include_incomplete",
        action="store_true",
        help="Include triplets with incomplete annotations",
    )
    return parser.parse_args()


def strip_html(text: Optional[str]) -> Optional[str]:
    """Remove HTML tags and clean text."""
    if not isinstance(text, str):
        return text
    # Remove tags
    no_tags = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities
    unescaped = html_lib.unescape(no_tags)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", unescaped).strip()
    return cleaned


def extract_summary_from_html(html_text: str) -> str:
    """Extract summary text from HTML display text."""
    # Find the paragraph content within the HTML
    p_match = re.search(r'<p>(.*?)</p>', html_text, re.DOTALL)
    if p_match:
        return strip_html(p_match.group(1))
    
    # Fallback: strip all HTML tags
    return strip_html(html_text)


def parse_comparison_scores(label_annotations: Dict[str, Any]) -> List[float]:
    """
    Parse comparison scores from label annotations.
    Returns [perspective, informativeness, neutrality, policy] scores.
    """
    score_mapping = {
        "Which summary is more representative of your perspective?": "perspective",
        "Which summary is more informative?": "informativeness", 
        "Which summary presents a more neutral and balanced view of the issue?": "neutrality",
        "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?": "policy"
    }
    
    scores = [0.0, 0.0, 0.0, 0.0]  # Default scores
    
    for question, dimension in score_mapping.items():
        if question in label_annotations:
            annotation = label_annotations[question]
            # Extract scale value (1 or 2, where 1 = A better, 2 = B better)
            if "scale_1" in annotation:
                score = 1.0  # A is better
            elif "scale_2" in annotation:
                score = 2.0  # B is better
            else:
                continue
            
            if dimension == "perspective":
                scores[0] = score
            elif dimension == "informativeness":
                scores[1] = score
            elif dimension == "neutrality":
                scores[2] = score
            elif dimension == "policy":
                scores[3] = score
    
    return scores


def process_user_directory(user_dir: Path) -> List[Dict[str, Any]]:
    """Process a single user directory containing both annotated_instances.jsonl and assigned_user_data.json."""
    results = []
    
    # Paths to the two files we need
    annotated_instances_file = user_dir / "annotated_instances.jsonl"
    assigned_user_data_file = user_dir / "assigned_user_data.json"
    
    if not annotated_instances_file.exists():
        print(f"Warning: No annotated_instances.jsonl found in {user_dir}", file=sys.stderr)
        return results
    
    if not assigned_user_data_file.exists():
        print(f"Warning: No assigned_user_data.json found in {user_dir}", file=sys.stderr)
        return results
    
    try:
        # Read annotated instances (contains the actual annotations)
        annotated_records = []
        with open(annotated_instances_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    annotated_records.append(json.loads(line))
        
        # Read assigned user data (contains metadata like summary_a_text and summary_b_text)
        with open(assigned_user_data_file, 'r', encoding='utf-8') as f:
            assigned_data = json.load(f)
        
    except Exception as e:
        print(f"Warning: Could not read files in {user_dir}: {e}", file=sys.stderr)
        return results
    
    # Group annotated records by triplet ID
    triplets = defaultdict(dict)
    
    for record_data in annotated_records:
        if not isinstance(record_data, dict):
            continue
            
        # Extract triplet ID from the id field
        record_id = record_data.get('id', '')
        triplet_id = re.sub(r'_(rating|comparison|question|intro|end)$', '', record_id)
        
        # Infer record type from ID since type field is empty
        if '_comparison' in record_id:
            record_type = 'comparison'
        elif '_question' in record_id:
            record_type = 'question'
        elif '_rating' in record_id:
            record_type = 'rating'
        else:
            record_type = record_data.get('type', '')
        
        triplets[triplet_id][record_type] = record_data
    
    # Process each triplet
    for triplet_id, triplet_data in triplets.items():
        # Skip if missing essential components
        if 'comparison' not in triplet_data:
            continue
            
        comparison_record = triplet_data['comparison']
        question_record = triplet_data.get('question', {})
        
        # Get metadata from assigned_user_data.json for this triplet
        comparison_key = f"{triplet_id}_comparison"
        if comparison_key not in assigned_data:
            continue
            
        metadata = assigned_data[comparison_key]
        
        # Extract question text
        question_text = ""
        if question_record:
            displayed_text = question_record.get('displayed_text', '')
            question_text = strip_html(displayed_text)
            # Remove [Question] prefix if present
            question_text = re.sub(r'^\[Question\]\s*', '', question_text, flags=re.IGNORECASE)
        
        # Extract user comment/answer
        comment_text = ""
        if question_record and 'label_annotations' in question_record:
            answer_annotation = question_record['label_annotations'].get('answer', {})
            if 'text_box' in answer_annotation:
                comment_text = answer_annotation['text_box']
        
        # Extract summary texts from metadata
        summary_a_text = metadata.get('summary_a_text', '')
        summary_b_text = metadata.get('summary_b_text', '')
        
        # Parse comparison scores
        comparison_scores = [0.0, 0.0, 0.0, 0.0]
        if 'label_annotations' in comparison_record:
            comparison_scores = parse_comparison_scores(comparison_record['label_annotations'])
        
        # Skip if no valid scores found
        if all(score == 0.0 for score in comparison_scores):
            continue
        
        # Create SFT example
        sft_example = {
            "question": question_text,
            "comment": comment_text,
            "summary_a": summary_a_text,
            "summary_b": summary_b_text,
            "comparison_scores": comparison_scores,  # [perspective, informativeness, neutrality, policy]
            "triplet_id": triplet_id,
            "source_file": str(user_dir)
        }
        
        results.append(sft_example)
    
    return results


def process_directory(input_dir: Path, include_incomplete: bool = False) -> List[Dict[str, Any]]:
    """Process all annotation files in a directory."""
    all_results = []
    
    # Find all user directories
    for user_dir in input_dir.iterdir():
        if user_dir.is_dir():
            print(f"Processing {user_dir}", file=sys.stderr)
            results = process_user_directory(user_dir)
            all_results.extend(results)
            print(f"  Found {len(results)} comparison examples", file=sys.stderr)
    
    return all_results


def create_sft_prompt(example: Dict[str, Any]) -> str:
    """Create SFT training prompt from example."""
    prompt = f"""Given the following question and user comment, compare two summaries across four dimensions: perspective representation, informativeness, neutrality, and policy usefulness.

Question: {example['question']}
User Comment: {example['comment']}

Summary A: {example['summary_a']}
Summary B: {example['summary_b']}

Please provide your comparison scores for each dimension (1 = Summary A is better, 2 = Summary B is better):

Perspective Representation, 
Informativeness, 
Neutrality, 
Policy Usefulness. """
    
    return prompt


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Process all annotation files
    print(f"Processing annotation files in {input_dir}", file=sys.stderr)
    all_examples = process_directory(input_dir, args.include_incomplete)
    
    print(f"Total examples found: {len(all_examples)}", file=sys.stderr)
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write SFT training data
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            sft_example = {
                "prompt": create_sft_prompt(example),
                "completion": f"Perspective: {int(example['comparison_scores'][0])}, Informativeness: {int(example['comparison_scores'][1])}, Neutrality: {int(example['comparison_scores'][2])}, Policy: {int(example['comparison_scores'][3])}",
                "metadata": {
                    "triplet_id": example['triplet_id'],
                    "source_file": example['source_file'],
                    "comparison_scores": example['comparison_scores']
                }
            }
            f.write(json.dumps(sft_example, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(all_examples)} SFT examples to {output_path}", file=sys.stderr)
    
    # Print statistics
    score_counts = defaultdict(int)
    for example in all_examples:
        for i, score in enumerate(example['comparison_scores']):
            if score > 0:
                score_counts[i] += 1
    
    dimensions = ['Perspective', 'Informativeness', 'Neutrality', 'Policy']
    print("\nScore distribution by dimension:", file=sys.stderr)
    for i, dimension in enumerate(dimensions):
        print(f"  {dimension}: {score_counts[i]} examples", file=sys.stderr)


if __name__ == "__main__":
    main()
