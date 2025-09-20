#!/usr/bin/env python3
"""
Script to merge annotation data with original datasets.
Extracts question and opinion data from annotation files and appends them to the original datasets.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


def extract_question_from_displayed_text(displayed_text: str) -> Optional[str]:
    """Extract question from displayed_text field."""
    # Look for pattern: <h3>[Question]</h3><h4>QUESTION_TEXT</h4>
    pattern = r'<h3>\[Question\]</h3><h4>(.*?)</h4>'
    match = re.search(pattern, displayed_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def load_annotation_data(annotation_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all annotation data and group by question.
    Returns: {question: [annotation_entries]}
    """
    annotation_data = {}
    annotation_path = Path(annotation_dir)
    
    if not annotation_path.exists():
        print(f"Annotation directory not found: {annotation_dir}")
        return annotation_data
    
    # Find all annotated_instances.jsonl files
    jsonl_files = list(annotation_path.rglob("annotated_instances.jsonl"))
    print(f"Found {len(jsonl_files)} annotation files")
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        
                        # Extract question from displayed_text
                        if 'displayed_text' in data:
                            question = extract_question_from_displayed_text(data['displayed_text'])
                            if question:
                                if question not in annotation_data:
                                    annotation_data[question] = []
                                annotation_data[question].append(data)
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue
    
    print(f"Loaded annotation data for {len(annotation_data)} unique questions")
    return annotation_data


def load_original_dataset(dataset_file: str) -> Dict[str, Any]:
    """Load original dataset file."""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_matching_question(original_question: str, annotation_questions: List[str]) -> Optional[str]:
    """
    Find the best matching question in annotation data.
    Uses simple string matching and normalization.
    """
    # Normalize question for comparison
    def normalize(q: str) -> str:
        # Remove common suffixes that appear in original questions but not in annotations
        q = re.sub(r'\s*Please answer briefly in \d+–\d+ sentences?\.?\s*$', '', q)
        q = re.sub(r'\s*Please answer briefly in \d+–\d+ sentences?\.?\s*$', '', q)
        q = re.sub(r'\s*Please answer briefly in \d+–\d+ sentences?\.?\s*$', '', q)
        return re.sub(r'\s+', ' ', q.lower().strip())
    
    original_norm = normalize(original_question)
    
    best_match = None
    best_score = 0
    
    for ann_question in annotation_questions:
        ann_norm = normalize(ann_question)
        
        # Check for exact match
        if original_norm == ann_norm:
            return ann_question
        
        # Check for substring match
        if original_norm in ann_norm or ann_norm in original_norm:
            score = min(len(original_norm), len(ann_norm)) / max(len(original_norm), len(ann_norm))
            if score > best_score:
                best_score = score
                best_match = ann_question
    
    # Return match if score is high enough
    if best_score > 0.7:
        return best_match
    
    return None


def extract_opinions_from_annotation(annotation_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract opinion data from annotation entries."""
    opinions = []
    
    for entry in annotation_entries:
        if entry.get('id', '').endswith('_question'):
            # Extract opinion from text_box
            label_annotations = entry.get('label_annotations', {})
            if 'answer' in label_annotations and 'text_box' in label_annotations['answer']:
                opinion_text = label_annotations['answer']['text_box'].strip()
                if opinion_text:
                    opinions.append({
                        'index': len(opinions),  # Will be updated later
                        'comment': opinion_text
                    })
    
    return opinions


def merge_dataset_with_annotations(original_data: Dict[str, Any], 
                                 annotation_data: Dict[str, List[Dict[str, Any]]],
                                 keep_original: int = 90) -> Dict[str, Any]:
    """
    Merge original dataset with annotation data.
    Keep first 'keep_original' comments from original data, then append new ones.
    """
    merged_data = original_data.copy()
    
    # Get original question
    original_question = original_data.get('question', '')
    print(f"Processing question: {original_question[:100]}...")
    
    # Find matching annotation question
    matching_question = find_matching_question(original_question, list(annotation_data.keys()))
    
    if not matching_question:
        print(f"No matching annotation found for: {original_question[:50]}...")
        return merged_data
    
    print(f"Found matching annotation: {matching_question[:50]}...")
    
    # Extract opinions from annotation
    new_opinions = extract_opinions_from_annotation(annotation_data[matching_question])
    
    if not new_opinions:
        print("No opinions found in annotation data")
        return merged_data
    
    # Get original comments
    original_comments = original_data.get('comments', [])
    
    # Keep first 'keep_original' comments from original data
    kept_comments = original_comments[:keep_original]
    
    # Update indices for kept comments
    for i, comment in enumerate(kept_comments):
        comment['index'] = i
    
    # Add new comments from annotation
    new_comments = []
    for i, opinion in enumerate(new_opinions):
        new_comment = {
            'index': len(kept_comments) + i,
            'comment': opinion['comment']
        }
        new_comments.append(new_comment)
    
    # Combine all comments
    merged_data['comments'] = kept_comments + new_comments
    
    print(f"Original comments: {len(original_comments)}")
    print(f"Kept original comments: {len(kept_comments)}")
    print(f"Added new comments: {len(new_comments)}")
    print(f"Total comments: {len(merged_data['comments'])}")
    
    return merged_data


def process_all_datasets(original_dataset_dir: str, 
                        annotation_dir: str, 
                        output_dir: str,
                        keep_original: int = 90):
    """Process all datasets in the original dataset directory."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load annotation data
    print("Loading annotation data...")
    annotation_data = load_annotation_data(annotation_dir)
    
    if not annotation_data:
        print("No annotation data found!")
        return
    
    # Process each original dataset
    original_dataset_path = Path(original_dataset_dir)
    json_files = list(original_dataset_path.glob("*.json"))
    
    print(f"Found {len(json_files)} original dataset files")
    
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        
        try:
            # Load original dataset
            original_data = load_original_dataset(str(json_file))
            
            # Merge with annotation data
            merged_data = merge_dataset_with_annotations(original_data, annotation_data, keep_original)
            
            # Save merged dataset
            output_file = output_path / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved merged dataset: {output_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\nProcessing complete! Merged datasets saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge annotation data with original datasets")
    parser.add_argument("--original-dataset-dir", 
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/cleaned_new_dataset",
                       help="Directory containing original dataset files")
    parser.add_argument("--annotation-dir", 
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full",
                       help="Directory containing annotation data")
    parser.add_argument("--output-dir", 
                       default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotationV0_V1_dataset",
                       help="Output directory for merged datasets")
    parser.add_argument("--keep-original", 
                       type=int, 
                       default=90,
                       help="Number of original comments to keep (default: 90)")
    
    args = parser.parse_args()
    
    print("Starting annotation data merge process...")
    print(f"Original dataset dir: {args.original_dataset_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Keep original comments: {args.keep_original}")
    
    process_all_datasets(args.original_dataset_dir, 
                        args.annotation_dir, 
                        args.output_dir,
                        args.keep_original)


if __name__ == "__main__":
    main()
