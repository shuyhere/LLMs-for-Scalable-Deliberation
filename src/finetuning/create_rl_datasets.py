#!/usr/bin/env python3
"""
Create RL datasets from annotation data for 4 dimensions.
Extracts choose/reject pairs from comparison annotations for reinforcement learning.
Uses the template format: "We have made a deliberation with many annotators on the issue: {question}"
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from collections import defaultdict

# Dimension mapping
DIMENSIONS = {
    "perspective": "Which summary is more representative of your perspective?",
    "informativeness": "Which summary is more informative?", 
    "neutrality": "Which summary presents a more neutral and balanced view of the issue?",
    "policy": "Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?"
}

# Question templates for each dimension (following the evaluation.py format)
QUESTION_TEMPLATES = {
    "perspective": "Which summary is more representative of the annotator's opinion?",
    "informativeness": "Which summary is more informative?",
    "neutrality": "Which summary presents a more neutral and balanced view of the issue?",
    "policy": "Which summary would you prefer to be used by policy makers to make decisions relevant to the issue?"
}

def clean_html_text(text: str) -> str:
    """Clean HTML tags and extract plain text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_summary_from_text(text: str) -> str:
    """Extract summary text from HTML content."""
    # Look for content between <p> tags or after <hr>
    if '<p>' in text:
        # Extract content between <p> tags
        p_match = re.search(r'<p>(.*?)</p>', text, re.DOTALL)
        if p_match:
            return clean_html_text(p_match.group(1))
    
    # If no <p> tags, try to extract after <hr>
    if '<hr>' in text:
        parts = text.split('<hr>')
        if len(parts) > 1:
            return clean_html_text(parts[1])
    
    # Fallback: clean the entire text
    return clean_html_text(text)

def create_rl_prompt(question: str, annotator_answer: str, chosen_summary: str, rejected_summary: str, dimension: str) -> str:
    """Create RL prompt following the evaluation.py template format."""
    
    # Include annotator's perspective if available
    perspective_text = ""
    if annotator_answer:
        perspective_text = f"""\n\nOne annotator's opinion on this question is:
{annotator_answer}\n"""
    
    # Create the prompt following the template
    prompt = f"""We have made a deliberation with many annotators on the issue: {question}{perspective_text}

{QUESTION_TEMPLATES[dimension]}"""
    
    return prompt

def process_annotation_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a single annotation file and extract RL data."""
    rl_data = []
    
    # Get the directory containing the annotation file
    annotation_dir = file_path.parent
    assigned_user_data_file = annotation_dir / "assigned_user_data.json"
    
    # Load assigned_user_data.json to get summary texts and IDs
    try:
        with open(assigned_user_data_file, 'r', encoding='utf-8') as f:
            assigned_data = json.load(f)
    except Exception as e:
        print(f"Error reading {assigned_user_data_file}: {e}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Parse JSONL format (one JSON object per line)
            annotations = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        line_data = json.loads(line)
                        annotations.append(line_data)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Group annotations by triplet ID
    triplet_data = {}
    for annotation in annotations:
        annotation_id = annotation.get('id', '')
        if '_' in annotation_id:
            triplet_id = annotation_id.rsplit('_', 1)[0]  # Remove suffix like _comparison, _question
            if triplet_id not in triplet_data:
                triplet_data[triplet_id] = {}
            
            if annotation_id.endswith('_question'):
                triplet_data[triplet_id]['question'] = annotation
            elif annotation_id.endswith('_comparison'):
                triplet_data[triplet_id]['comparison'] = annotation
    
    # Process each triplet
    for triplet_id, data in triplet_data.items():
        question_data = data.get('question')
        comparison_data = data.get('comparison')
        
        if not all([question_data, comparison_data]):
            continue
        
        # Get question text directly from assigned_user_data.json
        question_key = f"{triplet_id}_question"
        if question_key in assigned_data:
            question_info = assigned_data[question_key]
            question_text = question_info.get('question', '')
        else:
            # Fallback to question_data if not found in assigned_data
            question_text = question_data.get('question', '')
        
        # Remove [Question] prefix if present
        if question_text.startswith('[Question]'):
            question_text = question_text[11:].strip()
        
        # Extract annotator's answer from question data
        annotator_answer = ""
        question_annotations = question_data.get('label_annotations', {})
        if 'answer' in question_annotations:
            answer_data = question_annotations['answer']
            if 'text_box' in answer_data:
                annotator_answer = answer_data['text_box']
        
        # Get summary texts directly from comparison data in assigned_user_data
        comparison_key = f"{triplet_id}_comparison"
        if comparison_key in assigned_data:
            comparison_info = assigned_data[comparison_key]
            summary_a_text = comparison_info.get('summary_a_text', '')
            summary_b_text = comparison_info.get('summary_b_text', '')
            summary_a_id = comparison_info.get('summary_a_id', '')
            summary_b_id = comparison_info.get('summary_b_id', '')
        else:
            continue
        
        if not summary_a_text or not summary_b_text:
            continue
        
        # Use summary texts directly without cleaning
        summary_a_clean = summary_a_text
        summary_b_clean = summary_b_text
        
        # Extract comparison annotations
        comparison_annotations = comparison_data.get('label_annotations', {})
        
        # Process each dimension
        for dim_name, dim_question in DIMENSIONS.items():
            if dim_question in comparison_annotations:
                # Check for both scale_1 and scale_2 formats
                scale_value = None
                if 'scale_1' in comparison_annotations[dim_question]:
                    scale_value = comparison_annotations[dim_question]['scale_1']
                elif 'scale_2' in comparison_annotations[dim_question]:
                    scale_value = comparison_annotations[dim_question]['scale_2']
                
                if scale_value is None:
                    continue
                    
                # scale_value = "1" means Summary A is chosen, "2" means Summary B is chosen
                if scale_value == "1":
                    chosen_summary = summary_a_clean
                    rejected_summary = summary_b_clean
                else:  # scale_value == "2"
                    chosen_summary = summary_b_clean
                    rejected_summary = summary_a_clean
                
                # Create RL prompt
                rl_prompt = create_rl_prompt(
                    question_text, 
                    annotator_answer, 
                    chosen_summary, 
                    rejected_summary, 
                    dim_name
                )
                
                # Create RL data entry
                rl_entry = {
                    "id": f"{comparison_data.get('id', 'unknown')}_{dim_name}",
                    "prompt": rl_prompt,
                    "question": question_text,
                    "annotator_answer": annotator_answer,
                    "chosen": chosen_summary,
                    "rejected": rejected_summary,
                    "dimension": dim_name,
                    "dimension_question": QUESTION_TEMPLATES[dim_name],
                    "raw_id": comparison_data.get('raw_id', ''),
                    "model_a": comparison_data.get('model_a', ''),
                    "model_b": comparison_data.get('model_b', ''),
                    "summary_a_id": summary_a_id,
                    "summary_b_id": summary_b_id
                }
                
                rl_data.append(rl_entry)
    
    return rl_data

def process_all_annotations(annotation_dir: Path, max_files: int = None) -> Dict[str, List[Dict[str, Any]]]:
    """Process all annotation files and group by dimension."""
    all_data = defaultdict(list)
    
    # Find all annotation files (look for annotated_instances.jsonl files)
    json_files = []
    for file_path in annotation_dir.glob("**/annotated_instances.jsonl"):
        json_files.append(file_path)
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"Found {len(json_files)} annotation files")
    
    processed_count = 0
    for json_file in json_files:
        rl_data = process_annotation_file(json_file)
        for entry in rl_data:
            all_data[entry['dimension']].append(entry)
        if rl_data:
            processed_count += 1
    
    print(f"Processed {processed_count} files with valid data")
    return dict(all_data)

def save_rl_datasets(data_by_dimension: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Save RL datasets for each dimension."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dimension, data in data_by_dimension.items():
        output_file = output_dir / f"{dimension}_rl_dataset.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(data)} entries for {dimension} to {output_file}")
    
    # Also save a combined dataset
    all_data = []
    for data in data_by_dimension.values():
        all_data.extend(data)
    
    combined_file = output_dir / "all_dimensions_rl_dataset.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(all_data)} total entries to {combined_file}")

def print_statistics(data_by_dimension: Dict[str, List[Dict[str, Any]]]):
    """Print statistics about the generated datasets."""
    print("\n" + "="*60)
    print("RL DATASET STATISTICS")
    print("="*60)
    
    total_entries = 0
    for dimension, data in data_by_dimension.items():
        print(f"\n{dimension.upper()}:")
        print(f"  Entries: {len(data)}")
        total_entries += len(data)
        
        # Count unique questions
        unique_questions = len(set(entry['question'] for entry in data))
        print(f"  Unique questions: {unique_questions}")
        
        # Count model pairs
        model_pairs = set()
        for entry in data:
            model_pair = tuple(sorted([entry['model_a'], entry['model_b']]))
            model_pairs.add(model_pair)
        print(f"  Model pairs: {len(model_pairs)}")
        
        # Show sample prompt
        if data:
            print(f"  Sample prompt (first 200 chars):")
            print(f"    {data[0]['prompt'][:200]}...")
    
    print(f"\nTOTAL ENTRIES: {total_entries}")

def main():
    parser = argparse.ArgumentParser(description="Create RL datasets from annotation data")
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/annotation/summary-rating/annotation_output/full",
        help="Directory containing annotation files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/rl_datasets",
        help="Output directory for RL datasets"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    
    if not annotation_dir.exists():
        print(f"Annotation directory {annotation_dir} does not exist")
        return
    
    print(f"Processing annotations from {annotation_dir}")
    print(f"Output will be saved to {output_dir}")
    
    # Process all annotations
    data_by_dimension = process_all_annotations(annotation_dir, args.max_files)
    
    if not data_by_dimension:
        print("No valid data found")
        return
    
    # Print statistics
    print_statistics(data_by_dimension)
    
    # Save datasets
    save_rl_datasets(data_by_dimension, output_dir)
    
    print(f"\n✅ RL datasets created successfully in {output_dir}")

if __name__ == "__main__":
    main()
