#!/usr/bin/env python3
"""
Script to run human-LLM correlation experiments
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.llm_evaluation.human_aligned_evaluator import HumanAlignedEvaluator
from src.utils.data_processing import HumanAnnotationDataProcessor
from config.human_llm_correlation import (
    PROJECT_ROOT, DEFAULT_ANNOTATION_PATH, DEFAULT_OUTPUT_PATH,
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_SAMPLE_SIZE, DEFAULT_DEBUG
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_demo_mode(evaluator, annotation_path, output_path):
    """Run demo mode - show only prompts and outputs"""
    # Load annotations
    data_processor = HumanAnnotationDataProcessor(debug=True)
    rating_annotations, comparison_annotations = data_processor.load_human_annotations(annotation_path)
    
    # Test rating evaluation
    if rating_annotations:
        test_rating = rating_annotations[0]
        
        if 'metadata' in test_rating:
            metadata = test_rating['metadata']
            summary = metadata.get('text', '')
            question = metadata.get('question', '')
            annotator_answer = test_rating.get('annotator_answer', '')
            
            # Generate and show the prompt
            from src.utils.prompts.evaluation import HumanAnnotationPrompt
            prompt = HumanAnnotationPrompt(
                summary=summary,
                question=question,
                annotator_answer=annotator_answer,
                task_type="rating"
            )
            user_input = prompt.get_rating_prompt()
            print("RATING PROMPT:")
            print("=" * 60)
            print(user_input)
            print("=" * 60)
            
            # Get LLM response
            try:
                result = evaluator.evaluate_rating(summary, question, annotator_answer)
                print("\nRATING OUTPUT:")
                print("=" * 60)
                if result['status'] == 'success':
                    print(f"Ratings: {result['ratings']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                print("=" * 60)
            except Exception as e:
                print(f"Error: {str(e)}")
    
    # Test comparison evaluation
    if comparison_annotations:
        test_comparison = comparison_annotations[0]
        
        if 'metadata' in test_comparison:
            metadata = test_comparison['metadata']
            summary_a = metadata.get('summary_a_text', '')
            summary_b = metadata.get('summary_b_text', '')
            question = metadata.get('question', '')
            annotator_answer = test_comparison.get('annotator_answer', '')
            
            # Generate and show the prompt
            from src.utils.prompts.evaluation import HumanAnnotationPrompt
            prompt = HumanAnnotationPrompt(
                summary_a=summary_a,
                summary_b=summary_b,
                question=question,
                annotator_answer=annotator_answer,
                task_type="comparison"
            )
            user_input = prompt.get_comparison_prompt()
            print("\nCOMPARISON PROMPT:")
            print("=" * 60)
            print(user_input)
            print("=" * 60)
            
            # Get LLM response
            try:
                result = evaluator.evaluate_comparison(
                    summary_a,
                    summary_b,
                    question,
                    annotator_answer
                )
                print("\nCOMPARISON OUTPUT:")
                print("=" * 60)
                if result['status'] == 'success':
                    print(f"Comparisons: {result['comparisons']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                print("=" * 60)
            except Exception as e:
                print(f"Error: {str(e)}")


def run_incremental_correlation_experiment(evaluator, annotation_path, output_path, sample_size, resume=False, checkpoint_interval=10):
    """Run correlation experiment with incremental processing and checkpoint support"""
    
    # Define checkpoint file
    checkpoint_file = output_path / f"checkpoint_{evaluator.model}.json"
    final_output_file = output_path / f"human_llm_correlation_{evaluator.model}.json"
    
    # Load existing results if resuming
    if resume and checkpoint_file.exists():
        print(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed_rating_ids = set(result['annotation_id'] for result in results.get('rating_results', []))
        processed_comparison_ids = set(result['annotation_id'] for result in results.get('comparison_results', []))
        print(f"Found {len(processed_rating_ids)} rating and {len(processed_comparison_ids)} comparison results to resume from")
        print(f"Checkpoint contains {len(results.get('rating_results', []))} rating results and {len(results.get('comparison_results', []))} comparison results")
    else:
        print("Starting fresh experiment")
        results = {
            "experiment_metadata": {
                "model": evaluator.model,
                "temperature": evaluator.temperature,
                "timestamp": evaluator.timestamp if hasattr(evaluator, 'timestamp') else None,
                "n_rating_samples": 0,
                "n_comparison_samples": 0
            },
            "rating_results": [],
            "comparison_results": [],
            "correlations": {}
        }
        processed_rating_ids = set()
        processed_comparison_ids = set()
    
    # Load all annotations
    print(f"Loading human annotations from {annotation_path}")
    rating_annotations, comparison_annotations = evaluator.data_processor.load_human_annotations(annotation_path)
    
    print(f"Found {len(rating_annotations)} rating annotations")
    print(f"Found {len(comparison_annotations)} comparison annotations")
    
    # Filter out already processed annotations
    rating_annotations = [ann for ann in rating_annotations if ann['id'] not in processed_rating_ids]
    comparison_annotations = [ann for ann in comparison_annotations if ann['id'] not in processed_comparison_ids]
    
    print(f"Processing {len(rating_annotations)} new rating annotations")
    print(f"Processing {len(comparison_annotations)} new comparison annotations")
    
    if len(rating_annotations) == 0 and len(comparison_annotations) == 0:
        print("No new annotations to process. All data has been processed.")
        print("Calculating final correlations...")
        correlations = evaluator.calculate_all_correlations(results['rating_results'], results['comparison_results'])
        results['correlations'] = correlations
        save_checkpoint(results, final_output_file)
        print(f"Final results saved to: {final_output_file}")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("Checkpoint file cleaned up")
        return results
    
    # Process rating annotations incrementally
    for i, ann in enumerate(rating_annotations):
        try:
            print(f"Processing rating annotation {i+1}/{len(rating_annotations)}: {ann['id']}")
            
            # Get summary and question from metadata
            if 'metadata' not in ann:
                continue
            
            metadata = ann['metadata']
            summary = metadata.get('text', '')
            question = metadata.get('question', '')
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary or not question:
                continue
            
            # Get LLM evaluation
            llm_result = evaluator.evaluate_rating(summary, question, annotator_answer)
            human_ratings = evaluator.data_processor.extract_human_ratings(ann)
            
            if llm_result['status'] == 'success' and human_ratings:
                # Create result
                unique_id = f"rating_{ann['id']}_{evaluator.model}"
                data_source = {
                    'annotation_id': ann['id'],
                    'annotation_type': 'rating',
                    'relative_path': f"annotation/summary-rating/annotation_output/full/{ann.get('user_id', 'unknown')}/annotated_instances.jsonl",
                    'absolute_path': str(annotation_path / ann.get('user_id', 'unknown') / 'annotated_instances.jsonl'),
                    'user_id': ann.get('user_id', 'unknown'),
                    'timestamp': ann.get('timestamp', 'unknown')
                }
                
                result = {
                    'unique_id': unique_id,
                    'annotation_id': ann['id'],
                    'model_name': evaluator.model,
                    'model_temperature': evaluator.temperature,
                    'timestamp': evaluator.timestamp if hasattr(evaluator, 'timestamp') else None,
                    'human_ratings': human_ratings,
                    'llm_result': llm_result,
                    'question': question,
                    'has_annotator_answer': bool(annotator_answer),
                    'annotator_answer': annotator_answer,
                    'data_source': data_source,
                    'metadata': {
                        'topic': metadata.get('topic', ''),
                        'model': metadata.get('model', ''),
                        'comment_num': metadata.get('comment_num', 0),
                        'summary_length': len(summary) if summary else 0
                    }
                }
                
                results['rating_results'].append(result)
                processed_rating_ids.add(ann['id'])
                
                # Save checkpoint every N samples
                if (i + 1) % checkpoint_interval == 0:
                    save_checkpoint(results, checkpoint_file)
                    print(f"Checkpoint saved: {len(results['rating_results'])} rating results processed")
            
        except Exception as e:
            print(f"Error processing rating annotation {ann['id']}: {str(e)}")
            continue
    
    # Process comparison annotations incrementally
    for i, ann in enumerate(comparison_annotations):
        try:
            print(f"Processing comparison annotation {i+1}/{len(comparison_annotations)}: {ann['id']}")
            
            # Get summaries and question from metadata
            if 'metadata' not in ann:
                continue
            
            metadata = ann['metadata']
            summary_a = metadata.get('summary_a_text', '')
            summary_b = metadata.get('summary_b_text', '')
            question = metadata.get('question', '')
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary_a or not summary_b or not question:
                continue
            
            # Get LLM evaluation
            llm_result = evaluator.evaluate_comparison(summary_a, summary_b, question, annotator_answer)
            human_comparisons = evaluator.data_processor.extract_human_comparisons(ann)
            
            if llm_result['status'] == 'success' and human_comparisons:
                # Create result
                unique_id = f"comparison_{ann['id']}_{evaluator.model}"
                data_source = {
                    'annotation_id': ann['id'],
                    'annotation_type': 'comparison',
                    'relative_path': f"annotation/summary-rating/annotation_output/full/{ann.get('user_id', 'unknown')}/annotated_instances.jsonl",
                    'absolute_path': str(annotation_path / ann.get('user_id', 'unknown') / 'annotated_instances.jsonl'),
                    'user_id': ann.get('user_id', 'unknown'),
                    'timestamp': ann.get('timestamp', 'unknown')
                }
                
                result = {
                    'unique_id': unique_id,
                    'annotation_id': ann['id'],
                    'model_name': evaluator.model,
                    'model_temperature': evaluator.temperature,
                    'timestamp': evaluator.timestamp if hasattr(evaluator, 'timestamp') else None,
                    'human_comparisons': human_comparisons,
                    'llm_result': llm_result,
                    'question': question,
                    'has_annotator_answer': bool(annotator_answer),
                    'annotator_answer': annotator_answer,
                    'data_source': data_source,
                    'metadata': {
                        'topic': metadata.get('topic', ''),
                        'model_a': metadata.get('model_a', ''),
                        'model_b': metadata.get('model_b', ''),
                        'comment_num': metadata.get('comment_num', 0),
                        'summary_a_length': len(summary_a) if summary_a else 0,
                        'summary_b_length': len(summary_b) if summary_b else 0
                    }
                }
                
                results['comparison_results'].append(result)
                processed_comparison_ids.add(ann['id'])
                
                # Save checkpoint every N samples
                if (i + 1) % checkpoint_interval == 0:
                    save_checkpoint(results, checkpoint_file)
                    print(f"Checkpoint saved: {len(results['comparison_results'])} comparison results processed")
            
        except Exception as e:
            print(f"Error processing comparison annotation {ann['id']}: {str(e)}")
            continue
    
    # Update metadata
    results['experiment_metadata']['n_rating_samples'] = len(results['rating_results'])
    results['experiment_metadata']['n_comparison_samples'] = len(results['comparison_results'])
    
    # Calculate correlations
    print("Calculating correlations...")
    correlations = evaluator.calculate_all_correlations(results['rating_results'], results['comparison_results'])
    results['correlations'] = correlations
    
    # Save final results
    save_checkpoint(results, final_output_file)
    print(f"Final results saved to: {final_output_file}")
    
    # Clean up checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint file cleaned up")
    
    return results


def process_missing_data(evaluator, annotation_path, results_dir, model_name, missing_rating_ids, missing_comparison_ids):
    """Process missing data for a specific model and update its result file

    Args:
        evaluator: HumanAlignedEvaluator instance
        annotation_path: Path to annotation directory
        results_dir: Path to results directory
        model_name: Name of the model
        missing_rating_ids: List of missing rating IDs
        missing_comparison_ids: List of missing comparison IDs
    
    Returns:
        bool: True if all missing data was processed successfully
    """
    processed_all = True
    # Load the existing result file
    result_file = None
    for pattern in [f"checkpoint_{model_name}.json", f"human_llm_correlation_{model_name}.json"]:
        potential_file = Path(results_dir) / pattern
        if potential_file.exists():
            result_file = potential_file
            break
    
    if not result_file:
        print(f"Error: Could not find result file for model {model_name}")
        return
    
    print(f"\nProcessing missing data for model: {model_name}")
    print(f"Using result file: {result_file}")
    
    # Load existing results
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Load all annotations
    data_processor = HumanAnnotationDataProcessor(debug=False)
    rating_annotations, comparison_annotations = data_processor.load_human_annotations(annotation_path)
    
    # Process missing rating annotations
    print(f"\nProcessing {len(missing_rating_ids)} missing rating annotations:")
    for i, rating_id in enumerate(missing_rating_ids, 1):
        print(f"\nProcessing rating {i}/{len(missing_rating_ids)}: {rating_id}")
        try:
            # Find the annotation with this ID
            ann = next((a for a in rating_annotations if a['id'] == rating_id), None)
            if not ann or 'metadata' not in ann:
                processed_all = False
                print(f"  Failed: Annotation not found or missing metadata")
                continue
            
            metadata = ann['metadata']
            summary = metadata.get('text', '')
            question = metadata.get('question', '')
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary or not question:
                processed_all = False
                print(f"  Failed: Missing summary or question")
                continue
            
            # Get LLM evaluation
            print(f"  Getting LLM evaluation...")
            llm_result = evaluator.evaluate_rating(summary, question, annotator_answer)
            human_ratings = evaluator.data_processor.extract_human_ratings(ann)
            
            if llm_result['status'] == 'success' and human_ratings:
                # Create result
                unique_id = f"rating_{rating_id}_{evaluator.model}"
                result = {
                    'unique_id': unique_id,
                    'annotation_id': rating_id,
                    'model_name': evaluator.model,
                    'model_temperature': evaluator.temperature,
                    'timestamp': evaluator.timestamp if hasattr(evaluator, 'timestamp') else None,
                    'human_ratings': human_ratings,
                    'llm_result': llm_result,
                    'question': question,
                    'has_annotator_answer': bool(annotator_answer),
                    'annotator_answer': annotator_answer,
                    'data_source': {
                        'annotation_id': rating_id,
                        'annotation_type': 'rating',
                        'relative_path': f"annotation/summary-rating/annotation_output/full/{ann.get('user_id', 'unknown')}/annotated_instances.jsonl",
                        'absolute_path': str(annotation_path / ann.get('user_id', 'unknown') / 'annotated_instances.jsonl'),
                        'user_id': ann.get('user_id', 'unknown'),
                        'timestamp': ann.get('timestamp', 'unknown')
                    },
                    'metadata': {
                        'topic': metadata.get('topic', ''),
                        'model': metadata.get('model', ''),
                        'comment_num': metadata.get('comment_num', 0),
                        'summary_length': len(summary) if summary else 0
                    }
                }
                results['rating_results'].append(result)
                
                # Save after each successful processing
                if 'checkpoint_' in result_file.name:
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    new_file = Path(results_dir) / f"human_llm_correlation_{model_name}.json"
                    with open(new_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                else:
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"  Success: Rating processed and saved")
            else:
                processed_all = False
                print(f"  Failed: Invalid LLM result or missing human ratings")
            
        except Exception as e:
            processed_all = False
            print(f"  Error: {str(e)}")
            continue
    
    # Process missing comparison annotations
    print(f"\nProcessing {len(missing_comparison_ids)} missing comparison annotations:")
    for i, comparison_id in enumerate(missing_comparison_ids, 1):
        print(f"\nProcessing comparison {i}/{len(missing_comparison_ids)}: {comparison_id}")
        try:
            # Find the annotation with this ID
            ann = next((a for a in comparison_annotations if a['id'] == comparison_id), None)
            if not ann or 'metadata' not in ann:
                processed_all = False
                print(f"  Failed: Annotation not found or missing metadata")
                continue
            
            metadata = ann['metadata']
            summary_a = metadata.get('summary_a_text', '')
            summary_b = metadata.get('summary_b_text', '')
            question = metadata.get('question', '')
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary_a or not summary_b or not question:
                processed_all = False
                print(f"  Failed: Missing summary_a, summary_b, or question")
                continue
            
            # Get LLM evaluation
            print(f"  Getting LLM evaluation...")
            llm_result = evaluator.evaluate_comparison(summary_a, summary_b, question, annotator_answer)
            human_comparisons = evaluator.data_processor.extract_human_comparisons(ann)
            
            if llm_result['status'] == 'success' and human_comparisons:
                # Create result
                unique_id = f"comparison_{comparison_id}_{evaluator.model}"
                result = {
                    'unique_id': unique_id,
                    'annotation_id': comparison_id,
                    'model_name': evaluator.model,
                    'model_temperature': evaluator.temperature,
                    'timestamp': evaluator.timestamp if hasattr(evaluator, 'timestamp') else None,
                    'human_comparisons': human_comparisons,
                    'llm_result': llm_result,
                    'question': question,
                    'has_annotator_answer': bool(annotator_answer),
                    'annotator_answer': annotator_answer,
                    'data_source': {
                        'annotation_id': comparison_id,
                        'annotation_type': 'comparison',
                        'relative_path': f"annotation/summary-rating/annotation_output/full/{ann.get('user_id', 'unknown')}/annotated_instances.jsonl",
                        'absolute_path': str(annotation_path / ann.get('user_id', 'unknown') / 'annotated_instances.jsonl'),
                        'user_id': ann.get('user_id', 'unknown'),
                        'timestamp': ann.get('timestamp', 'unknown')
                    },
                    'metadata': {
                        'topic': metadata.get('topic', ''),
                        'model_a': metadata.get('model_a', ''),
                        'model_b': metadata.get('model_b', ''),
                        'comment_num': metadata.get('comment_num', 0),
                        'summary_a_length': len(summary_a) if summary_a else 0,
                        'summary_b_length': len(summary_b) if summary_b else 0
                    }
                }
                results['comparison_results'].append(result)
                
                # Save after each successful processing
                if 'checkpoint_' in result_file.name:
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    new_file = Path(results_dir) / f"human_llm_correlation_{model_name}.json"
                    with open(new_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                else:
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"  Success: Comparison processed and saved")
            else:
                processed_all = False
                print(f"  Failed: Invalid LLM result or missing human comparisons")
            
        except Exception as e:
            processed_all = False
            print(f"  Error: {str(e)}")
            continue
    
    # Calculate correlations
    correlations = evaluator.calculate_all_correlations(results['rating_results'], results['comparison_results'])
    results['correlations'] = correlations
    
    # Save updated results
    if 'checkpoint_' in result_file.name:
        # For checkpoint files, update both checkpoint and human_llm_correlation files
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated checkpoint file: {result_file}")
        
        new_file = Path(results_dir) / f"human_llm_correlation_{model_name}.json"
        with open(new_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to: {new_file}")
    else:
        # For human_llm_correlation files, just update the same file
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated results file: {result_file}")
    
    # Check if we processed all missing data
    if not processed_all:
        print("\nWARNING: Some data could not be processed. Running coverage check again...")
        # Run coverage check again
        coverage_stats = check_coverage_status(annotation_path, results_dir)
        model_stats = coverage_stats['models'].get(f"llm_correlation_{model_name}")
        if model_stats:
            print(f"\nUpdated coverage for {model_name}:")
            print(f"Rating coverage: {model_stats['rating_coverage_percentage']:.2f}%")
            print(f"Comparison coverage: {model_stats['comparison_coverage_percentage']:.2f}%")
            if model_stats['missing_rating_ids']:
                print(f"Still missing {len(model_stats['missing_rating_ids'])} ratings")
            if model_stats['missing_comparison_ids']:
                print(f"Still missing {len(model_stats['missing_comparison_ids'])} comparisons")
    
    return processed_all

def check_coverage_status(annotation_path, results_dir):
    """Check if all annotations in the full directory are covered by the results

    Args:
        annotation_path: Path to the annotation directory
        results_dir: Path to the directory containing result files

    Returns:
        dict: Coverage statistics and missing IDs for each model
    """
    print(f"Loading annotations from: {annotation_path}")
    print(f"Checking results in: {results_dir}")
    
    # Load all annotations from full directory
    data_processor = HumanAnnotationDataProcessor(debug=False)
    rating_annotations, comparison_annotations = data_processor.load_human_annotations(annotation_path)
    
    # Get all annotation IDs
    all_rating_ids = set(ann['id'] for ann in rating_annotations)
    all_comparison_ids = set(ann['id'] for ann in comparison_annotations)
    
    print(f"\nFound {len(all_rating_ids)} rating annotations and {len(all_comparison_ids)} comparison annotations in dataset")
    
    # Initialize coverage stats for each model
    coverage_by_model = {}
    
    # Scan all result files in the results directory
    results_path = Path(results_dir)
    result_files = list(results_path.glob("*_*.json"))  # Match files like checkpoint_gpt-4o-mini.json or result_gpt-4o-mini.json
    
    print(f"\nFound {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f.name}")
    
    for result_file in result_files:
        try:
            # Extract model name from filename (everything after the first underscore)
            model_name = result_file.stem.split('_', 1)[1] if '_' in result_file.stem else result_file.stem
            file_type = result_file.stem.split('_', 1)[0] if '_' in result_file.stem else 'unknown'
            print(f"\nProcessing {result_file.name} for model: {model_name}")
            
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
                # Initialize sets for this model if not exists
                if model_name not in coverage_by_model:
                    coverage_by_model[model_name] = {
                        'processed_rating_ids': set(),
                        'processed_comparison_ids': set(),
                        'file_type': file_type
                    }
                
                # Get results from the file
                rating_results = results.get('rating_results', [])
                comparison_results = results.get('comparison_results', [])
                
                # Add IDs from this result file
                coverage_by_model[model_name]['processed_rating_ids'].update(
                    result['annotation_id'] for result in rating_results
                )
                coverage_by_model[model_name]['processed_comparison_ids'].update(
                    result['annotation_id'] for result in comparison_results
                )
                
                print(f"  Found {len(rating_results)} rating results and {len(comparison_results)} comparison results")
                
        except Exception as e:
            print(f"Error reading {result_file}: {str(e)}")
            continue
    
    # Calculate coverage statistics for each model
    coverage_stats = {
        'total_rating_annotations': len(all_rating_ids),
        'total_comparison_annotations': len(all_comparison_ids),
        'models': {}
    }
    
    for model_name, model_data in coverage_by_model.items():
        processed_rating_ids = model_data['processed_rating_ids']
        processed_comparison_ids = model_data['processed_comparison_ids']
        
        # Calculate missing IDs for this model
        missing_rating_ids = all_rating_ids - processed_rating_ids
        missing_comparison_ids = all_comparison_ids - processed_comparison_ids
        
        # Calculate coverage percentages
        rating_coverage = len(processed_rating_ids) / len(all_rating_ids) * 100 if all_rating_ids else 100
        comparison_coverage = len(processed_comparison_ids) / len(all_comparison_ids) * 100 if all_comparison_ids else 100
        
        coverage_stats['models'][model_name] = {
            'file_type': model_data['file_type'],
            'processed_rating_annotations': len(processed_rating_ids),
            'processed_comparison_annotations': len(processed_comparison_ids),
            'rating_coverage_percentage': rating_coverage,
            'comparison_coverage_percentage': comparison_coverage,
            'missing_rating_ids': sorted(list(missing_rating_ids)),
            'missing_comparison_ids': sorted(list(missing_comparison_ids))
        }
    
    return coverage_stats

def check_and_process_all_data(evaluator, annotation_path, output_path, checkpoint_interval=10):
    """Check if all data in full directory has been processed and process missing ones"""
    
    print("Checking if all data in full directory has been processed...")
    print("=" * 60)
    
    # Define checkpoint file
    checkpoint_file = output_path / f"checkpoint_{evaluator.model}.json"
    final_output_file = output_path / f"human_llm_correlation_{evaluator.model}.json"
    
    # Load existing results
    if checkpoint_file.exists():
        print(f"Loading existing checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed_rating_ids = set(result['annotation_id'] for result in results.get('rating_results', []))
        processed_comparison_ids = set(result['annotation_id'] for result in results.get('comparison_results', []))
        print(f"Found {len(processed_rating_ids)} rating and {len(processed_comparison_ids)} comparison results already processed")
    else:
        print("No existing checkpoint found, starting fresh")
        results = {
            "experiment_metadata": {
                "model": evaluator.model,
                "temperature": evaluator.temperature,
                "timestamp": evaluator.timestamp if hasattr(evaluator, 'timestamp') else None,
                "n_rating_samples": 0,
                "n_comparison_samples": 0
            },
            "rating_results": [],
            "comparison_results": [],
            "correlations": {}
        }
        processed_rating_ids = set()
        processed_comparison_ids = set()
    
    # Load all annotations from full directory
    print(f"Loading all annotations from {annotation_path}")
    rating_annotations, comparison_annotations = evaluator.data_processor.load_human_annotations(annotation_path)
    
    print(f"Total rating annotations in full directory: {len(rating_annotations)}")
    print(f"Total comparison annotations in full directory: {len(comparison_annotations)}")
    
    # Find missing annotations
    all_rating_ids = set(ann['id'] for ann in rating_annotations)
    all_comparison_ids = set(ann['id'] for ann in comparison_annotations)
    
    missing_rating_ids = all_rating_ids - processed_rating_ids
    missing_comparison_ids = all_comparison_ids - processed_comparison_ids
    
    print(f"Missing rating annotations: {len(missing_rating_ids)}")
    print(f"Missing comparison annotations: {len(missing_comparison_ids)}")
    
    if len(missing_rating_ids) == 0 and len(missing_comparison_ids) == 0:
        print("All data has been processed!")
        return results
    
    # Process missing rating annotations
    missing_rating_annotations = [ann for ann in rating_annotations if ann['id'] in missing_rating_ids]
    print(f"Processing {len(missing_rating_annotations)} missing rating annotations...")
    
    for i, ann in enumerate(missing_rating_annotations):
        try:
            print(f"Processing rating annotation {i+1}/{len(missing_rating_annotations)}: {ann['id']}")
            
            # Get summary and question from metadata
            if 'metadata' not in ann:
                print(f"  Skipping {ann['id']}: No metadata")
                continue
            
            metadata = ann['metadata']
            summary = metadata.get('text', '')
            question = metadata.get('question', '')
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary or not question:
                print(f"  Skipping {ann['id']}: Missing summary or question")
                continue
            
            # Get LLM evaluation
            llm_result = evaluator.evaluate_rating(summary, question, annotator_answer)
            human_ratings = evaluator.data_processor.extract_human_ratings(ann)
            
            if llm_result['status'] == 'success' and human_ratings:
                # Create result
                unique_id = f"rating_{ann['id']}_{evaluator.model}"
                result = {
                    'unique_id': unique_id,
                    'annotation_id': ann['id'],
                    'data_source': str(annotation_path),
                    'summary': summary,
                    'question': question,
                    'human_ratings': human_ratings,
                    'llm_result': llm_result,
                    'model': evaluator.model
                }
                results['rating_results'].append(result)
                processed_rating_ids.add(ann['id'])
                print(f"  Successfully processed rating {ann['id']}")
            else:
                print(f"  Failed to process rating {ann['id']}: {llm_result.get('error', 'Unknown error')}")
            
            # Save checkpoint every N samples
            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(results, checkpoint_file)
                print(f"  Checkpoint saved: {len(results['rating_results'])} rating results processed")
        
        except Exception as e:
            print(f"  Error processing rating annotation {ann['id']}: {str(e)}")
            continue
    
    # Process missing comparison annotations
    missing_comparison_annotations = [ann for ann in comparison_annotations if ann['id'] in missing_comparison_ids]
    print(f"Processing {len(missing_comparison_annotations)} missing comparison annotations...")
    
    for i, ann in enumerate(missing_comparison_annotations):
        try:
            print(f"Processing comparison annotation {i+1}/{len(missing_comparison_annotations)}: {ann['id']}")
            
            # Get comparison data from metadata
            if 'metadata' not in ann:
                print(f"  Skipping {ann['id']}: No metadata")
                continue
            
            metadata = ann['metadata']
            summary_a = metadata.get('summary_a', '')
            summary_b = metadata.get('summary_b', '')
            question = metadata.get('question', '')
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary_a or not summary_b or not question:
                print(f"  Skipping {ann['id']}: Missing summary or question")
                continue
            
            # Get LLM evaluation
            llm_result = evaluator.evaluate_comparison(summary_a, summary_b, question, annotator_answer)
            human_comparisons = evaluator.data_processor.extract_human_comparisons(ann)
            
            if llm_result['status'] == 'success' and human_comparisons:
                # Create result
                unique_id = f"comparison_{ann['id']}_{evaluator.model}"
                result = {
                    'unique_id': unique_id,
                    'annotation_id': ann['id'],
                    'data_source': str(annotation_path),
                    'summary_a': summary_a,
                    'summary_b': summary_b,
                    'question': question,
                    'human_comparisons': human_comparisons,
                    'llm_result': llm_result,
                    'model': evaluator.model
                }
                results['comparison_results'].append(result)
                processed_comparison_ids.add(ann['id'])
                print(f"  Successfully processed comparison {ann['id']}")
            else:
                print(f"  Failed to process comparison {ann['id']}: {llm_result.get('error', 'Unknown error')}")
            
            # Save checkpoint every N samples
            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(results, checkpoint_file)
                print(f"  Checkpoint saved: {len(results['comparison_results'])} comparison results processed")
        
        except Exception as e:
            print(f"  Error processing comparison annotation {ann['id']}: {str(e)}")
            continue
    
    # Update metadata
    results['experiment_metadata']['n_rating_samples'] = len(results['rating_results'])
    results['experiment_metadata']['n_comparison_samples'] = len(results['comparison_results'])
    
    # Calculate correlations
    print("Calculating correlations...")
    correlations = evaluator.calculate_all_correlations(results['rating_results'], results['comparison_results'])
    results['correlations'] = correlations
    
    # Save final results
    save_checkpoint(results, final_output_file)
    print(f"Final results saved to: {final_output_file}")
    
    # Clean up checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint file cleaned up")
    
    print("=" * 60)
    print("COMPLETION SUMMARY")
    print("=" * 60)
    print(f"Total rating results: {len(results['rating_results'])}")
    print(f"Total comparison results: {len(results['comparison_results'])}")
    print(f"Total processed: {len(results['rating_results']) + len(results['comparison_results'])}")
    
    return results


def save_checkpoint(results, file_path):
    """Save results to checkpoint file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run the correlation experiment"""
    parser = argparse.ArgumentParser(description='Human-LLM Correlation Experiment')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help='LLM model to use for evaluation')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--debug', action='store_true', default=DEFAULT_DEBUG,
                       help='Debug mode - only process first 3 samples')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help='Temperature for model generation')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_PATH),
                       help='Output directory for results')
    parser.add_argument('--results-dir', type=str, 
                       default='/ibex/project/c2328/LLMs-Scalable-Deliberation/results/eval_llm_human_correlation',
                       help='Directory containing result files')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode with detailed debugging')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N processed samples')
    parser.add_argument('--check-all', action='store_true',
                       help='Check if all data in full directory has been processed and process missing ones')
    parser.add_argument('--check-coverage', action='store_true',
                       help='Check coverage of results against full annotation directory')
    parser.add_argument('--show-missing-ids', action='store_true',
                       help='Show missing annotation IDs in coverage report')
    parser.add_argument('--process-missing', action='store_true',
                       help='Process missing annotations and update result files')
    parser.add_argument('--target-model', type=str,
                       help='Only process missing data for this specific model (e.g., gpt-5-nano)')
    
    args = parser.parse_args()
    
    # Configuration
    annotation_path = DEFAULT_ANNOTATION_PATH
    output_path = Path(args.output_dir)
    
    # Override sample size if debug mode or demo mode
    sample_size = 3 if (args.debug or args.demo) else args.sample_size
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator with debug mode
    evaluator = HumanAlignedEvaluator(
        model=args.model,
        temperature=args.temperature,
        debug=args.debug or args.demo
    )
    
    # Run experiment
    print("Starting Human-LLM Correlation Experiment")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Sample Size: {sample_size}")
    print(f"Debug Mode: {args.debug or args.demo}")
    print("=" * 60)
    
    if args.check_coverage or args.process_missing:
        # Check coverage of results against full annotation directory
        coverage_stats = check_coverage_status(annotation_path, args.results_dir)
        
        print("\n" + "=" * 80)
        print("COVERAGE STATUS SUMMARY")
        print("=" * 80)
        print(f"Total annotations in dataset:")
        print(f"  - Rating annotations: {coverage_stats['total_rating_annotations']}")
        print(f"  - Comparison annotations: {coverage_stats['total_comparison_annotations']}")
        
        print("\n" + "=" * 80)
        print("COVERAGE BY MODEL")
        print("=" * 80)
        
        # Sort models by name
        for model_name in sorted(coverage_stats['models'].keys()):
            model_stats = coverage_stats['models'][model_name]
            print(f"\nModel: {model_name} ({model_stats['file_type']})")
            print("-" * 40)
            
            # Rating statistics
            print(f"Rating annotations:")
            print(f"  - Processed: {model_stats['processed_rating_annotations']} / {coverage_stats['total_rating_annotations']}")
            print(f"  - Coverage: {model_stats['rating_coverage_percentage']:.2f}%")
            if model_stats['missing_rating_ids']:
                print(f"  - Missing: {len(model_stats['missing_rating_ids'])} ratings")
            
            # Comparison statistics
            print(f"\nComparison annotations:")
            print(f"  - Processed: {model_stats['processed_comparison_annotations']} / {coverage_stats['total_comparison_annotations']}")
            print(f"  - Coverage: {model_stats['comparison_coverage_percentage']:.2f}%")
            if model_stats['missing_comparison_ids']:
                print(f"  - Missing: {len(model_stats['missing_comparison_ids'])} comparisons")
            
            # Show missing IDs if requested
            if args.show_missing_ids and (model_stats['missing_rating_ids'] or model_stats['missing_comparison_ids']):
                print("\nMissing IDs:")
                if model_stats['missing_rating_ids']:
                    print(f"  Missing Rating IDs ({len(model_stats['missing_rating_ids'])}):")
                    for id_ in model_stats['missing_rating_ids'][:10]:  # Show only first 10
                        print(f"    - {id_}")
                    if len(model_stats['missing_rating_ids']) > 10:
                        print(f"    ... and {len(model_stats['missing_rating_ids']) - 10} more")
                
                if model_stats['missing_comparison_ids']:
                    print(f"\n  Missing Comparison IDs ({len(model_stats['missing_comparison_ids'])}):")
                    for id_ in model_stats['missing_comparison_ids'][:10]:  # Show only first 10
                        print(f"    - {id_}")
                    if len(model_stats['missing_comparison_ids']) > 10:
                        print(f"    ... and {len(model_stats['missing_comparison_ids']) - 10} more")
            
            # Process missing data if requested
            if args.process_missing and (model_stats['missing_rating_ids'] or model_stats['missing_comparison_ids']):
                # Extract model name without prefix
                clean_model_name = model_name.replace('llm_correlation_', '')
                
                # Skip if target model is specified and doesn't match
                if args.target_model and clean_model_name != args.target_model:
                    print(f"\nSkipping model {clean_model_name} (not target model)")
                    continue
                
                # Initialize evaluator with the correct model
                evaluator = HumanAlignedEvaluator(
                    model=clean_model_name,
                    temperature=args.temperature,
                    debug=args.debug
                )
                
                # Process missing data
                success = process_missing_data(
                    evaluator=evaluator,
                    annotation_path=annotation_path,
                    results_dir=args.results_dir,
                    model_name=clean_model_name,
                    missing_rating_ids=model_stats['missing_rating_ids'],
                    missing_comparison_ids=model_stats['missing_comparison_ids']
                )
                
                if not success:
                    print(f"\nWARNING: Not all missing data was processed for {model_name}")
                    print("You may want to run the process again to try processing the remaining data")
        
    elif args.demo:
        run_demo_mode(evaluator, annotation_path, output_path)
    elif args.check_all:
        # Check and process all data in full directory
        results = check_and_process_all_data(
            evaluator=evaluator,
            annotation_path=annotation_path,
            output_path=output_path,
            checkpoint_interval=args.checkpoint_interval
        )
    else:
        # Run with incremental processing and checkpoint support
        results = run_incremental_correlation_experiment(
            evaluator=evaluator,
            annotation_path=annotation_path,
            output_path=output_path,
            sample_size=sample_size,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval
        )
    
    # Print summary for regular and check-all modes (not for check-coverage mode)
    if not args.demo and not args.check_coverage:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        if results and 'correlations' in results:
            for metric, corr in results['correlations'].items():
                if isinstance(corr, dict) and 'pearson_r' in corr:
                    print(f"\n{metric}:")
                    print(f"  Pearson r: {corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f})")
                    print(f"  Spearman r: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})")
                    print(f"  Cohen's kappa: {corr['cohen_kappa']:.3f}")
                    print(f"  MAE: {corr['mae']:.3f}")
                    print(f"  N samples: {corr['n_samples']}")


if __name__ == "__main__":
    main()
