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
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode with detailed debugging')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N processed samples')
    
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
    
    if args.demo:
        run_demo_mode(evaluator, annotation_path, output_path)
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
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        if 'correlations' in results:
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
