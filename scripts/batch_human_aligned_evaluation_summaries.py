#!/usr/bin/env python3
"""
Script to run human-LLM correlation experiments
"""

import argparse
import logging
from pathlib import Path
import sys

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
        results = evaluator.run_correlation_experiment(
            annotation_path=annotation_path,
            output_path=output_path,
            sample_size=sample_size
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
