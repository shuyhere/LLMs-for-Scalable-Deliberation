#!/usr/bin/env python3
"""
Example script showing how to use DebertaEvaluator to evaluate comment-summary pairs.

Usage:
    python scripts/evaluate_with_deberta.py --model /path/to/trained/model --data /path/to/data.jsonl
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_evaluation.evaluator import DebertaEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate comment-summary pairs with trained DeBERTa model")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained DeBERTa model directory")
    parser.add_argument("--data", type=str, 
                       help="Path to JSONL file with evaluation data (optional)")
    parser.add_argument("--question", type=str, 
                       help="Question for single evaluation (optional)")
    parser.add_argument("--comment", type=str,
                       help="Comment for single evaluation (optional)")
    parser.add_argument("--summary", type=str,
                       help="Summary for single evaluation (optional)")
    parser.add_argument("--output", type=str,
                       help="Output file to save results (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return
    
    # Initialize evaluator
    try:
        print(f"Loading DeBERTa model from {model_path}")
        evaluator = DebertaEvaluator(str(model_path), device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    results = []
    
    if args.data:
        # Evaluate from JSONL file
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: Data file {data_path} does not exist")
            return
        
        print(f"Evaluating data from {data_path}")
        results = evaluator.evaluate_from_jsonl(str(data_path), max_length=args.max_length)
        
    elif args.question and args.comment and args.summary:
        # Single evaluation
        print("Evaluating single item")
        result = evaluator.evaluate_single(
            args.question, 
            args.comment, 
            args.summary, 
            max_length=args.max_length
        )
        results = [result]
        
    else:
        # Example evaluation with sample data
        print("No data provided, using sample data")
        sample_data = [
            {
                "question": "What are your thoughts on artificial intelligence development?",
                "comment": "I believe AI will revolutionize healthcare and education, but we need strong regulations to prevent misuse.",
                "summary": "The discussion on AI development revealed mixed perspectives. Some participants emphasized the potential benefits in healthcare and education, while others expressed concerns about the need for proper regulation and oversight to ensure responsible development and deployment of AI technologies."
            },
            {
                "question": "What is your view on climate change policies?",
                "comment": "We need immediate action on carbon emissions reduction.",
                "summary": "Participants discussed various climate policy approaches with emphasis on urgent action."
            }
        ]
        
        results = evaluator.evaluate_batch(sample_data, max_length=args.max_length)
    
    # Print results
    print(f"\n=== EVALUATION RESULTS ({len(results)} items) ===")
    
    for i, result in enumerate(results):
        print(f"\nItem {i+1}:")
        print(f"  Status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            print(f"  Question: {result.get('question', 'N/A')}")
            print(f"  Comment: {result.get('comment', 'N/A')[:100]}...")
            print(f"  Summary: {result.get('summary', 'N/A')[:100]}...")
            print("  Predictions:")
            predictions = result.get('predictions', {})
            for key, value in predictions.items():
                if value is not None:
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: N/A")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Print statistics
    print(f"\n=== EVALUATION STATISTICS ===")
    stats = evaluator.get_evaluation_statistics(results)
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
