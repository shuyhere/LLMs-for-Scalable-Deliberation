#!/usr/bin/env python3
"""
Test script for evaluating summaries with LLMs.
Reads summaries and comments from JSON files and evaluates comment representation.
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_evaluation.evaluator import SummaryEvaluator
from llm_summarization.summarizer import CommentSummarizer
from utils.data_loader import load_dataset, detect_file_format


def generate_summary_from_comments(comments: list[str], model: str = "gpt-4o-mini") -> str:
    """
    Generate a summary from comments using the CommentSummarizer.
    
    Args:
        comments: List of comments to summarize
        model: Model to use for summarization
        
    Returns:
        Generated summary
    """
    try:
        # Format comments for summarization
        formatted_comments = "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(comments)])
        
        # Generate summary
        summarizer = CommentSummarizer(model=model)
        summary = summarizer.summarize_main_points(formatted_comments)
        
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error: Could not generate summary from comments."


def main():
    """Main function to test summary evaluation."""
    
    input_file = "datasets/protest.json"  # Default to protest dataset
    
    print(f"Using input file: {input_file}")
    
    # Check file format and load data
    file_format = detect_file_format(input_file)
    if file_format == "unknown":
        print("Unsupported file format. Please use .json or .csv files.")
        return
    
    try:
        # Load dataset using universal loader
        question, comments, detected_format = load_dataset(input_file)
        data_type = detected_format.upper()
        print(f"Question: {question}")
        
        print(f"Comments: {comments}")
        
        if not comments:
            print("No comments found in file.")
            return
        
        print(f"Data type: {data_type}")
        
        # Always generate summary from comments
        print("\nGenerating summary from comments...")
        summary = generate_summary_from_comments(comments)
        print(f"Generated summary: {summary}")
        
        evaluator = SummaryEvaluator(model="gpt-4o-mini")
        
        print("\n=== EVALUATING COMMENT REPRESENTATION ===")
        
        # Evaluate all comments
        comments = comments[:10]
        results = evaluator.evaluate_multiple_comments(summary, comments)
        
        # Display results
        for result in results:
            print(f"\nComment {result['comment_index'] + 1}: {result['comment'][:80]}...")
            print(f"Score: {result['score']}")
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Response: {result['evaluation_response']}")
        
        # Display statistics
        print("\n=== EVALUATION STATISTICS ===")
        stats = evaluator.get_evaluation_statistics(results)
        for key, value in stats.items():
            if key == 'score_distribution':
                print(f"{key}:")
                for score, count in value.items():
                    print(f"  Score {score}: {count} comments")
            else:
                print(f"{key}: {value}")
        
        # Save results to file
        output_file = f"evaluation_results_{Path(input_file).stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SUMMARY EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Data type: {data_type}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Generated Summary: {summary}\n\n")
            f.write(f"Total Comments: {len(comments)}\n\n")
            
            for result in results:
                f.write(f"Comment {result['comment_index'] + 1}: {result['comment']}\n")
                f.write(f"Score: {result['score']}\n")
                f.write(f"Status: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"Response: {result['evaluation_response']}\n")
                f.write("-" * 40 + "\n\n")
            
            f.write("STATISTICS\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                if key == 'score_distribution':
                    f.write(f"{key}:\n")
                    for score, count in value.items():
                        f.write(f"  Score {score}: {count} comments\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
