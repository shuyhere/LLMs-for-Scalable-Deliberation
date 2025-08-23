#!/usr/bin/env python3
"""
Test script for comment summarization using LLM models.
Reads comments from JSON files and generates summaries.
"""

import sys
import os
from pathlib import Path


# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_summarization.summarizer import CommentSummarizer
from utils.data_loader import read_json_dataset


def main():
    """Main function to test comment summarization."""
    
    json_file = "datasets/protest.json"
    
    if not os.path.exists(json_file):
        print(f"JSON file '{json_file}' not found.")
        return
    
    print(f"Using JSON file: {json_file}")
    
    # Read comments from JSON
    question, formatted_comments = read_json_dataset(json_file)
    
    if not formatted_comments:
        print("No comments found in JSON file.")
        return
    
    print(f"\nQuestion: {question}")
    print(f"Formatted comments length: {len(formatted_comments)} characters")
    
    # Initialize summarizer
    summarizer = CommentSummarizer(model="gpt-4o-mini")
    
    print("\n=== GENERATING SUMMARIES ===")
    
    # Generate topic modeling summary
    print("\n1. Topic Modeling Summary:")
    topic_summary = summarizer.summarize_topic_modeling(formatted_comments)
    print(topic_summary)
    
    # Generate main points summary
    print("\n2. Main Points Summary:")
    main_summary = summarizer.summarize_main_points(formatted_comments)
    print(main_summary)
    
    # Generate custom summary
    print("\n3. Custom Summary:")
    custom_summary = summarizer.summarize_with_custom_prompt(
        comments=formatted_comments,
        custom_system_prompt="You are an expert data analyst specializing in public opinion analysis.",
        custom_user_prompt="Analyze the sentiment and key themes in these comments. Focus on areas of agreement and disagreement: {comments}"
    )
    print(custom_summary)
    
    # Save results to file
    output_file = f"summarization_results_{Path(json_file).stem}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("COMMENT SUMMARIZATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {json_file}\n")
        f.write(f"Question: {question}\n\n")
        f.write(f"Formatted comments:\n{formatted_comments}\n\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. TOPIC MODELING SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(topic_summary)
        f.write("\n\n")
        
        f.write("2. MAIN POINTS SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(main_summary)
        f.write("\n\n")
        
        f.write("3. CUSTOM SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(custom_summary)
        f.write("\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
