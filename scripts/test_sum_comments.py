#!/usr/bin/env python3
"""
Script to test comment summarization by reading from CSV and running summarization.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_summarization.summarizer import CommentSummarizer


def read_comments_from_csv(csv_path: str) -> tuple[str, str]:
    """
    Read CSV, extract all questions and their comments, and return the first question's comments.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        A tuple (selected_question, formatted_comments)
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        # Validate required columns
        if 'question' not in df.columns:
            raise ValueError("CSV must contain a 'question' column")
        if 'comment' not in df.columns and 'text' not in df.columns:
            raise ValueError("CSV must contain a 'comment' or 'text' column")

        # Get all unique questions (preserving order)
        unique_questions = df['question'].astype(str).dropna().drop_duplicates().tolist()
        print(f"Found {len(unique_questions)} unique questions. Using the first one.")

        # Optionally list questions with counts
        counts = df.groupby('question').size().reset_index(name='count')
        for idx, row in counts.iterrows():
            # Show a short preview of the question
            q_preview = (row['question'][:80] + '...') if len(str(row['question'])) > 80 else row['question']
            print(f"[{idx}] {q_preview}  (comments: {row['count']})")

        if not unique_questions:
            print("No questions found in CSV")
            return "", ""

        selected_question = unique_questions[0]
        df_filtered = df[df['question'] == selected_question]
        print(f"Selected question has {len(df_filtered)} comments")

        # Format comments for summarization
        formatted_comments = []
        for idx, row in df_filtered.reset_index(drop=True).iterrows():
            comment_text = row['comment'] if 'comment' in df.columns else row['text']
            comment_parts = [f"Comment {idx + 1}: {comment_text}"]

            # Optional: include any available voting fields if present
            if 'group_0_agree' in df.columns and 'group_0_disagree' in df.columns:
                agree_0 = row.get('group_0_agree', 0)
                disagree_0 = row.get('group_0_disagree', 0)
                comment_parts.append(f"Group 0: {agree_0}% agree, {disagree_0}% disagree")

            if 'group_1_agree' in df.columns and 'group_1_disagree' in df.columns:
                agree_1 = row.get('group_1_agree', 0)
                disagree_1 = row.get('group_1_disagree', 0)
                comment_parts.append(f"Group 1: {agree_1}% agree, {disagree_1}% disagree")

            formatted_comments.append(". ".join(comment_parts))

        return selected_question, "\n".join(formatted_comments)

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return "", ""


def main():
    """Main function to run comment summarization."""
    
    csv_path = "datasets/demo_datasets/deliberation_comments_aggregated.csv"
    
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    selected_question, comments = read_comments_from_csv(csv_path)
    
    if not comments:
        print("No comments found or error reading CSV")
        return
    
    print("\nSelected question:")
    print(selected_question)
    print(f"\nFormatted comments (first 500 chars):\n{comments[:500]}...")
    
    summarizer = CommentSummarizer(model="gpt-4o-mini")
    
    print("\n=== RUNNING SUMMARIZATION ===")
    
    print("\n--- Topic Modeling Summary ---")
    topic_summary = summarizer.summarize_topic_modeling(comments)
    print(topic_summary)
    
    print("\n--- Main Points Summary ---")
    main_summary = summarizer.summarize_main_points(comments)
    print(main_summary)
    
    print("\n--- Custom Analysis Summary ---")
    custom_summary = summarizer.summarize_with_custom_prompt(
        comments=comments,
        custom_system_prompt="You are an expert social scientist analyzing deliberation data.",
        custom_user_prompt="Analyze the key themes and patterns in these deliberation comments: {comments}"
    )
    print(custom_summary)


if __name__ == "__main__":
    main()
