"""
Data loading utilities for different file formats.
Supports JSON and CSV files with various data structures.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any


def read_json_dataset(json_file: str) -> Tuple[str, List[str]]:
    """
    Read dataset from JSON file.
    
    Expected JSON format:
    {
        "question": "question text",
        "comments": [
            {"index": 0, "comment": "comment text"},
            {"index": 1, "comment": "comment text"},
            ...
        ]
    }
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        Tuple of (question, comments_list)
    """
    question = ""
    comments = []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Extract question
            question = data.get("question", "").strip()
            
            # Extract comments
            comments_data = data.get("comments", [])
            for comment_obj in comments_data:
                if isinstance(comment_obj, dict) and "comment" in comment_obj:
                    comment_text = comment_obj["comment"].strip()
                    if comment_text:  # Skip empty comments
                        comments.append(comment_text)
                        
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        return "", []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return "", []
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return "", []
    
    return question, comments


def read_csv_dataset(csv_file: str) -> Tuple[str, List[str]]:
    """
    Read dataset from CSV file.
    
    Expected CSV format:
    - First row: summary in the first column
    - Subsequent rows: comments in the first column
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Tuple of (summary, comments_list)
    """
    summary = ""
    comments = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            # Read first row as summary
            first_row = next(reader, None)
            if first_row and first_row[0]:
                summary = first_row[0].strip()
            
            # Read remaining rows as comments
            for row in reader:
                if row and row[0]:
                    comment = row[0].strip()
                    if comment:  # Skip empty comments
                        comments.append(comment)
                        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return "", []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return "", []
    
    return summary, comments


def read_comments_from_csv(csv_path: str) -> Tuple[str, str]:
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


def detect_file_format(file_path: str) -> str:
    """
    Detect the format of a data file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File format: 'json', 'csv', or 'unknown'
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.json':
        return 'json'
    elif extension == '.csv':
        return 'csv'
    else:
        return 'unknown'


def load_dataset(file_path: str) -> Tuple[str, List[str], str]:
    """
    Universal dataset loader that automatically detects file format and loads data.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Tuple of (question/summary, comments_list, file_format)
    """
    file_format = detect_file_format(file_path)
    
    if file_format == 'json':
        question, comments = read_json_dataset(file_path)
        return question, comments, 'json'
    elif file_format == 'csv':
        summary, comments = read_csv_dataset(file_path)
        return summary, comments, 'csv'
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Please use .json or .csv files.")
