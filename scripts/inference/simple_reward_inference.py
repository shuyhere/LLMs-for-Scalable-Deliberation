
#!/usr/bin/env python3
"""
Simple Reward Model Inference Script

This script loads a trained reward model and performs inference on test data.
It preserves all original fields from the test data and adds reward scores to chosen and rejected responses.

Usage:
python scripts/inference/simple_reward_inference.py \
    --model_path outputs/reward_models/informativeness_reward_model_lora \
    --test_data_path datasets/rl_datasets/trl_format/informativeness_trl_dataset/test.jsonl \
    --output_path results/reward_model_inference/informativeness_scores.jsonl

Output format:
- Preserves all original fields from input data
- Adds reward_score to chosen and rejected conversations
- Adds score_difference and preference_correct fields
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
from trl import setup_chat_format
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reward_model(model_path: str) -> Tuple[Any, Any]:
    """Load the trained reward model and tokenizer"""
    logger.info(f"Loading reward model from {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Setup chat format if needed
        try:
            model, tokenizer = setup_chat_format(model, tokenizer)
        except Exception as e:
            logger.warning(f"Could not setup chat format: {e}")
        
        model.eval()
        logger.info("Reward model loaded successfully")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load reward model: {e}")
        raise

def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format conversation messages into a single string"""
    formatted_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            formatted_text += f"Human: {content}\n\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}\n\n"
    return formatted_text.strip()

def compute_single_score(model, tokenizer, conversation: str, max_length: int = 8192) -> float:
    """Compute reward score for a single conversation"""
    # Tokenize
    inputs = tokenizer(
        conversation,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze(-1).item()
    
    return score

def load_test_data(test_data_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL file"""
    logger.info(f"Loading test data from {test_data_path}")
    
    test_data = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(test_data)} test samples")
    return test_data

def process_test_data(model, tokenizer, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process test data and compute reward scores"""
    results = []
    
    for sample in tqdm(test_data, desc="Processing test samples"):
        # Format chosen and rejected conversations
        chosen_conversation = format_conversation(sample["chosen"])
        rejected_conversation = format_conversation(sample["rejected"])
        
        # Compute scores for both
        chosen_score = compute_single_score(model, tokenizer, chosen_conversation)
        rejected_score = compute_single_score(model, tokenizer, rejected_conversation)
        
        # Create result entry preserving all original fields and adding reward scores
        result = sample.copy()  # Keep all original fields
        
        # Add reward scores to chosen and rejected (they are lists, so we add to the last message)
        result["chosen"] = sample["chosen"].copy()
        if result["chosen"] and len(result["chosen"]) > 0:
            # Add reward_score to the last message in chosen (assistant response)
            result["chosen"][-1]["reward_score"] = chosen_score
        
        result["rejected"] = sample["rejected"].copy()
        if result["rejected"] and len(result["rejected"]) > 0:
            # Add reward_score to the last message in rejected (assistant response)
            result["rejected"][-1]["reward_score"] = rejected_score
        
        # Add additional computed fields
        result["score_difference"] = chosen_score - rejected_score
        result["preference_correct"] = chosen_score > rejected_score
        
        results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSONL file"""
    logger.info(f"Saving results to {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    chosen_scores = [r["chosen"][-1]["reward_score"] for r in results if r["chosen"] and len(r["chosen"]) > 0 and "reward_score" in r["chosen"][-1]]
    rejected_scores = [r["rejected"][-1]["reward_score"] for r in results if r["rejected"] and len(r["rejected"]) > 0 and "reward_score" in r["rejected"][-1]]
    correct_preferences = sum(1 for r in results if r.get("preference_correct", False))
    
    logger.info(f"Total samples processed: {len(results)}")
    logger.info(f"Average chosen score: {sum(chosen_scores)/len(chosen_scores):.4f}")
    logger.info(f"Average rejected score: {sum(rejected_scores)/len(rejected_scores):.4f}")
    logger.info(f"Score difference (chosen - rejected): {sum(chosen_scores)/len(chosen_scores) - sum(rejected_scores)/len(rejected_scores):.4f}")
    logger.info(f"Preference accuracy: {correct_preferences}/{len(results)} ({correct_preferences/len(results)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Simple Reward Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained reward model')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to the test data JSONL file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the inference results')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_reward_model(args.model_path)
    
    # Load test data
    test_data = load_test_data(args.test_data_path)
    
    # Process test data
    results = process_test_data(model, tokenizer, test_data)
    
    # Save results
    save_results(results, args.output_path)
    
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()
