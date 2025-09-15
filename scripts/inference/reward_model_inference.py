#!/usr/bin/env python3
"""
Reward Model Inference Script

This script loads a trained reward model and performs inference on test data.
It outputs scores for both chosen and rejected responses.

Usage:
python scripts/inference/reward_model_inference.py \
    --model_path outputs/reward_models/informativeness_reward_model_lora \
    --test_data_path datasets/rl_datasets/trl_format/informativeness_trl_dataset/test.jsonl \
    --output_path results/reward_model_inference/informativeness_scores.jsonl \
    --batch_size 4 \
    --max_length 8192
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from trl import setup_chat_format
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_reward_model(model_path: str, device: str = "auto") -> Tuple[Any, Any]:
    """Load the trained reward model and tokenizer"""
    logger.info(f"Loading reward model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    # Setup chat format if needed
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    model.eval()
    logger.info("Reward model loaded successfully")
    
    return model, tokenizer

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

def tokenize_batch(tokenizer, conversations: List[str], max_length: int = 8192) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of conversations"""
    return tokenizer(
        conversations,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def compute_reward_scores(model, tokenizer, conversations: List[str], batch_size: int = 4, max_length: int = 8192) -> List[float]:
    """Compute reward scores for a list of conversations"""
    scores = []
    
    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Computing reward scores"):
        batch_conversations = conversations[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenize_batch(tokenizer, batch_conversations, max_length)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            # Handle single item case
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            
            scores.extend(batch_scores)
    
    return scores

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

def process_test_data(model, tokenizer, test_data: List[Dict[str, Any]], batch_size: int = 4, max_length: int = 8192) -> List[Dict[str, Any]]:
    """Process test data and compute reward scores"""
    results = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Processing test samples")):
        # Format chosen and rejected conversations
        chosen_conversation = format_conversation(sample["chosen"])
        rejected_conversation = format_conversation(sample["rejected"])
        
        # Compute scores for both
        conversations = [chosen_conversation, rejected_conversation]
        scores = compute_reward_scores(model, tokenizer, conversations, batch_size, max_length)
        
        # Create result entry
        result = {
            "sample_id": i,
            "chosen": {
                "conversation": sample["chosen"],
                "formatted_text": chosen_conversation,
                "reward_score": scores[0]
            },
            "rejected": {
                "conversation": sample["rejected"],
                "formatted_text": rejected_conversation,
                "reward_score": scores[1]
            },
            "score_difference": scores[0] - scores[1],  # chosen - rejected
            "preference_correct": scores[0] > scores[1]  # True if model prefers chosen
        }
        
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
    
    # Also save a summary
    summary_path = output_path.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples": len(results),
        "correct_preferences": sum(1 for r in results if r["preference_correct"]),
        "accuracy": sum(1 for r in results if r["preference_correct"]) / len(results),
        "avg_chosen_score": sum(r["chosen"]["reward_score"] for r in results) / len(results),
        "avg_rejected_score": sum(r["rejected"]["reward_score"] for r in results) / len(results),
        "avg_score_difference": sum(r["score_difference"] for r in results) / len(results)
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Accuracy: {summary['accuracy']:.4f}")
    logger.info(f"Average chosen score: {summary['avg_chosen_score']:.4f}")
    logger.info(f"Average rejected score: {summary['avg_rejected_score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Reward Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained reward model')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to the test data JSONL file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the inference results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=8192,
                       help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_reward_model(args.model_path, args.device)
    
    # Load test data
    test_data = load_test_data(args.test_data_path)
    
    # Process test data
    results = process_test_data(model, tokenizer, test_data, args.batch_size, args.max_length)
    
    # Save results
    save_results(results, args.output_path)
    
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()
