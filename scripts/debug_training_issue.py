#!/usr/bin/env python3
"""
Debug script to identify why training is not working.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from transformers import AutoTokenizer

def analyze_dataset():
    """Analyze the dataset to identify potential issues."""
    print("=== Dataset Analysis ===")
    
    dataset_path = "/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_dataset/comparison_sft_dataset.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    
    # Analyze first few examples
    print("\n--- Sample Examples ---")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  Prompt length: {len(example['prompt'])}")
        print(f"  Completion: {example['completion']}")
        print(f"  Metadata scores: {example['metadata']['comparison_scores']}")
        
        # Check if prompt is properly formatted
        if "Perspective Representation:" not in example['prompt']:
            print("  ❌ WARNING: Prompt format seems incorrect")
        else:
            print("  ✅ Prompt format looks correct")
    
    # Analyze label distribution
    print("\n--- Label Distribution Analysis ---")
    all_scores = []
    for example in dataset:
        scores = example['metadata']['comparison_scores']
        all_scores.extend(scores)
    
    scores_array = np.array(all_scores)
    print(f"Total scores: {len(scores_array)}")
    print(f"Score distribution:")
    print(f"  1.0 (Summary A better): {np.sum(scores_array == 1.0)} ({np.mean(scores_array == 1.0)*100:.1f}%)")
    print(f"  2.0 (Summary B better): {np.sum(scores_array == 2.0)} ({np.mean(scores_array == 2.0)*100:.1f}%)")
    
    # Check for class imbalance
    if np.mean(scores_array == 1.0) > 0.8 or np.mean(scores_array == 2.0) > 0.8:
        print("❌ WARNING: Severe class imbalance detected!")
    else:
        print("✅ Class balance looks reasonable")
    
    return True

def test_tokenization():
    """Test if tokenization is working correctly."""
    print("\n=== Tokenization Test ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load one example
        dataset_path = "/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_dataset/comparison_sft_dataset.jsonl"
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        example = dataset[0]
        prompt = example['prompt']
        
        # Tokenize
        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=4096,
            return_tensors=None
        )
        
        print(f"✅ Tokenization successful")
        print(f"  Input length: {len(tokenized['input_ids'])}")
        print(f"  Attention mask sum: {sum(tokenized['attention_mask'])}")
        
        # Check if tokens look reasonable
        if len(tokenized['input_ids']) < 10:
            print("❌ WARNING: Input is too short!")
        else:
            print("✅ Input length looks reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False

def test_model_forward():
    """Test if model forward pass works correctly."""
    print("\n=== Model Forward Pass Test ===")
    
    try:
        from src.finetuning.comparison_binary_classifier import ComparisonBinaryClassifier
        
        model = ComparisonBinaryClassifier("microsoft/deberta-v3-large", label_smoothing=0.0)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_len = 100
        
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        print(f"✅ Forward pass successful")
        print(f"  Loss: {outputs['loss']:.4f}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Logits range: [{outputs['logits'].min():.4f}, {outputs['logits'].max():.4f}]")
        
        # Check if logits are reasonable
        logits = outputs['logits']
        if torch.allclose(logits, torch.zeros_like(logits)):
            print("❌ WARNING: All logits are zero!")
        elif torch.std(logits) < 0.01:
            print("❌ WARNING: Logits have very low variance!")
        else:
            print("✅ Logits look reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return False

def test_label_processing():
    """Test label processing logic."""
    print("\n=== Label Processing Test ===")
    
    # Test the label conversion logic
    test_scores = [1.0, 2.0, 1.0, 2.0]
    
    # This is the logic from the training script
    binary_labels = [1.0 if score == 1.0 else 0.0 for score in test_scores]
    
    print(f"Original scores: {test_scores}")
    print(f"Binary labels: {binary_labels}")
    
    # Check if this makes sense
    expected = [1.0, 0.0, 1.0, 0.0]
    if binary_labels == expected:
        print("✅ Label conversion logic is correct")
    else:
        print(f"❌ Label conversion logic is wrong! Expected {expected}, got {binary_labels}")
    
    return binary_labels == expected

def create_minimal_test():
    """Create a minimal test to isolate the problem."""
    print("\n=== Creating Minimal Test ===")
    
    # Create a very simple synthetic dataset
    synthetic_data = []
    for i in range(100):
        # Create simple patterns
        if i < 50:
            # Pattern 1: Summary A is always better
            scores = [1.0, 1.0, 1.0, 1.0]
            prompt = f"This is test example {i}. Summary A is clearly better. Summary B is worse."
        else:
            # Pattern 2: Summary B is always better  
            scores = [2.0, 2.0, 2.0, 2.0]
            prompt = f"This is test example {i}. Summary B is clearly better. Summary A is worse."
        
        synthetic_data.append({
            "prompt": prompt,
            "completion": f"Scores: {scores}",
            "metadata": {
                "comparison_scores": scores
            }
        })
    
    # Save synthetic dataset
    output_path = "/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/synthetic_test.jsonl"
    with open(output_path, 'w') as f:
        for item in synthetic_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created synthetic dataset at {output_path}")
    print("  This dataset has clear patterns that should be easy to learn")
    
    return output_path

def main():
    """Run all diagnostic tests."""
    print("🔍 DIAGNOSING TRAINING ISSUES\n")
    
    tests = [
        ("Dataset Analysis", analyze_dataset),
        ("Tokenization Test", test_tokenization),
        ("Model Forward Pass", test_model_forward),
        ("Label Processing", test_label_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Create minimal test
    print(f"\n{'='*50}")
    print("Creating Minimal Test Dataset")
    print('='*50)
    try:
        synthetic_path = create_minimal_test()
        results.append(("Synthetic Dataset Creation", True))
    except Exception as e:
        print(f"❌ Synthetic dataset creation failed: {e}")
        results.append(("Synthetic Dataset Creation", False))
        synthetic_path = None
    
    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print('='*50)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed_tests = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if synthetic_path and passed_tests >= 3:
        print(f"\n🚀 RECOMMENDATION:")
        print(f"Try training on the synthetic dataset first:")
        print(f"python src/finetuning/comparison_binary_classifier.py \\")
        print(f"    --dataset_path {synthetic_path} \\")
        print(f"    --output_dir outputs/synthetic_test \\")
        print(f"    --num_train_epochs 2 \\")
        print(f"    --per_device_train_batch_size 4 \\")
        print(f"    --learning_rate 1e-4 \\")
        print(f"    --eval_steps 10")
        print(f"\nIf this works, the issue is with your real dataset.")
        print(f"If this fails, the issue is with the model/training setup.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
