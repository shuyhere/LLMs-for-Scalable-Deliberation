#!/usr/bin/env python3
"""
Evaluate trained Qwen4B models using direct vLLM integration
Based on the BaseLRM pattern provided by user
"""

import os
import sys
import json
import argparse
import re
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}. Please install vllm and transformers.") from e

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    # Fallback progress bar
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.desc = kwargs.get('desc', 'Processing')
            self.total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
            self.i = 0
            
        def __iter__(self):
            for item in self.iterable:
                self.i += 1
                if self.total:
                    percent = (self.i / self.total) * 100
                    print(f"\r{self.desc}: {self.i}/{self.total} ({percent:.1f}%)", end='', flush=True)
                else:
                    print(f"\r{self.desc}: {self.i}", end='', flush=True)
                yield item
            print()
            
        def set_postfix(self, **kwargs):
            pass


def extract_answer(text: str) -> str:
    """Extract answer from model output - simplified version for comparison task."""
    try:
        # For comparison tasks, look for JSON format
        if '{' in text and '}' in text:
            # Try to extract JSON
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                try:
                    result = json.loads(json_str)
                    return result
                except:
                    pass
        
        # Fallback: look for simple patterns
        patterns = [
            r'"choice":\s*"([^"]+)"',
            r'"preference":\s*"([^"]+)"',
            r'"answer":\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"choice": match.group(1)}
        
        return {"choice": "unknown"}
    except:
        return {"choice": "unknown"}


class DirectVLLMEvaluator:
    """Direct vLLM evaluator for trained models"""
    
    def __init__(self, model_path: str, model_name: str = "trained-model"):
        self.model_path = model_path
        self.model_name = model_name
        
        print(f"🔄 Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Determine GPU configuration
        num_gpus = self._get_gpu_count()
        tensor_parallel_size = min(num_gpus, 2)  # Use max 2 for TP to avoid memory issues
        
        print(f"🔄 Initializing vLLM with {num_gpus} GPUs, TP={tensor_parallel_size}...")
        
        # Initialize vLLM model
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.75
        )
        
        # Set sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.1,
            top_p=0.1,
        )
        
        print("✅ vLLM model loaded successfully")
    
    def _get_gpu_count(self) -> int:
        """Get available GPU count"""
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip() != "":
            return len([d for d in visible.split(",") if d.strip() != ""])
        try:
            return torch.cuda.device_count()
        except Exception:
            return 1
    
    def prepare_prompt(self, instruction: str) -> str:
        """Prepare prompt for the model"""
        messages = [
            {"role": "user", "content": instruction}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_response(self, instruction: str) -> str:
        """Generate response for a single instruction"""
        prompt = self.prepare_prompt(instruction)
        
        # Generate response
        outputs = self.model.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text
        
        return response
    
    def evaluate_batch(self, test_data: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Evaluate model on test data with batching"""
        print(f"🔄 Evaluating {len(test_data)} samples in batches of {batch_size}...")
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc="Processing batches"):
            batch = test_data[i:i + batch_size]
            
            # Prepare prompts for batch
            prompts = []
            for item in batch:
                instruction = item.get('instruction', '')
                prompt = self.prepare_prompt(instruction)
                prompts.append(prompt)
            
            # Generate responses for batch
            outputs = self.model.generate(prompts, self.sampling_params)
            
            # Process batch results
            for j, output in enumerate(outputs):
                item = batch[j]
                response = output.outputs[0].text
                
                # Parse prediction
                pred_json = extract_answer(response)
                
                # Parse ground truth
                gt_json = extract_answer(item.get('output', '{}'))
                
                results.append({
                    'instruction': item.get('instruction', ''),
                    'prediction': pred_json,
                    'ground_truth': gt_json,
                    'raw_response': response,
                    'sample_index': i + j
                })
        
        return results


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}...")
                    continue
    return data


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate evaluation metrics"""
    total_samples = len(results)
    valid_predictions = 0
    
    # Track accuracy for each dimension
    dimensions = ['perspective_representation', 'informativeness', 'neutrality_balance', 'policy_approval']
    dimension_correct = {dim: 0 for dim in dimensions}
    dimension_total = {dim: 0 for dim in dimensions}
    
    # Track exact match (all dimensions correct)
    exact_matches = 0
    
    for result in results:
        pred = result.get('prediction', {})
        gt = result.get('ground_truth', {})
        
        if pred and gt and isinstance(pred, dict) and isinstance(gt, dict):
            valid_predictions += 1
            
            # Check each dimension
            sample_exact_match = True
            for dim in dimensions:
                if dim in pred and dim in gt:
                    dimension_total[dim] += 1
                    if pred[dim] == gt[dim]:
                        dimension_correct[dim] += 1
                    else:
                        sample_exact_match = False
                else:
                    sample_exact_match = False
            
            if sample_exact_match:
                exact_matches += 1
    
    # Calculate accuracies
    dimension_accuracies = {}
    for dim in dimensions:
        if dimension_total[dim] > 0:
            dimension_accuracies[dim] = dimension_correct[dim] / dimension_total[dim]
        else:
            dimension_accuracies[dim] = 0.0
    
    # Overall accuracy (exact match)
    overall_accuracy = exact_matches / valid_predictions if valid_predictions > 0 else 0.0
    
    # Average dimension accuracy
    avg_dimension_accuracy = sum(dimension_accuracies.values()) / len(dimensions) if dimensions else 0.0
    
    return {
        'total_samples': total_samples,
        'valid_predictions': valid_predictions,
        'exact_matches': exact_matches,
        'overall_accuracy': overall_accuracy,
        'average_dimension_accuracy': avg_dimension_accuracy,
        'dimension_accuracies': dimension_accuracies,
        'dimension_correct': dimension_correct,
        'dimension_total': dimension_total,
        'valid_prediction_rate': valid_predictions / total_samples if total_samples > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model using direct vLLM")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    
    # Test data
    parser.add_argument("--test_path", type=str, required=True,
                       help="Path to test data JSONL file")
    
    # Output
    parser.add_argument("--output_file", type=str, default="evaluation_results_direct.json",
                       help="Output file for results")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    print("🚀 Starting direct vLLM evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_path}")
    
    # Load test data
    print("🔄 Loading test data...")
    test_data = load_test_data(args.test_path)
    print(f"✅ Loaded {len(test_data)} samples")
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(test_data):
        test_data = test_data[:args.max_samples]
        print(f"🔄 Limited to {len(test_data)} samples for testing")
    
    # Initialize evaluator
    evaluator = DirectVLLMEvaluator(args.model_path)
    
    # Run evaluation
    print("🔄 Starting evaluation...")
    results = evaluator.evaluate_batch(test_data, args.batch_size)
    
    # Calculate metrics
    print("🔄 Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Save results
    output_data = {
        'model_path': args.model_path,
        'test_path': args.test_path,
        'metrics': metrics,
        'results': results[:10]  # Save first 10 results for inspection
    }
    
    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n📊 Evaluation Results:")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid predictions: {metrics['valid_predictions']}")
    print(f"Exact matches (all dimensions correct): {metrics['exact_matches']}")
    print(f"Overall accuracy (exact match): {metrics['overall_accuracy']:.4f}")
    print(f"Average dimension accuracy: {metrics['average_dimension_accuracy']:.4f}")
    print(f"Valid prediction rate: {metrics['valid_prediction_rate']:.4f}")
    
    print(f"\n📈 Per-Dimension Accuracy:")
    for dim, acc in metrics['dimension_accuracies'].items():
        correct = metrics['dimension_correct'][dim]
        total = metrics['dimension_total'][dim]
        print(f"  {dim}: {acc:.4f} ({correct}/{total})")
    
    print(f"\n💾 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
