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
import math

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


def _build_instruction_from_item(item: Dict[str, Any]) -> Optional[str]:
    """Construct an instruction from domain fields if Alpaca fields are absent."""
    q = item.get('question')
    c = item.get('comment')
    s = item.get('summary')
    if not (q and c and s):
        return None
    # Ask the model to output a strict JSON with the four dimensions
    return (
        "You are a judge. Read the question, the annotator's opinion, and the summary. "
        "Rate the summary on four dimensions with integer scores in [1,5].\n\n"
        f"Question: {q}\n"
        f"Annotator opinion: {c}\n"
        f"Summary: {s}\n\n"
        "Return ONLY a JSON object with exactly these keys: "
        "perspective_representation, informativeness, neutrality_balance, policy_approval. "
        "Example: {\"perspective_representation\": 3, \"informativeness\": 4, \"neutrality_balance\": 3, \"policy_approval\": 2}"
    )


def _extract_ground_truth_from_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Prefer standard fields
    for key in ('output', 'labels', 'scores', 'rating_scores', 'targets'):
        if key in item:
            val = item[key]
            if isinstance(val, str):
                try:
                    j = json.loads(val)
                    if isinstance(j, dict):
                        return j
                except Exception:
                    pass
            if isinstance(val, dict):
                return val
    return None


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSONL or JSON and ensure each item has instruction/output."""
    path = Path(file_path)
    raw_items: List[Dict[str, Any]] = []

    # Try JSONL first
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_items.append(json.loads(line))
    except Exception:
        # Fallback: try parsing as a single JSON array or object
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(obj, list):
                raw_items = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                raw_items = [obj]
            else:
                raw_items = []
        except Exception:
            print(f"Error: cannot parse test file: {file_path}")
            return []

    norm_items: List[Dict[str, Any]] = []
    for it in raw_items:
        # Prefer explicit prompt/instruction from the dataset
        instr = it.get('instruction') or it.get('prompt')
        if not instr:
            instr = _build_instruction_from_item(it)
        gt = _extract_ground_truth_from_item(it)
        norm_items.append({
            **it,
            'instruction': instr or '',
            'output': json.dumps(gt, ensure_ascii=False) if isinstance(gt, dict) else it.get('output', '')
        })

    return norm_items


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
    
    # Correlations between predicted and ground-truth values
    def to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        # Direct numeric
        if isinstance(value, (int, float)):
            try:
                if math.isfinite(float(value)):
                    return float(value)
            except Exception:
                return None
            return None
        # String numeric
        if isinstance(value, str):
            v = value.strip()
            # Common categorical aliases mapping
            alias_map = {
                'yes': 1.0, 'no': 0.0,
                'true': 1.0, 'false': 0.0,
                'positive': 1.0, 'negative': -1.0,
                'neutral': 0.0
            }
            lower_v = v.lower()
            if lower_v in alias_map:
                return alias_map[lower_v]
            try:
                return float(v)
            except Exception:
                return None
        # Unsupported types
        return None

    def safe_pearson(xs: List[float], ys: List[float]) -> Optional[float]:
        if len(xs) < 2:
            return None
        # Check variance
        if all(x == xs[0] for x in xs) or all(y == ys[0] for y in ys):
            return None
        try:
            import numpy as np
            r = np.corrcoef(np.array(xs, dtype=float), np.array(ys, dtype=float))[0, 1]
            if np.isnan(r):
                return None
            return float(r)
        except Exception:
            return None

    def safe_spearman(xs: List[float], ys: List[float]) -> Optional[float]:
        try:
            from scipy.stats import spearmanr
        except Exception:
            return None
        if len(xs) < 2:
            return None
        try:
            r, _ = spearmanr(xs, ys)
            if r is None or (isinstance(r, float) and (math.isnan(r) or math.isinf(r))):
                return None
            return float(r)
        except Exception:
            return None

    dimension_correlations: Dict[str, Dict[str, Optional[float]]]= {}
    for dim in dimensions:
        xs: List[float] = []
        ys: List[float] = []
        for result in results:
            pred = result.get('prediction', {})
            gt = result.get('ground_truth', {})
            if not isinstance(pred, dict) or not isinstance(gt, dict):
                continue
            if dim in pred and dim in gt:
                x = to_float(pred[dim])
                y = to_float(gt[dim])
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
        pearson = safe_pearson(xs, ys)
        spearman = safe_spearman(xs, ys)
        dimension_correlations[dim] = {
            'pearson': pearson,
            'spearman': spearman,
            'count': len(xs)
        }

    return {
        'total_samples': total_samples,
        'valid_predictions': valid_predictions,
        'exact_matches': exact_matches,
        'overall_accuracy': overall_accuracy,
        'average_dimension_accuracy': avg_dimension_accuracy,
        'dimension_accuracies': dimension_accuracies,
        'dimension_correct': dimension_correct,
        'dimension_total': dimension_total,
        'valid_prediction_rate': valid_predictions / total_samples if total_samples > 0 else 0.0,
        'dimension_correlations': dimension_correlations
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
