import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.LanguageModel import LanguageModel
from utils.prompts.evaluation import EvaluationPrompt


class SummaryEvaluator:
    """
    A class to evaluate how well comments are represented in summaries using LLM models.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "You are a helpful assistant", temperature: float = 0.7, verbose: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            model: The LLM model to use for evaluation
            system_prompt: Custom system prompt for the model
            temperature: Temperature for model generation (0.0 to 2.0)
        """
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = LanguageModel(model_name=model, temperature=temperature)
        self.verbose = verbose
    
    def evaluate_comment_representation(self, summary: str, comment: str) -> Dict[str, Any]:
        """
        Evaluate how well a comment is represented in a summary.
        
        Args:
            summary: The summary to evaluate against
            comment: The comment to evaluate representation of
            
        Returns:
            Dictionary containing full evaluation response and extracted score
        """
        prompt = EvaluationPrompt(summary=summary, comment=comment, system_prompt=self.system_prompt)
        system_prompt, user_input = prompt.summary_evaluation_prompt_from_comments()
        if self.verbose:
            print(f"System prompt: {system_prompt}")
            print(f"User input: {user_input}")
        
        # Call the language model
        response = self.client.chat_completion(system_prompt=system_prompt, input_text=user_input)
        
        # Check if response is None (API call failed)
        if response is None:
            raise Exception("Language model returned None - API call failed")
        
        # Extract score from response
        score = self._extract_score_from_response(response)
        
        return {
            "full_response": response,  # Complete model response
            "extracted_score": score,   # Extracted numerical score
            "evaluation_timestamp": datetime.now().isoformat()  # When evaluation was performed
        }
    
    def evaluate_multiple_comments(self, summary: str, comments: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple comments against a single summary.
        
        Args:
            summary: The summary to evaluate against
            comments: List of comments to evaluate
            
        Returns:
            List of evaluation results with scores and responses
        """
        results = []
        
        for i, comment in enumerate(comments):
            try:
                evaluation_result = self.evaluate_comment_representation(summary, comment)
                
                result = {
                    "comment_index": i,
                    "comment": comment,
                    "evaluation_response": evaluation_result["full_response"], 
                    "extracted_score": evaluation_result["extracted_score"],   # Extracted score
                    "score": evaluation_result["extracted_score"],            # Backward compatibility
                    "status": "success",
                    "evaluation_model": self.model,                          # Model used for this evaluation
                    "evaluation_details": {
                        "evaluation_timestamp": evaluation_result["evaluation_timestamp"]
                    }
                }
            except Exception as e:
                result = {
                    "comment_index": i,
                    "comment": comment,
                    "evaluation_response": f"Error: {str(e)}",
                    "extracted_score": None,
                    "score": None,
                    "status": "error",
                    "evaluation_model": self.model,                          # Model used for this evaluation
                    "evaluation_details": {
                        "evaluation_timestamp": datetime.now().isoformat()
                    }
                }
            
            results.append(result)
        
        return results
    
    def _extract_score_from_response(self, response: str) -> int:
        """
        Extract the numerical score from the evaluation response.
        
        Args:
            response: The model's evaluation response
            
        Returns:
            Extracted score (1-5) or None if extraction fails
        """
        try:
            # Priority 1: Look for \boxed{1}, \boxed{2}, etc. format
            import re
            boxed_match = re.search(r'\\boxed\{(\d+)\}', response)
            if boxed_match:
                score = int(boxed_match.group(1))
                if 1 <= score <= 5:
                    return score
            
            # Priority 2: Look for patterns like "(1)", "(2)", etc.
            score_match = re.search(r'\((\d+)\)', response)
            if score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 5:
                    return score
            
            # Priority 3: Look for just the number
            number_match = re.search(r'\b([1-5])\b', response)
            if number_match:
                score = int(number_match.group(1))
                if 1 <= score <= 5:
                    return score
            
            return None
        except Exception:
            return None
    
    def get_evaluation_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with evaluation statistics
        """
        if not results:
            return {}
        
        successful_results = [r for r in results if r["status"] == "success" and r["score"] is not None]
        
        if not successful_results:
            return {
                "total_comments": len(results),
                "successful_evaluations": 0,
                "error_evaluations": len(results),
                "average_score": None,
                "score_distribution": {}
            }
        
        scores = [r["score"] for r in successful_results]
        
        # Calculate score distribution
        score_distribution = {}
        for score in range(1, 6):
            score_distribution[score] = scores.count(score)
        
        # Calculate average score
        average_score = sum(scores) / len(scores)
        
        return {
            "total_comments": len(results),
            "successful_evaluations": len(successful_results),
            "error_evaluations": len(results) - len(successful_results),
            "average_score": round(average_score, 2),
            "score_distribution": score_distribution,
            "min_score": min(scores),
            "max_score": max(scores)
        }


class DebertaEvaluator:
    """
    A class to evaluate comment-summary pairs using a trained DeBERTa regression model.
    
    This evaluator loads a pre-trained DeBERTa model that was trained to predict
    4 dimensions: perspective_representation, informativeness, neutrality_balance, policy_approval.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the DeBERTa evaluator.
        
        Args:
            model_path: Path to the trained DeBERTa model directory
            device: Device to run the model on ("cuda" or "cpu")
        """
        import torch
        from transformers import AutoTokenizer
        
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        print(f"Loading DeBERTa model from {model_path}")
        # Use the base model tokenizer to ensure compatibility
        # The training used microsoft/deberta-v3-base tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the trained model
        self.model = self._load_trained_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Target dimensions (same as training)
        self.target_keys = [
            "perspective_representation",
            "informativeness", 
            "neutrality_balance",
            "policy_approval"
        ]
        
        print(f"DeBERTa evaluator loaded successfully on {self.device}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model configuration.
        
        Returns:
            Dictionary containing model configuration and loading details
        """
        return {
            "successful_config": getattr(self, 'successful_config', None),
            "successful_weight_file": getattr(self, 'successful_weight_file', None),
            "device": self.device,
            "model_path": self.model_path,
            "target_keys": self.target_keys
        }
    
    def _load_trained_model(self, model_path: str):
        """Load the trained DeBERTa model"""
        import torch
        import torch.nn as nn
        from transformers import AutoModel
        
        class MultiOutputRegressor(nn.Module):
            """Model architecture matching the training script exactly"""
            def __init__(self, base_model_name: str, num_dims: int = 4, dropout_rate: float = 0.1, 
                         use_tanh: bool = False, use_sigmoid: bool = False, use_relu: bool = False,
                         use_leaky_relu: bool = False, use_elu: bool = False):
                super().__init__()
                
                # Load the model exactly as it was trained
                # The training script used microsoft/deberta-v3-base with its default vocab size
                from transformers import AutoConfig
                
                # Important: Just create the model architecture without loading pretrained weights
                # The actual weights will be loaded later from the checkpoint
                print(f"Creating model architecture based on microsoft/deberta-v3-base")
                
                # Get the config from the base model
                config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
                
                # Create model with the config (not loading pretrained weights)
                self.encoder = AutoModel.from_config(config)
                hidden = self.encoder.config.hidden_size
                self.dropout = nn.Dropout(dropout_rate)
                # Add a hidden layer for better feature extraction
                self.hidden = nn.Linear(hidden, hidden // 2)
                self.activation = nn.GELU()
                self.head = nn.Linear(hidden // 2, num_dims)
                
                # Check for parameter conflicts
                activation_count = sum([use_tanh, use_sigmoid, use_relu, use_leaky_relu, use_elu])
                if activation_count > 1:
                    raise ValueError("Cannot use multiple activation functions at the same time. Choose only one.")
                
                self.use_tanh = use_tanh
                self.use_sigmoid = use_sigmoid
                self.use_relu = use_relu
                self.use_leaky_relu = use_leaky_relu
                self.use_elu = use_elu
                
                if use_tanh:
                    self.output_activation = nn.Tanh()  # Output in [-1, 1]
                elif use_sigmoid:
                    self.output_activation = nn.Sigmoid()  # Output in [0, 1]
                elif use_relu:
                    self.output_activation = nn.ReLU()  # Output in [0, +inf)
                elif use_leaky_relu:
                    self.output_activation = nn.LeakyReLU(negative_slope=0.01)  # Output in (-inf, +inf)
                elif use_elu:
                    self.output_activation = nn.ELU()  # Output in [-1, +inf)
                else:
                    self.output_activation = None  # No activation, raw logits
                
                self.num_dims = num_dims
                self.loss_fct = nn.MSELoss()
                
                # Add config attribute for compatibility with Transformers Trainer
                self.config = self.encoder.config

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                # Remove any unexpected kwargs
                kwargs.pop("num_items_in_batch", None)
                
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                pooled = out.last_hidden_state[:, 0, :]  # Use [CLS] token
                pooled = self.dropout(pooled)
                hidden = self.activation(self.hidden(pooled))
                hidden = self.dropout(hidden)
                logits = self.head(hidden)  # (B, num_dims)
                
                if self.output_activation is not None:
                    predictions = self.output_activation(logits)
                else:
                    # No constraint, let the model learn the range
                    predictions = logits
                
                loss = None
                if labels is not None:
                    # Use Huber loss for robustness
                    huber_loss = nn.HuberLoss(delta=1.0)
                    loss = huber_loss(predictions, labels)
                
                return {"loss": loss, "logits": predictions}
        
        # Automatically detect activation function from model path name
        # Parse the model path to identify the activation function used
        model_path_lower = model_path.lower()
        
        # Define the primary configuration based on model name
        primary_config = None
        
        if 'leaky_relu' in model_path_lower:
            print(f"Detected LeakyReLU activation from model path")
            primary_config = {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": True, "use_elu": False}
        elif 'elu' in model_path_lower:
            print(f"Detected ELU activation from model path")
            primary_config = {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": True}
        elif 'relu' in model_path_lower and 'leaky' not in model_path_lower:
            print(f"Detected ReLU activation from model path")
            primary_config = {"use_tanh": False, "use_sigmoid": False, "use_relu": True, "use_leaky_relu": False, "use_elu": False}
        elif 'sigmoid' in model_path_lower:
            print(f"Detected Sigmoid activation from model path")
            primary_config = {"use_tanh": False, "use_sigmoid": True, "use_relu": False, "use_leaky_relu": False, "use_elu": False}
        elif 'tanh' in model_path_lower:
            print(f"Detected Tanh activation from model path")
            primary_config = {"use_tanh": True, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": False}
        else:
            print(f"No activation function detected in model path, using default (no activation)")
            primary_config = {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": False}
        
        # Create model configurations list with detected config first
        model_configs = [primary_config]
        
        # Add fallback configurations in case the primary doesn't work
        fallback_configs = [
            {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": False},  # No activation
            {"use_tanh": True, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": False},   # Tanh
            {"use_tanh": False, "use_sigmoid": True, "use_relu": False, "use_leaky_relu": False, "use_elu": False},   # Sigmoid
            {"use_tanh": False, "use_sigmoid": False, "use_relu": True, "use_leaky_relu": False, "use_elu": False},   # ReLU
            {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": True, "use_elu": False},  # LeakyReLU
            {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": True},   # ELU
        ]
        
        # Add fallback configs that are different from the primary
        for config in fallback_configs:
            if config != primary_config:
                model_configs.append(config)
        
        model = None
        successful_config = None
        successful_weight_file = None
        
        for i, config in enumerate(model_configs):
            try:
                print(f"Trying model configuration {i+1}: {config}")
                model = MultiOutputRegressor(model_path, **config)
                
                # Try loading weights
                try:
                    state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
                    model.load_state_dict(state_dict)
                    successful_config = config
                    successful_weight_file = "pytorch_model.bin"
                    print(f"✓ Successfully loaded model weights from pytorch_model.bin with config {i+1}")
                    print(f"✓ Final configuration: {config}")
                    break
                except FileNotFoundError:
                    try:
                        # Try loading from safetensors format
                        from safetensors.torch import load_file
                        state_dict = load_file(f"{model_path}/model.safetensors")
                        model.load_state_dict(state_dict)
                        successful_config = config
                        successful_weight_file = "model.safetensors"
                        print(f"✓ Successfully loaded model weights from model.safetensors with config {i+1}")
                        print(f"✓ Final configuration: {config}")
                        break
                    except Exception as e:
                        print(f"✗ Failed to load weights with config {i+1}: {e}")
                        continue
            except Exception as e:
                print(f"✗ Failed to create model with config {i+1}: {e}")
                continue
        
        if model is None:
            print("⚠️  Warning: Could not load trained weights with any configuration")
            print("⚠️  Using randomly initialized model with default configuration")
            model = MultiOutputRegressor(model_path)
            successful_config = {"use_tanh": False, "use_sigmoid": False, "use_relu": False, "use_leaky_relu": False, "use_elu": False}
            successful_weight_file = "none (random initialization)"
        
        # Store the successful configuration for reference
        self.successful_config = successful_config
        self.successful_weight_file = successful_weight_file
        
        print(f"\n🎯 Model Loading Summary:")
        print(f"   Configuration: {successful_config}")
        print(f"   Weight file: {successful_weight_file}")
        print(f"   Device: {self.device}")
        
        return model
    
    def _normalize_score(self, score: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
        """Normalize score from [min_val, max_val] to [0, 1] (same as training)"""
        return (score - min_val) / (max_val - min_val)
    
    def _denormalize_score(self, normalized: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
        """Denormalize score from [0, 1] back to [min_val, max_val]"""
        return normalized * (max_val - min_val) + min_val
    
    def _prepare_input(self, question: str, comment: str, summary: str, max_length: int = 2048) -> dict:
        """
        Prepare input in the same format as training data.
        
        Args:
            question: The question/topic being discussed
            comment: The annotator's opinion/comment
            summary: The summary to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Use the same format as training: "Question: {q} [SEP] Annotator opinion: {c} [SEP] Summary: {s}"
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"
        text = f"Question: {question} {sep_token} Annotator opinion: {comment} {sep_token} Summary: {summary}"
        
        # Tokenize
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return enc
    
    def evaluate_single(self, question: str, comment: str, summary: str, 
                       max_length: int = 2048) -> Dict[str, Any]:
        """
        Evaluate a single comment-summary pair.
        
        Args:
            question: The question/topic being discussed
            comment: The annotator's opinion/comment  
            summary: The summary to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing predictions for all 4 dimensions
        """
        import torch
        
        # Prepare input
        inputs = self._prepare_input(question, comment, summary, max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs["logits"].cpu().numpy()[0]  # (4,)
        
        # Use predictions as-is (no denormalization needed)
        denorm_predictions = predictions
        
        # Create results dictionary
        results = {
            "question": question,
            "comment": comment, 
            "summary": summary,
            "predictions": {
                key: float(denorm_predictions[i]) 
                for i, key in enumerate(self.target_keys)
            }
        }
        
        return results
    
    def evaluate_batch(self, data: List[Dict[str, str]], max_length: int = 2048) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of comment-summary pairs.
        
        Args:
            data: List of dictionaries with 'question', 'comment', 'summary' keys
            max_length: Maximum sequence length
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, item in enumerate(data):
            try:
                question = item.get("question", "")
                comment = item.get("comment", "")
                summary = item.get("summary", "")
                
                if not all([question, comment, summary]):
                    print(f"Warning: Item {i} missing required fields, skipping")
                    continue
                
                result = self.evaluate_single(question, comment, summary, max_length)
                result["item_index"] = i
                result["status"] = "success"
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating item {i}: {e}")
                results.append({
                    "item_index": i,
                    "question": item.get("question", ""),
                    "comment": item.get("comment", ""),
                    "summary": item.get("summary", ""),
                    "predictions": {key: None for key in self.target_keys},
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def evaluate_from_jsonl(self, jsonl_path: str, max_length: int = 2048) -> List[Dict[str, Any]]:
        """
        Evaluate data from a JSONL file in the same format as training data.
        
        Args:
            jsonl_path: Path to JSONL file with 'question', 'comment', 'summary' fields
            max_length: Maximum sequence length
            
        Returns:
            List of evaluation results
        """
        import json
        
        data = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON line: {e}")
                    continue
        
        print(f"Loaded {len(data)} items from {jsonl_path}")
        return self.evaluate_batch(data, max_length)
    
    def get_evaluation_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from evaluation results.
        
        Args:
            results: List of evaluation results from evaluate_batch or evaluate_from_jsonl
            
        Returns:
            Dictionary with evaluation statistics
        """
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("status") == "success"]
        
        if not successful_results:
            return {
                "total_items": len(results),
                "successful_evaluations": 0,
                "error_evaluations": len(results),
                "average_scores": {},
                "score_distributions": {}
            }
        
        # Calculate statistics for each dimension
        stats = {
            "total_items": len(results),
            "successful_evaluations": len(successful_results),
            "error_evaluations": len(results) - len(successful_results),
            "average_scores": {},
            "score_distributions": {},
            "score_ranges": {}
        }
        
        for key in self.target_keys:
            scores = [r["predictions"][key] for r in successful_results if key in r["predictions"]]
            
            if scores:
                scores = [s for s in scores if s is not None]
                if scores:
                    stats["average_scores"][key] = sum(scores) / len(scores)
                    stats["score_ranges"][key] = {
                        "min": min(scores),
                        "max": max(scores),
                        "std": (sum((s - stats["average_scores"][key])**2 for s in scores) / len(scores))**0.5
                    }
                    
                    # Score distribution (binned)
                    bins = [i for i in range(-1, 8)]  # -1 to 7 range
                    hist, _ = np.histogram(scores, bins=bins)
                    stats["score_distributions"][key] = {
                        f"score_{bins[i]}": int(hist[i]) 
                        for i in range(len(hist))
                    }
        
        return stats
def main():
    """Example usage of both SummaryEvaluator and DebertaEvaluator."""
    
    # Sample data for evaluation
    sample_question = "What are your thoughts on artificial intelligence development?"
    sample_comment = "I believe AI will revolutionize healthcare and education, but we need strong regulations to prevent misuse."
    sample_summary = """
    The discussion on AI development revealed mixed perspectives. Some participants emphasized the potential benefits 
    in healthcare and education, while others expressed concerns about the need for proper regulation and oversight 
    to ensure responsible development and deployment of AI technologies.
    """
    
    print("=== LLM-BASED EVALUATION (SummaryEvaluator) ===")
    
    # Initialize LLM evaluator
    # llm_evaluator = SummaryEvaluator(model="gpt-4o-mini")
    
    # print("Single comment evaluation:")
    # single_eval = llm_evaluator.evaluate_comment_representation(sample_summary, sample_comment)
    # print(f"Comment: {sample_comment}")
    # print(f"Evaluation: {single_eval}")
    
    print("\n=== DEBERTA-BASED EVALUATION (DebertaEvaluator) ===")
    
    # Example usage of DebertaEvaluator (requires trained model)
    model_path = "checkpoints/deberta_regression_base_v10_pair_split_sigmoid"
    
    if Path(model_path).exists():
        try:
            # Initialize DeBERTa evaluator
            deberta_evaluator = DebertaEvaluator(model_path, device="cuda")
            
            print("Single evaluation:")
            result = deberta_evaluator.evaluate_single(sample_question, sample_comment, sample_summary)
            print(f"Question: {result['question']}")
            print(f"Comment: {result['comment']}")
            print(f"Summary: {result['summary']}")
            print("Predictions:")
            for key, value in result['predictions'].items():
                print(f"  {key}: {value:.4f}")
            
            # Batch evaluation example
            batch_data = [
                {
                    "question": sample_question,
                    "comment": sample_comment,
                    "summary": sample_summary
                },
                {
                    "question": "What is your view on climate change policies?",
                    "comment": "We need immediate action on carbon emissions reduction.",
                    "summary": "Participants discussed various climate policy approaches with emphasis on urgent action."
                }
            ]
            
            print("\nBatch evaluation:")
            batch_results = deberta_evaluator.evaluate_batch(batch_data)
            
            for i, result in enumerate(batch_results):
                print(f"\nItem {i+1}:")
                print(f"  Status: {result['status']}")
                if result['status'] == 'success':
                    print("  Predictions:")
                    for key, value in result['predictions'].items():
                        print(f"    {key}: {value:.4f}")
            
            # Get statistics
            print("\nEvaluation Statistics:")
            stats = deberta_evaluator.get_evaluation_statistics(batch_results)
            for key, value in stats.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"Error loading DeBERTa model: {e}")
            print("Make sure you have a trained model at the specified path")
    else:
        print(f"DeBERTa model not found at {model_path}")
        print("Please train a model first using sft_train_multioutput_regression.py")
    
    print("\n=== JSONL EVALUATION EXAMPLE ===")
    print("To evaluate from a JSONL file:")
    print("  evaluator = DebertaEvaluator('/path/to/trained/model')")
    print("  results = evaluator.evaluate_from_jsonl('/path/to/data.jsonl')")
    print("  stats = evaluator.get_evaluation_statistics(results)")


if __name__ == "__main__":
    main()
