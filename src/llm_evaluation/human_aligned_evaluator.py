#!/usr/bin/env python3
"""
Core human-aligned evaluator for LLM correlation experiments
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from datetime import datetime

from models.LanguageModel import LanguageModel
from utils.prompts.evaluation import HumanAnnotationPrompt
from utils.data_processing import HumanAnnotationDataProcessor

logger = logging.getLogger(__name__)


class HumanAlignedEvaluator:
    """
    Core evaluator for human-LLM correlation experiments
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3, debug: bool = False):
        """
        Initialize the evaluator
        
        Args:
            model: The LLM model to use for evaluation
            temperature: Temperature for model generation (lower for more consistency)
            debug: Enable debug mode for detailed logging
        """
        self.model = model
        self.temperature = temperature
        self.debug = debug
        self.client = LanguageModel(model_name=model, temperature=temperature)
        self.data_processor = HumanAnnotationDataProcessor(debug=debug)
        self.system_prompt = "You are an expert evaluator analyzing summaries of public opinions."
        
        if debug:
            logger.setLevel(logging.DEBUG)
    
    def evaluate_rating(self, summary: str, question: str, annotator_answer: str = "") -> Dict[str, Any]:
        """
        Evaluate a single summary with rating questions
        
        Args:
            summary: The summary to evaluate
            question: The original question
            annotator_answer: The annotator's own answer to the question
            
        Returns:
            Dictionary with ratings and reasoning
        """
        if self.debug:
            logger.debug(f"Evaluating rating for question: {question}")
        
        prompt = HumanAnnotationPrompt(
            summary=summary,
            question=question,
            annotator_answer=annotator_answer,
            task_type="rating"
        )
        
        user_input = prompt.get_rating_prompt()
        
        if self.debug:
            logger.debug(f"Generated rating prompt")
        
        try:
            response = self.client.chat_completion(
                system_prompt=self.system_prompt,
                input_text=user_input
            )
            
            if self.debug:
                logger.debug(f"Received LLM response")
            
            # Parse JSON response - handle markdown code blocks
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]  # Remove ```json
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]  # Remove ```
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]  # Remove trailing ```
            response_clean = response_clean.strip()
            
            result = json.loads(response_clean)
            
            # Validate all required fields are present
            required_fields = ['perspective_representation', 'informativeness', 
                             'neutrality_balance', 'policy_approval']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
                if not isinstance(result[field], int) or result[field] < 1 or result[field] > 5:
                    raise ValueError(f"Invalid value for {field}: {result[field]} (must be 1-5)")
            
            if self.debug:
                logger.debug(f"Parsed ratings successfully")
            
            return {
                "status": "success",
                "ratings": result,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            if self.debug:
                logger.error(f"Error in evaluate_rating: {str(e)}")
                logger.error(f"Response was: {response if 'response' in locals() else 'No response'}")
            return {
                "status": "error",
                "error": str(e),
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_comparison(self, summary_a: str, summary_b: str, question: str, annotator_answer: str = "") -> Dict[str, Any]:
        """
        Evaluate a comparison between two summaries
        
        Args:
            summary_a: First summary
            summary_b: Second summary
            question: The original question
            annotator_answer: The annotator's own answer to the question
            
        Returns:
            Dictionary with comparison results
        """
        if self.debug:
            logger.debug(f"Evaluating comparison for question: {question}")
        
        prompt = HumanAnnotationPrompt(
            summary_a=summary_a,
            summary_b=summary_b,
            question=question,
            annotator_answer=annotator_answer,
            task_type="comparison"
        )
        
        user_input = prompt.get_comparison_prompt()
        
        if self.debug:
            logger.debug(f"Generated comparison prompt")
        
        try:
            response = self.client.chat_completion(
                system_prompt=self.system_prompt,
                input_text=user_input
            )
            
            if self.debug:
                logger.debug(f"Received LLM response")
            
            # Parse JSON response - handle markdown code blocks
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]  # Remove ```json
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]  # Remove ```
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]  # Remove trailing ```
            response_clean = response_clean.strip()
            
            result = json.loads(response_clean)
            
            # Validate all required fields are present
            required_fields = ['perspective_representation', 'informativeness', 
                             'neutrality_balance', 'policy_approval']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
                if result[field] not in [1, 2]:
                    raise ValueError(f"Invalid value for {field}: {result[field]} (must be 1 or 2)")
            
            if self.debug:
                logger.debug(f"Parsed comparisons successfully")
            
            return {
                "status": "success",
                "comparisons": result,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            if self.debug:
                logger.error(f"Error in evaluate_comparison: {str(e)}")
                logger.error(f"Response was: {response if 'response' in locals() else 'No response'}")
            return {
                "status": "error",
                "error": str(e),
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_correlation(self, human_scores: List[float], llm_scores: List[float]) -> Dict[str, float]:
        """
        Calculate correlation metrics between human and LLM scores
        
        Args:
            human_scores: List of human annotation scores
            llm_scores: List of LLM evaluation scores
            
        Returns:
            Dictionary with correlation metrics
        """
        if len(human_scores) != len(llm_scores) or len(human_scores) < 2:
            return {"error": "Insufficient data for correlation"}
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(human_scores, llm_scores)
        
        # Spearman correlation (for ordinal data)
        spearman_r, spearman_p = stats.spearmanr(human_scores, llm_scores)
        
        # Cohen's kappa (for agreement)
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(human_scores, llm_scores, weights='linear')
        
        # Mean Absolute Error
        mae = np.mean(np.abs(np.array(human_scores) - np.array(llm_scores)))
        
        return {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "cohen_kappa": kappa,
            "mae": mae,
            "n_samples": len(human_scores)
        }
    
    def process_rating_annotations(self, annotations: List[Dict], annotation_path: Path) -> List[Dict]:
        """Process rating annotations and get LLM evaluations"""
        results = []
        
        for i, ann in enumerate(annotations):
            if i % 10 == 0:
                print(f"Processing rating annotation {i+1}/{len(annotations)}")
            
            # Get summary and question from metadata
            if 'metadata' not in ann:
                continue
            
            metadata = ann['metadata']
            summary = metadata.get('text', '')
            question = metadata.get('question', '')
            
            # Get annotator's answer
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary or not question:
                continue
            
            # Get LLM evaluation with annotator's perspective
            llm_result = self.evaluate_rating(summary, question, annotator_answer)
            
            # Extract human ratings
            human_ratings = self.data_processor.extract_human_ratings(ann)
            
            # Create unique ID for this result
            unique_id = f"rating_{ann['id']}_{self.model}"
            
            # Create data source information
            data_source = {
                'annotation_id': ann['id'],
                'annotation_type': 'rating',
                'relative_path': f"annotation/summary-rating/annotation_output/full/{ann.get('user_id', 'unknown')}/annotated_instances.jsonl",
                'absolute_path': str(annotation_path / ann.get('user_id', 'unknown') / 'annotated_instances.jsonl'),
                'user_id': ann.get('user_id', 'unknown'),
                'timestamp': ann.get('timestamp', 'unknown')
            }
            
            if llm_result['status'] == 'success' and human_ratings:
                results.append({
                    'unique_id': unique_id,
                    'annotation_id': ann['id'],
                    'model_name': self.model,
                    'model_temperature': self.temperature,
                    'timestamp': datetime.now().isoformat(),
                    'human_ratings': human_ratings,
                    'llm_result': llm_result,
                    'question': question,
                    'has_annotator_answer': bool(annotator_answer),
                    'annotator_answer': annotator_answer,
                    'data_source': data_source,
                    'metadata': {
                        'topic': metadata.get('topic', ''),
                        'model': metadata.get('model', ''),
                        'comment_num': metadata.get('comment_num', 0),
                        'summary_length': len(summary) if summary else 0
                    }
                })
        
        return results
    
    def process_comparison_annotations(self, annotations: List[Dict], annotation_path: Path) -> List[Dict]:
        """Process comparison annotations and get LLM evaluations"""
        results = []
        
        for i, ann in enumerate(annotations):
            if i % 10 == 0:
                print(f"Processing comparison annotation {i+1}/{len(annotations)}")
            
            # Get summaries and question from metadata
            if 'metadata' not in ann:
                continue
            
            metadata = ann['metadata']
            summary_a = metadata.get('summary_a_text', '')
            summary_b = metadata.get('summary_b_text', '')
            question = metadata.get('question', '')
            
            # Get annotator's answer
            annotator_answer = ann.get('annotator_answer', '')
            
            if not summary_a or not summary_b or not question:
                continue
            
            # Get LLM evaluation with annotator's perspective
            llm_result = self.evaluate_comparison(summary_a, summary_b, question, annotator_answer)
            
            # Extract human comparisons
            human_comparisons = self.data_processor.extract_human_comparisons(ann)
            
            # Create unique ID for this result
            unique_id = f"comparison_{ann['id']}_{self.model}"
            
            # Create data source information
            data_source = {
                'annotation_id': ann['id'],
                'annotation_type': 'comparison',
                'relative_path': f"annotation/summary-rating/annotation_output/full/{ann.get('user_id', 'unknown')}/annotated_instances.jsonl",
                'absolute_path': str(annotation_path / ann.get('user_id', 'unknown') / 'annotated_instances.jsonl'),
                'user_id': ann.get('user_id', 'unknown'),
                'timestamp': ann.get('timestamp', 'unknown')
            }
            
            if llm_result['status'] == 'success' and human_comparisons:
                results.append({
                    'unique_id': unique_id,
                    'annotation_id': ann['id'],
                    'model_name': self.model,
                    'model_temperature': self.temperature,
                    'timestamp': datetime.now().isoformat(),
                    'human_comparisons': human_comparisons,
                    'llm_result': llm_result,
                    'question': question,
                    'has_annotator_answer': bool(annotator_answer),
                    'annotator_answer': annotator_answer,
                    'data_source': data_source,
                    'metadata': {
                        'topic': metadata.get('topic', ''),
                        'model_a': metadata.get('model_a', ''),
                        'model_b': metadata.get('model_b', ''),
                        'comment_num': metadata.get('comment_num', 0),
                        'summary_a_length': len(summary_a) if summary_a else 0,
                        'summary_b_length': len(summary_b) if summary_b else 0
                    }
                })
        
        return results
    
    def calculate_all_correlations(self, rating_results: List[Dict], 
                                  comparison_results: List[Dict]) -> Dict[str, Any]:
        """Calculate correlations for all metrics"""
        correlations = {}
        
        # Rating correlations
        if rating_results:
            rating_metrics = ['perspective_representation', 'informativeness', 
                            'neutrality_balance', 'policy_approval']
            
            for metric in rating_metrics:
                human_scores = []
                llm_scores = []
                
                for result in rating_results:
                    # Check if both human and LLM ratings exist and are successful
                    if (metric in result['human_ratings'] and 
                        result['llm_result']['status'] == 'success' and
                        metric in result['llm_result']['ratings']):
                        human_scores.append(result['human_ratings'][metric])
                        llm_scores.append(result['llm_result']['ratings'][metric])
                
                if human_scores:
                    correlations[f"rating_{metric}"] = self.calculate_correlation(
                        human_scores, llm_scores
                    )
        
        # Comparison correlations
        if comparison_results:
            comparison_metrics = ['perspective_representation', 'informativeness',
                                'neutrality_balance', 'policy_approval']
            
            for metric in comparison_metrics:
                human_choices = []
                llm_choices = []
                
                for result in comparison_results:
                    # Check if both human and LLM comparisons exist and are successful
                    if (metric in result['human_comparisons'] and 
                        result['llm_result']['status'] == 'success' and
                        metric in result['llm_result']['comparisons']):
                        human_choices.append(result['human_comparisons'][metric])
                        llm_choices.append(result['llm_result']['comparisons'][metric])
                
                if human_choices:
                    correlations[f"comparison_{metric}"] = self.calculate_correlation(
                        human_choices, llm_choices
                    )
        
        return correlations

    def run_correlation_experiment(self, annotation_path: Path, output_path: Path, 
                                  sample_size: int = None) -> Dict[str, Any]:
        """
        Run the full correlation experiment
        
        Args:
            annotation_path: Path to human annotation data
            output_path: Path to save results
            sample_size: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with experiment results
        """
        print(f"Loading human annotations from {annotation_path}")
        rating_annotations, comparison_annotations = self.data_processor.load_human_annotations(annotation_path)
        
        print(f"Found {len(rating_annotations)} rating annotations")
        print(f"Found {len(comparison_annotations)} comparison annotations")
        
        # Sample if needed
        if sample_size:
            import random
            random.seed(42)  # For reproducibility
            if len(rating_annotations) > sample_size:
                rating_annotations = random.sample(rating_annotations, sample_size)
            if len(comparison_annotations) > sample_size:
                comparison_annotations = random.sample(comparison_annotations, sample_size)
        
        # Process rating annotations
        rating_results = self.process_rating_annotations(rating_annotations, annotation_path)
        
        # Process comparison annotations
        comparison_results = self.process_comparison_annotations(comparison_annotations, annotation_path)
        
        # Calculate correlations
        correlations = self.calculate_all_correlations(rating_results, comparison_results)
        
        # Save results
        experiment_results = {
            "experiment_metadata": {
                "model": self.model,
                "temperature": self.temperature,
                "timestamp": datetime.now().isoformat(),
                "n_rating_samples": len(rating_results),
                "n_comparison_samples": len(comparison_results)
            },
            "rating_results": rating_results,
            "comparison_results": comparison_results,
            "correlations": correlations
        }
        
        # Save to file with model name only (no timestamp)
        output_file = output_path / f"human_llm_correlation_{self.model}.json"
        with open(output_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        return experiment_results