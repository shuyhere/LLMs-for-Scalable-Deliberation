import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.LanguageModel import LanguageModel
from utils.prompts.evaluation import EvaluationPrompt


class SummaryEvaluator:
    """
    A class to evaluate how well comments are represented in summaries using LLM models.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the evaluator.
        
        Args:
            model: The LLM model to use for evaluation
            system_prompt: Custom system prompt for the model
        """
        self.model = model
        self.system_prompt = system_prompt
        self.client = LanguageModel(model_name=model)
    
    def evaluate_comment_representation(self, summary: str, comment: str) -> str:
        """
        Evaluate how well a comment is represented in a summary.
        
        Args:
            summary: The summary to evaluate against
            comment: The comment to evaluate representation of
            
        Returns:
            Evaluation result with score (1-5) and reasoning
        """
        prompt = EvaluationPrompt(summary=summary, comment=comment, system_prompt=self.system_prompt)
        system_prompt, user_input = prompt.summary_evaluation_prompt_from_comments()
        print(f"System prompt: {system_prompt}")
        print(f"User input: {user_input}")
        
        # Call the language model
        response = self.client.chat_completion(system_prompt=system_prompt, input_text=user_input)
        
        # Check if response is None (API call failed)
        if response is None:
            raise Exception("Language model returned None - API call failed")
        
        return response
    
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
                response = self.evaluate_comment_representation(summary, comment)
                
                # Try to extract the score from the response
                score = self._extract_score_from_response(response)
                
                result = {
                    "comment_index": i,
                    "comment": comment,
                    "evaluation_response": response,
                    "score": score,
                    "status": "success"
                }
            except Exception as e:
                result = {
                    "comment_index": i,
                    "comment": comment,
                    "evaluation_response": f"Error: {str(e)}",
                    "score": None,
                    "status": "error"
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


def main():
    """Example usage of the SummaryEvaluator."""
    
    # Sample summary and comments
    sample_summary = """
    The digital town hall discussion covered several key topics:
    1. Machine learning and AI development - most participants agreed this is important for future technology
    2. Job displacement concerns - there was mixed agreement about AI replacing human jobs
    3. Technology hype - some participants felt neural networks are overhyped while others disagreed
    """
    
    sample_comments = [
        "Machine learning is the future of technology and we should invest heavily in it.",
        "AI will definitely replace most human jobs within the next decade.",
        "Neural networks are overhyped and won't solve real-world problems.",
        "We need to balance AI development with human welfare considerations."
    ]
    
    # Initialize evaluator
    evaluator = SummaryEvaluator(model="gpt-4o-mini")
    
    print("=== SINGLE COMMENT EVALUATION ===")
    single_eval = evaluator.evaluate_comment_representation(sample_summary, sample_comments[0])
    print(f"Comment: {sample_comments[0]}")
    print(f"Evaluation: {single_eval}")
    
    print("\n=== MULTIPLE COMMENTS EVALUATION ===")
    multiple_evals = evaluator.evaluate_multiple_comments(sample_summary, sample_comments)
    
    for result in multiple_evals:
        print(f"\nComment {result['comment_index'] + 1}: {result['comment'][:50]}...")
        print(f"Score: {result['score']}")
        print(f"Response: {result['evaluation_response']}")
    
    print("\n=== EVALUATION STATISTICS ===")
    stats = evaluator.get_evaluation_statistics(multiple_evals)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
