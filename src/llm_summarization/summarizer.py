import sys
import os
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.LanguageModel import LanguageModel
from utils.prompts.summarization import SummarizationPrompt


class CommentSummarizer:
    """
    A class to generate summaries from comments using LLM models.
    Supports different summarization strategies and models.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the summarizer.
        
        Args:
            model: The LLM model to use for summarization
            system_prompt: Custom system prompt for the model
        """
        self.model = model
        self.system_prompt = system_prompt
        
        # Auto-adjust temperature for GPT-5 models
        temperature = self._get_optimal_temperature(model)
        
        self.client = LanguageModel(model_name=model, temperature=temperature)
    
    def _get_optimal_temperature(self, model: str) -> float:
        """
        Get optimal temperature setting for the given model.
        
        Args:
            model: Model name
            
        Returns:
            Optimal temperature value
        """
        model_lower = model.lower()
        
        # GPT-5 models only support temperature=1 (default)
        if "gpt-5" in model_lower or "gpt5" in model_lower:
            print(f"  Auto-adjusting temperature to 1.0 for GPT-5 model: {model}")
            return 1.0
        
        # For other models, use default temperature
        return 0.7
    
    def summarize_topic_modeling(self, comments: str) -> str:
        """
        Generate topic modeling summary from comments.
        
        Args:
            comments: The comments to analyze
            
        Returns:
            Generated summary with detected topics
        """
        prompt = SummarizationPrompt(comments=comments, system_prompt=self.system_prompt)
        system_prompt, user_input = prompt.topic_modeling_prompt_first_round()
        
        try:
            response = self.client.chat_completion(system_prompt=system_prompt, input_text=user_input)
            return response
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def summarize_merge_topics(self, comments: str, topics: str) -> str:
        """
        Merge and consolidate topics from previous analysis.
        
        Args:
            comments: Original comments
            topics: Topics from first round analysis
            
        Returns:
            Consolidated topics summary
        """
        prompt = SummarizationPrompt(comments=comments, system_prompt=self.system_prompt)
        system_prompt, user_input = prompt.topic_modeling_prompt_second_round(topics)
        
        try:
            response = self.client.chat_completion(system_prompt=system_prompt, input_text=user_input)
            return response
        except Exception as e:
            return f"Error merging topics: {str(e)}"
    
    def summarize_main_points(self, comments: str) -> str:
        """
        Generate comprehensive summary with topic modeling and agreement analysis.
        
        Args:
            comments: The comments to analyze (should include voting data)
            
        Returns:
            Comprehensive summary with topics and agreement analysis
        """
        prompt = SummarizationPrompt(comments=comments, system_prompt=self.system_prompt)
        system_prompt, user_input = prompt.summarizing_main_points_prompt_from_comments()
        
        try:
            response = self.client.chat_completion(system_prompt=system_prompt, input_text=user_input)
            return response
        except Exception as e:
            return f"Error generating main points summary: {str(e)}"
    
    def summarize_with_custom_prompt(self, comments: str, custom_system_prompt: str, custom_user_prompt: str) -> str:
        """
        Generate summary using custom prompts.
        
        Args:
            comments: The comments to analyze
            custom_system_prompt: Custom system prompt
            custom_user_prompt: Custom user prompt (can include {comments} placeholder)
            
        Returns:
            Generated summary
        """
        try:
            # Replace {comments} placeholder in user prompt
            formatted_user_prompt = custom_user_prompt.format(comments=comments)
            
            response = self.client.chat_completion(
                system_prompt=custom_system_prompt, 
                input_text=formatted_user_prompt
            )
            return response
        except Exception as e:
            return f"Error generating custom summary: {str(e)}"


def main():
    """Example usage of the CommentSummarizer."""
    
    # Sample comments with voting data
    sample_comments = """
    Comment 1: Machine learning is the future of technology. Group 0: 80% agree, 20% disagree. Group 1: 75% agree, 25% disagree.
    Comment 2: AI will replace human jobs. Group 0: 60% agree, 40% disagree. Group 1: 45% agree, 55% disagree.
    Comment 3: Neural networks are overhyped. Group 0: 30% agree, 70% disagree. Group 1: 40% agree, 60% disagree.
    """
    
    # Initialize summarizer
    summarizer = CommentSummarizer(model="gpt-4o-mini")
    
    print("=== TOPIC MODELING SUMMARY ===")
    topic_summary = summarizer.summarize_topic_modeling(sample_comments)
    print(topic_summary)
    
    print("\n=== MAIN POINTS SUMMARY ===")
    main_summary = summarizer.summarize_main_points(sample_comments)
    print(main_summary)
    
    print("\n=== CUSTOM PROMPT SUMMARY ===")
    custom_summary = summarizer.summarize_with_custom_prompt(
        comments=sample_comments,
        custom_system_prompt="You are an expert data analyst.",
        custom_user_prompt="Analyze the sentiment in these comments: {comments}"
    )
    print(custom_summary)


if __name__ == "__main__":
    main()
