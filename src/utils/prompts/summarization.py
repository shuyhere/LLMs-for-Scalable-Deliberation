from .base_prompt import BasePrompt

class SummarizationPrompt(BasePrompt):
    """
    Prompt class for summarization tasks that inherits from BasePrompt.
    Comments are passed as external information.
    """
    
    def __init__(self, comments: str, system_prompt: str = "You are a helpful assistant"):
        """
        Initialize with comments and optional system prompt.
        
        Args:
            comments: The comments text to be summarized
            system_prompt: Custom system prompt (defaults to "You are a helpful assistant")
        """
        external_info = {"comments": comments}
        super().__init__(external_info)
        self.system_prompt = system_prompt

    def get_system_prompt(self) -> str:
        """Return the system prompt."""
        return self.system_prompt
    
    def get_user_input(self) -> str:
        """Required abstract method - defaults to first round behavior."""
        return self.get_user_input_first_round()
    
    def get_user_input_first_round(self) -> str:
        """Generate user input with comments from external info."""
        return f"I want you to do topic modeling on the given comments. Print the detected topics line by line. Here are the comments: {self.get_external_info('comments')}"
    
    def get_user_input_second_round(self) -> str:
        """Generate user input for second round topic modeling."""
        return f"I want you to merge the given lists of topics into smallest set of topics that are comprehensive. Here are the lists of topics: {self.get_external_info('topics')}"
    
    def topic_modeling_prompt_first_round(self) -> tuple[str, str]:
        """
        Get the prompt pair for the first round of topic modeling.
        
        Returns:
            Tuple of (system_prompt, user_input)
        """
        return self.get_system_prompt(), self.get_user_input_first_round()
    
    def topic_modeling_prompt_second_round(self, topics: str) -> tuple[str, str]:
        """
        Get the prompt pair for the second round of topic modeling.
        
        Args:
            topics: The topics from the first round to be merged
            
        Returns:
            Tuple of (system_prompt, user_input)
        """
        # Update external info with the new topics
        self.update_external_info({"topics": topics})
        
        # Return system prompt and the second round user input
        return self.get_system_prompt(), self.get_user_input_second_round()
    
    def get_user_input_summarizing_main_points(self) -> str:
        """Generate user input for summarizing main points with topic modeling and agreement analysis."""
        return f"""In each line, I provide you with comments and percentage of votes that agreed and disagreed with them for Group 0 and Group 1. I want you to do topic modeling on the given comments. Print the detected topics line by line. At the end, generate an overall summary of the comments. In the summary, make sure to include information and quantification on how much agreement versus disagreement there was across Group 0 and Group 1 for different topics. Here are the comments: {self.get_external_info('comments')}"""

    def summarizing_main_points_prompt(self) -> tuple[str, str]:
        """
        Get the prompt pair for summarizing main points.
        
        Returns:
            Tuple of (system_prompt, user_input)
        """
        return self.get_system_prompt(), self.get_user_input_summarizing_main_points()

if __name__ == "__main__":
    # Example usage - First round
    prompt = SummarizationPrompt(
        comments="This is a sample comment about machine learning and AI."
    )
    
    system_prompt, user_input = prompt.topic_modeling_prompt_first_round()
    print("=== FIRST ROUND ===")
    print("System Prompt:", system_prompt)
    print("User Input:", user_input)
    
    # Example usage - Second round
    topics_from_first_round = "machine learning, artificial intelligence, neural networks, deep learning"
    system_prompt2, user_input2 = prompt.topic_modeling_prompt_second_round(topics_from_first_round)
    print("\n=== SECOND ROUND ===")
    print("System Prompt:", system_prompt2)
    print("User Input:", user_input2)
    
    # Example with custom system prompt
    custom_prompt = SummarizationPrompt(
        comments="Another comment about deep learning.",
        system_prompt="You are an expert data analyst specializing in text analysis."
    )
    
    custom_system, custom_user = custom_prompt.topic_modeling_prompt_first_round()
    print("\n=== CUSTOM SYSTEM PROMPT ===")
    print("Custom System Prompt:", custom_system)
    print("Custom User Input:", custom_user)
    
    # Example usage - Summarizing main points
    print("\n=== SUMMARIZING MAIN POINTS ===")
    system_prompt3, user_input3 = prompt.summarizing_main_points_prompt()
    print("System Prompt:", system_prompt3)
    print("User Input:", user_input3)