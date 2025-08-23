from .base_prompt import BasePrompt


class EvaluationPrompt(BasePrompt):
    """
    A prompt class for evaluating how well comments are represented in summaries.
    """
    
    def __init__(self, summary: str = "", comment: str = "", system_prompt: str = "You are a helpful assistant"):
        """
        Initialize the evaluation prompt.
        
        Args:
            summary: The summary to evaluate against
            comment: The comment to evaluate representation of
            system_prompt: Custom system prompt for the model
        """
        external_info = {
            "summary": summary,
            "comment": comment
        }
        super().__init__(external_info=external_info)
        self.system_prompt = system_prompt
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for evaluation."""
        return self.system_prompt
    
    def get_user_input(self) -> str:
        """Get the user input for evaluation (defaults to summary evaluation)."""
        return self.get_user_input_summary_evaluation()
    
    def get_summary_evaluation_from_comments(self) -> str:
        """
        Get the user input for evaluating comment representation in summary.
        
        Returns:
            Formatted user input for summary evaluation
        """
        summary = self.external_info.get("summary", "")
        comment = self.external_info.get("comment", "")
        
        return f"""I'm going to ask you to evaluate whether a comment by a digital town hall participant is represented in a summary of the town hall.

Here is the summary:
<summary>
{summary}
</summary>

Here is the comment:
<comment>
{comment}
</comment>

How well is the comment represented in the summary? Would someone who had read the summary gain new information as a result of reading the comment? Please choose one of the following options:

(1) The comment is not represented at all; the summary ignored the comment entirely.
(2) The summary contains some material relevant to the comment but is missing most of the content.
(3) The summary substantially represents the comment but is still missing something important.
(4) The summary covers most of the information in the comment, but is missing some nuance or detail.
(5) The summary covers all the information in the comment.

Please respond with full reasoning and one of these options. Put your final option inside \\boxed{{option}}."""
    
    def summary_evaluation_prompt_from_comments(self) -> tuple[str, str]:
        """
        Get the complete prompt for summary evaluation.
        
        Returns:
            Tuple of (system_prompt, user_input)
        """
        return self.get_system_prompt(), self.get_summary_evaluation_from_comments()
