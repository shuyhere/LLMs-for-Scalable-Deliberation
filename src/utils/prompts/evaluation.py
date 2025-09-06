from .base_prompt import BasePrompt
import re


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
        return self.get_summary_evaluation_from_comments()
    
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


class HumanAnnotationPrompt(BasePrompt):
    """
    Prompt class for mimicking human annotation tasks with 4 rating questions
    """
    
    def __init__(self, summary: str = "", question: str = "", 
                 summary_a: str = "", summary_b: str = "", 
                 annotator_answer: str = "",
                 task_type: str = "rating"):
        """
        Initialize the prompt for human-like annotation
        
        Args:
            summary: Single summary for rating tasks
            question: The original question being answered
            summary_a: First summary for comparison tasks
            summary_b: Second summary for comparison tasks
            annotator_answer: The annotator's own answer to the question
            task_type: Either "rating" or "comparison"
        """
        external_info = {
            "summary": summary,
            "question": question,
            "summary_a": summary_a,
            "summary_b": summary_b,
            "annotator_answer": annotator_answer,
            "task_type": task_type
        }
        super().__init__(external_info=external_info)
    
    def _clean_html_text(self, text: str) -> str:
        """Clean HTML tags and formatting from text"""
        if not text:
            return ""
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the evaluation task"""
        return "You are an expert evaluator analyzing summaries of public opinions."
    
    def get_user_input(self) -> str:
        """Get the user input based on task type"""
        if self.external_info.get("task_type") == "rating":
            return self.get_rating_prompt()
        else:
            return self.get_comparison_prompt()
    
    def get_rating_prompt(self) -> str:
        """Get prompt for rating a single summary - matches human annotation format"""
        summary = self.external_info.get("summary", "")
        question = self.external_info.get("question", "")
        annotator_answer = self.external_info.get("annotator_answer", "")
        
        # Clean the summary text (remove HTML tags and formatting)
        clean_summary = self._clean_html_text(summary)
        
        # Include annotator's perspective if available
        perspective_text = ""
        if annotator_answer:
            perspective_text = f"""\n\nOne annotator's opinion on this question is:
{annotator_answer}\n"""
        
        return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Below is a summary of all people's opinions on the issue.

{clean_summary}

Please evaluate this summary on the following 4 criteria using a 1-5 scale:

1. **To what extent is the annotator's opinion represented in this response?**
   (1 = Not at all, 2 = Slightly, 3 = Moderately, 4 = Well, 5 = Very well)

2. **How informative is this summary?**
   (1 = Not informative, 2 = Slightly informative, 3 = Moderately informative, 4 = Very informative, 5 = Extremely informative)

3. **Do you think this summary presents a neutral and balanced view of the issue?**
   (1 = Very biased, 2 = Somewhat biased, 3 = Neutral, 4 = Fairly balanced, 5 = Very balanced)

4. **Would the annotator approve of this summary being used by the policy makers to make decisions relevant to the issue?**
   (1 = Strongly disapprove, 2 = Disapprove, 3 = Neutral, 4 = Approve, 5 = Strongly approve)

Please provide your evaluation in the following JSON format:
```json
{{
    "perspective_representation": <1-5>,
    "informativeness": <1-5>,
    "neutrality_balance": <1-5>,
    "policy_approval": <1-5>
}}
```

Important: Respond ONLY with the JSON object, no additional text."""
    
    def get_comparison_prompt(self) -> str:
        """Get prompt for comparing two summaries - matches human annotation format"""
        summary_a = self.external_info.get("summary_a", "")
        summary_b = self.external_info.get("summary_b", "")
        question = self.external_info.get("question", "")
        annotator_answer = self.external_info.get("annotator_answer", "")
        
        # Clean the summary texts
        clean_summary_a = self._clean_html_text(summary_a)
        clean_summary_b = self._clean_html_text(summary_b)
        
        # Include annotator's perspective if available
        perspective_text = ""
        if annotator_answer:
            perspective_text = f"""\n\nOne annotator's opinion on this question is:
{annotator_answer}\n"""
        
        return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Two summaries of all people's opinions are shown below. Read carefully and answer according to your prior opinion.

Summary A:
{clean_summary_a}

Summary B:
{clean_summary_b}

Please compare these summaries on the following 4 criteria. For each criterion, choose which summary is better (1 for Summary A, 2 for Summary B):

1. **Which summary is more representative of the annotator's opinion?**
2. **Which summary is more informative?**
3. **Which summary presents a more neutral and balanced view of the issue?**
4. **Which summary would the annotator prefer of being used by the policy makers to make decisions relevant to the issue?**

Please provide your evaluation in the following JSON format:
```json
{{
    "perspective_representation": <1 or 2>,
    "informativeness": <1 or 2>,
    "neutrality_balance": <1 or 2>,
    "policy_approval": <1 or 2>
}}
```

Important: 
- Use 1 for Summary A, 2 for Summary B
- Respond ONLY with the JSON object, no additional text."""
