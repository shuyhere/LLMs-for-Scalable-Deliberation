from .base_prompt import BasePrompt
import re
import json
from pathlib import Path
from typing import Dict, List


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
				 task_type: str = "rating",
				 assigned_user_path: str = "",
				 comparison_entry_id: str = ""):
		"""
		Initialize the prompt for human-like annotation
		
		Args:
			summary: Single summary for rating tasks
			question: The original question being answered
			summary_a: First summary for comparison tasks
			summary_b: Second summary for comparison tasks
			annotator_answer: The annotator's own answer to the question
			task_type: Either "rating" or "comparison"
			assigned_user_path: Path to assigned_user_data.json for exact texts
			comparison_entry_id: Key like "triplet_47_comparison" to load A/B and question
		"""
		external_info = {
			"summary": summary,
			"question": question,
			"summary_a": summary_a,
			"summary_b": summary_b,
			"annotator_answer": annotator_answer,
			"task_type": task_type,
			"assigned_user_path": assigned_user_path,
			"comparison_entry_id": comparison_entry_id,
		}
		super().__init__(external_info=external_info)
		self.label_catalog = self._load_label_catalog()
	
	def _clean_html_text(self, text: str) -> str:
		"""Clean HTML tags and formatting from text"""
		if not text:
			return ""
		
		# Remove HTML tags
		clean_text = re.sub(r'<[^>]+>', '', text)
		
		# Clean up extra whitespace
		clean_text = re.sub(r'\s+', ' ', clean_text).strip()
		
		return clean_text

	def _normalize_question(self, question: str) -> str:
		if not question:
			return question
		# Remove trailing instruction like "Please answer briefly in 2–3 sentences."
		question = re.sub(r"\s*Please\s+answer\s+briefly\s+in\s+2[–-]3\s+sentences\.?\s*$", "", question, flags=re.IGNORECASE)
		return question.strip()

	def _normalize_summary(self, text: str) -> str:
		if not text:
			return text
		# Remove duplicated boilerplate prefixes
		patterns = [
			r"^Below\s+is\s+a\s+summary\s+of\s+people's\s+opinions\s+on\s+the\s+issue\.?\s*",
			r"^Below\s+is\s+a\s+summary\s+of\s+all\s+people's\s+opinions\s+on\s+the\s+issue\.?\s*",
			r"^Here\s+is\s+an\s+overall\s+summary\s+of\s+the\s+comments\s+provided:\s*",
		]
		normalized = text
		changed = True
		while changed:
			changed = False
			for pat in patterns:
				new_norm = re.sub(pat, "", normalized, flags=re.IGNORECASE)
				if new_norm != normalized:
					normalized = new_norm
					changed = True
		# Collapse extra spaces after removals
		normalized = re.sub(r"\s+", " ", normalized).strip()
		return normalized
	
	def _load_label_catalog(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
		# Default path synchronized with data_process script output
		default_report = Path("datasets/sft_annotation_format_full_augment/labels_report.json")
		if default_report.exists():
			try:
				data = json.loads(default_report.read_text(encoding="utf-8"))
				if isinstance(data, dict) and "rating" in data and "comparison" in data:
					return data
			except Exception:
				pass
		return {"rating": {}, "comparison": {}}
	
	def _format_options_block(self, section: str, human_q: str) -> str:
		mapping = self.label_catalog.get(section, {}).get(human_q, {})
		if not isinstance(mapping, dict):
			return ""
		lines: List[str] = []
		for i in range(1, 6):
			k = str(i)
			if k in mapping and mapping[k]:
				descs = sorted(mapping[k])
				lines.append(f"{i}: {' | '.join(descs)}")
		return "\n".join(lines)
	
	def _load_comparison_from_assigned(self) -> tuple[str, str, str]:
		assigned_path = self.external_info.get("assigned_user_path") or ""
		entry_id = self.external_info.get("comparison_entry_id") or ""
		if not assigned_path or not entry_id:
			return "", "", ""
		try:
			p = Path(assigned_path)
			if not p.exists():
				return "", "", ""
			data = json.loads(p.read_text(encoding="utf-8"))
			info = data.get(entry_id, {})
			sum_a = info.get("summary_a_text", "")
			sum_b = info.get("summary_b_text", "")
			question = info.get("question", "")
			return sum_a, sum_b, question
		except Exception:
			return "", "", ""
	
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
		question = self._normalize_question(self.external_info.get("question", ""))
		annotator_answer = self.external_info.get("annotator_answer", "")
		
		# Clean the summary text (remove HTML tags and formatting)
		clean_summary = self._normalize_summary(self._clean_html_text(summary))
		
		# Include annotator's perspective if available
		perspective_text = ""
		if annotator_answer:
			perspective_text = f"""

One annotator's opinion on this question is:
{annotator_answer}
"""
		
		return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Below is a summary of all people's opinions on the issue.

{clean_summary}

Please evaluate this summary on the following 4 criteria using a 1-5 scale. For each criterion, choose one of the exact options shown:

1. **To what extent is the annotator's opinion represented in this response?**
Options:
{self._format_options_block('rating', 'To what extent is your perspective represented in this response?')}

2. **How informative is this summary?**
Options:
{self._format_options_block('rating', 'How informative is this summary?')}

3. **Do you think this summary presents a neutral and balanced view of the issue?**
Options:
{self._format_options_block('rating', 'Do you think this summary presents a neutral and balanced view of the issue?')}

4. **Would the annotator approve of this summary being used by the policy makers to make decisions relevant to the issue?**
Options:
{self._format_options_block('rating', 'Would you approve of this summary being used by the policy makers to make decisions relevant to the issue?')}

Please provide your evaluation in the following JSON format:
```json
{{
    "Representiveness": <1-5>,
    "Informativeness": <1-5>,
    "Neutrality": <1-5>,
    "Policy Approval": <1-5>
}}
```

Important: Respond ONLY with the JSON object, no additional text."""
	
	def get_comparison_prompt(self) -> str:
		"""Get prompt for comparing two summaries - matches human annotation format"""
		# Prefer loading from assigned file for exact raw texts
		loaded_a, loaded_b, loaded_q = self._load_comparison_from_assigned()
		summary_a = loaded_a or self.external_info.get("summary_a", "")
		summary_b = loaded_b or self.external_info.get("summary_b", "")
		question = self._normalize_question(loaded_q or self.external_info.get("question", ""))
		annotator_answer = self.external_info.get("annotator_answer", "")
		
		# Use raw (no cleaning) if loaded; otherwise clean provided strings
		clean_summary_a = summary_a if loaded_a else self._clean_html_text(summary_a)
		clean_summary_b = summary_b if loaded_b else self._clean_html_text(summary_b)
		clean_summary_a = self._normalize_summary(clean_summary_a)
		clean_summary_b = self._normalize_summary(clean_summary_b)
		
		# Include annotator's perspective if available
		perspective_text = ""
		if annotator_answer:
			perspective_text = f"""

One annotator's opinion on this question is:
{annotator_answer}
"""
		
		return f"""
We have made a deliberation with many annotators on the issue: {question}

{perspective_text}

Two summaries of opinions are shown below. Read carefully and answer according to the opinion of the annotator.

Summary A:
{clean_summary_a}

Summary B:
{clean_summary_b}

Please compare these summaries on the following 4 criteria. For each criterion, choose one of the exact options shown:

1. **Which summary is more representative of the annotator's opinion?**
Options:
{self._format_options_block('comparison', 'Which summary is more representative of your perspective?')}

2. **Which summary is more informative?**
Options:
{self._format_options_block('comparison', 'Which summary is more informative?')}

3. **Which summary presents a more neutral and balanced view of the issue?**
Options:
{self._format_options_block('comparison', 'Which summary presents a more neutral and balanced view of the issue?')}

4. **Which summary would the annotator prefer of being used by the policy makers to make decisions relevant to the issue?**
Options:
{self._format_options_block('comparison', 'Which summary would you prefer of being used by the policy makers to make decisions relevant to the issue?')}

Please provide your evaluation in the following JSON format:
```json
{{
    "Representiveness": <1-5>,
    "Informativeness": <1-5>,
    "Neutrality": <1-5>,
    "Policy Approval": <1-5>
}}
```

Important:
- Respond ONLY with the JSON object, no additional text."""
