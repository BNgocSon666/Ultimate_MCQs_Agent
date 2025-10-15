from typing import Dict, Any
from .tools import call_gemini_summarize, call_gemini_generate_mcqs

class Agent:
    SUMMARY_THRESHOLD_CHARS = 3000

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    def decide_and_run(self, content: str, num_questions: int = 5, force_summary: bool = False) -> Dict[str, Any]:
        if force_summary or len(content) > self.SUMMARY_THRESHOLD_CHARS:
            mode = 'summary+mcqs'
        else:
            mode = 'mcqs'

        if mode == 'summary+mcqs':
            summary = call_gemini_summarize(content, model_name=self.model_name)
            questions = call_gemini_generate_mcqs(summary, num_questions=num_questions, model_name=self.model_name)
            return {"mode": mode, "summary": summary, "questions": questions}
        else:
            questions = call_gemini_generate_mcqs(content, num_questions=num_questions, model_name=self.model_name)
            return {"mode": mode, "questions": questions}
