from typing import Dict, Any
from .tools import call_gemini_summarize, call_gemini_generate_mcqs, evaluate_mcq

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
            # evaluate each question and attach score/evaluation
            evaluated = [evaluate_mcq(q, context_text=summary) for q in questions]
            return {"mode": mode, "summary": summary, "questions": evaluated}
        else:
            questions = call_gemini_generate_mcqs(content, num_questions=num_questions, model_name=self.model_name)
            evaluated = [evaluate_mcq(q, context_text=content) for q in questions]
            return {"mode": mode, "questions": evaluated}
