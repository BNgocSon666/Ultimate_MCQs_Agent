from enum import Enum
from .tools import call_gemini_summarize, call_gemini_generate_mcqs

class SummaryMode(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    NONE = "none"


class Agent:
    def decide_and_run(
        self,
        text: str,
        num_questions: int = 5,
        summary_mode: str = SummaryMode.AUTO,
        is_summary: bool = False
    ):
        """
        Quyết định pipeline xử lý:
        - Nếu is_summary=True → text đã là tóm tắt sẵn, chỉ sinh câu hỏi.
        - Nếu summary_mode="force" → luôn tóm tắt.
        - Nếu summary_mode="none" → không bao giờ tóm tắt.
        - Nếu summary_mode="auto" → tóm tắt nếu text dài > 3000 ký tự.
        """

        # Nếu input là bản tóm tắt đã có sẵn (ví dụ từ audio summary)
        if is_summary:
            mcqs = call_gemini_generate_mcqs(text, num_questions)
            return {
                "mode": "mcqs",
                "questions": mcqs
            }

        # Chế độ FORCE — luôn tóm tắt
        if summary_mode == SummaryMode.FORCE:
            summary = call_gemini_summarize(text)
            mcqs = call_gemini_generate_mcqs(summary, num_questions)
            return {
                "mode": "summary+mcqs",
                "summary": summary,
                "questions": mcqs
            }

        # Chế độ NONE — không bao giờ tóm tắt
        if summary_mode == SummaryMode.NONE:
            mcqs = call_gemini_generate_mcqs(text, num_questions)
            return {
                "mode": "mcqs",
                "questions": mcqs
            }

        # Chế độ AUTO — tóm tắt nếu text quá dài
        if len(text) > 3000:
            summary = call_gemini_summarize(text)
            mcqs = call_gemini_generate_mcqs(summary, num_questions)
            return {
                "mode": "summary+mcqs",
                "summary": summary,
                "questions": mcqs
            }

        # Mặc định: sinh câu hỏi trực tiếp
        mcqs = call_gemini_generate_mcqs(text, num_questions)
        return {
            "mode": "mcqs",
            "questions": mcqs
        }
