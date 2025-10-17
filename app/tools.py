import tempfile
import os
from typing import Tuple
import fitz
from docx import Document
from .utils import clean_text, check_file_size_bytes, safe_filename
from .config import GOOGLE_API_KEY, MAX_FILE_SIZE_MB
import google.generativeai as genai
import json

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf_bytes(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    text = ""
    try:
        with fitz.open(tmp_path) as pdf:
            for page in pdf:
                text += page.get_text()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text

def extract_text_from_docx_bytes(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    text = ""
    try:
        doc = Document(tmp_path)
        text = '\n'.join([p.text for p in doc.paragraphs])
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text

def extract_text_from_txt_bytes(content: bytes) -> str:
    return content.decode('utf-8', errors='ignore')

async def extract_and_clean_from_uploadfile(upload_file) -> Tuple[bool, str]:
    raw = await upload_file.read()
    ok, msg = check_file_size_bytes(raw, MAX_FILE_SIZE_MB)
    if not ok:
        return False, msg

    suffix = os.path.splitext(safe_filename(upload_file.filename))[1].lower()
    if suffix == '.pdf':
        text = extract_text_from_pdf_bytes(raw)
    elif suffix in ('.doc', '.docx'):
        text = extract_text_from_docx_bytes(raw)
    elif suffix == '.txt':
        text = extract_text_from_txt_bytes(raw)
    else:
        return False, 'Định dạng không hỗ trợ. Hỗ trợ: PDF, DOCX, TXT.'

    cleaned = clean_text(text)
    if not cleaned:
        return False, 'Không thể trích xuất nội dung từ file (file rỗng hoặc lỗi).'
    return True, cleaned

def call_gemini_summarize(text: str, model_name: str = "gemini-2.5-flash") -> str:
    model = genai.GenerativeModel(model_name)
    # Yêu cầu model tự phát hiện ngôn ngữ đầu vào và trả về bản tóm tắt bằng cùng ngôn ngữ (tiếng Việt nếu nguồn là tiếng Việt)
    prompt = (
        "Vui lòng phát hiện ngôn ngữ của văn bản dưới đây. Sau đó tạo một bản tóm tắt ngắn gọn, rõ ràng và đầy đủ bằng cùng ngôn ngữ với văn bản nguồn.\n"
        f"Nội dung:\n{text}"
    )
    resp = model.generate_content(prompt)
    return resp.text.strip()

def call_gemini_generate_mcqs(text: str, num_questions: int = 5, model_name: str = "gemini-2.5-flash") -> list:
    model = genai.GenerativeModel(model_name)
    # Yêu cầu model phát hiện ngôn ngữ đầu vào và sinh câu hỏi trắc nghiệm bằng cùng ngôn ngữ
    # Trả về đúng định dạng một mảng các đối tượng (chỉ trả về đoạn văn bản là mảng dữ liệu, không thêm giải thích):
    # mỗi đối tượng có các trường: context, question, options (mảng), answer
    prompt = (
        "Vui lòng phát hiện ngôn ngữ của nội dung dưới đây. Sau đó sinh ra "
        f"{num_questions} câu hỏi trắc nghiệm dựa trên nội dung, bằng cùng ngôn ngữ với nội dung nguồn. "
        "Chỉ trả về một mảng dữ liệu (không kèm chú thích) với các trường: context, question, options (mảng các lựa chọn), answer.\n"
        f"Nội dung:\n{text}"
    )
    resp = model.generate_content(prompt)
    mcq_text = resp.text.strip()

    if mcq_text.startswith("```json"):
        mcq_text = mcq_text[7:]
    if mcq_text.startswith("```"):
        mcq_text = mcq_text[3:]
    if mcq_text.endswith("```"):
        mcq_text = mcq_text[:-3]
    mcq_text = mcq_text.strip()

    try:
        data = json.loads(mcq_text)
    except Exception:
        data = [{
            "context": text[:200] + "...",
            "question": "Không thể tạo câu hỏi hợp lệ từ nội dung này.",
            "options": ["Lỗi", "Lỗi", "Lỗi", "Lỗi"],
            "answer": "A. Lỗi"
        }]
    return data

def save_json_to_disk(obj, filename: str) -> str:
    out_path = os.path.join(tempfile.gettempdir(), filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path


# Evaluation weights and thresholds
EVAL_WEIGHTS = {
    "accuracy": 50,
    "distractors": 20,
    "alignment": 25,
    "clarity": 5,
}

EVAL_THRESHOLDS = {
    "pass": 80,
    "need_review_min": 60,
}


def evaluate_mcq(mcq: dict, context_text: str = "") -> dict:
    """
    Đánh giá một câu hỏi trắc nghiệm (MCQ) và trả về một dict có 'score' (0-100) và 'evaluation' (nhận xét).

    Hàm này dùng các quy tắc heuristic đơn giản để chạy cục bộ (không gọi API):
    - accuracy (độ chính xác): kiểm tra xem đáp án có nằm trong các lựa chọn hay là ký hiệu chữ cái (A/B/C/D)
    - distractors (độ phân tán lựa chọn): kiểm tra các lựa chọn có trùng lặp hay tương đồng quá mức
    - alignment (sự phù hợp): kiểm tra mức độ chồng token giữa context và question/options
    - clarity (độ rõ ràng): kiểm tra độ dài câu hỏi và dấu câu (ví dụ dấu hỏi)

    Ngưỡng và trọng số đã được định nghĩa bằng các hằng phía trên.
    Nếu tổng điểm < 60 thì verdict = 'rejected' (vẫn giữ câu hỏi trong kết quả nhưng được gắn tag).

    Trả về bản sao của mcq đã được mở rộng với các trường: 'score', 'evaluation', 'status', 'tags', '_eval_breakdown'.
    """
    # Defensive defaults
    question = (mcq.get("question") or "").strip()
    options = mcq.get("options") or []
    answer = (mcq.get("answer") or "").strip()
    context = (mcq.get("context") or context_text or "").strip()

    # Phát hiện các trường hợp generator trả về placeholder lỗi hoặc dữ liệu không hợp lệ
    invalid_generated = False
    q_lower = question.lower()
    opt_lowers = [o.lower().strip() for o in options]
    # Nếu câu hỏi có chứa chuỗi báo lỗi hoặc tất cả lựa chọn giống nhau là 'lỗi', đánh dấu là invalid
    if any(s in q_lower for s in ["không thể tạo câu hỏi hợp lệ", "không thể tạo", "lỗi"]):
        invalid_generated = True
    if opt_lowers and len(set(opt_lowers)) == 1 and list(set(opt_lowers))[0] in ["lỗi", "error", "err"]:
        invalid_generated = True

    # Accuracy (50): true answer must be one of the options; prefer letter labels
    acc_score = 0
    normalized_opts = [str(o).strip() for o in options]
    # Extract letter from answer like 'A. X' or 'A'
    ans_letter = None
    if isinstance(answer, str) and answer:
        if answer and len(answer) >= 1 and answer[0].upper() in "ABCD":
            ans_letter = answer[0].upper()
    # Map letters to options
    letter_map = {}
    for idx, opt in enumerate(normalized_opts):
        letter = chr(ord('A') + idx)
        letter_map[letter] = opt

    if ans_letter and ans_letter in letter_map:
        # check that the answer text is non-empty and not the same as others
        ans_text = letter_map[ans_letter]
        if ans_text and ans_text.lower() not in [o.lower() for o in normalized_opts if o != ans_text]:
            acc_score = EVAL_WEIGHTS['accuracy']
        else:
            acc_score = int(EVAL_WEIGHTS['accuracy'] * 0.6)
    else:
        # if answer is full text, check membership
        if answer and any(answer.lower() in o.lower() or o.lower() in answer.lower() for o in normalized_opts):
            acc_score = int(EVAL_WEIGHTS['accuracy'] * 0.8)
        else:
            acc_score = 0

    # Distractors (20): penalize duplicates and extremely short/long options
    dist_score = EVAL_WEIGHTS['distractors']
    if len(normalized_opts) < 2:
        dist_score = 0
    else:
        uniq = len(set([o.lower() for o in normalized_opts]))
        dup_factor = uniq / max(1, len(normalized_opts))
        # length diversity penalty
        lengths = [len(o) for o in normalized_opts if o]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            len_penalty = max(0, 1 - (sum(abs(l - avg_len) for l in lengths) / (len(lengths) * max(1, avg_len))))
        else:
            len_penalty = 0
        dist_score = int(dist_score * dup_factor * len_penalty)

    # Alignment (20): token overlap between context and question/options
    align_score = 0
    try:
        ctx_tokens = set([t.lower() for t in context.split() if len(t) > 3])
        q_tokens = set([t.lower() for t in question.split() if len(t) > 3])
        opt_tokens = set()
        for o in normalized_opts:
            opt_tokens.update([t.lower() for t in o.split() if len(t) > 3])
        overlap = len((q_tokens | opt_tokens) & ctx_tokens)
        total = max(1, len(q_tokens | opt_tokens))
        align_score = int(EVAL_WEIGHTS['alignment'] * (overlap / total))
    except Exception:
        align_score = 0

    # Clarity (10): prefer question length between 20 and 150 chars and presence of question mark
    clarity_score = 0
    qlen = len(question)
    if 20 <= qlen <= 200:
        clarity_score = EVAL_WEIGHTS['clarity']
    else:
        # partial credit for short/long
        clarity_score = int(EVAL_WEIGHTS['clarity'] * max(0.0, 1 - abs(qlen - 80) / 200))
    if '?' in question:
        clarity_score = min(EVAL_WEIGHTS['clarity'], clarity_score + 2)

    total_score = acc_score + dist_score + align_score + clarity_score
    # clamp
    total_score = max(0, min(100, int(total_score)))

    # Nếu phát hiện là placeholder/error do generator, ép score về 0 và verdict là 'rejected'
    if invalid_generated:
        total_score = 0
        verdict = 'rejected'
    else:
        if total_score >= EVAL_THRESHOLDS['pass']:
            verdict = 'pass'
        elif total_score >= EVAL_THRESHOLDS['need_review_min']:
            verdict = 'need review'
        else:
            verdict = 'rejected'

    # Gắn kết quả vào object câu hỏi (không xóa câu hỏi ngay cả khi rejected)
    result = dict(mcq)
    result['score'] = total_score
    result['evaluation'] = verdict
    # Trạng thái/tags rõ ràng để dễ lọc: ví dụ 'rejected', 'pass', 'need review'
    result['status'] = verdict
    # tags là danh sách để dễ mở rộng về sau
    result['tags'] = [verdict]
    if invalid_generated:
        # tag thêm để dễ lọc các câu hỏi sinh ra là placeholder/error
        result['tags'].append('invalid_generated')
        result['_eval_notes'] = 'generator_returned_placeholder_or_error'
    # Chi tiết điểm từng tiêu chí (dùng để debug hoặc hiển thị nội bộ)
    result['_eval_breakdown'] = {
        'accuracy': acc_score,
        'distractors': dist_score,
        'alignment': align_score,
        'clarity': clarity_score,
    }
    return result
