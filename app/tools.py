import tempfile
import os
from typing import Tuple
import fitz
from docx import Document
from .utils import clean_text, check_file_size_bytes, safe_filename
from .config import GOOGLE_API_KEY, MAX_FILE_SIZE_MB
import google.generativeai as genai_old
import json
from google import genai  # SDK mới


if GOOGLE_API_KEY:
    genai_old.configure(api_key=GOOGLE_API_KEY)

client = genai.Client(api_key=GOOGLE_API_KEY)

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
    model = genai_old.GenerativeModel(model_name, tools=[])
    # Yêu cầu model tự phát hiện ngôn ngữ đầu vào và trả về bản tóm tắt bằng cùng ngôn ngữ
    prompt = (
        "Hãy phát hiện ngôn ngữ của văn bản dưới đây. Sau đó tạo một bản tóm tắt ngắn gọn, rõ ràng và đầy đủ bằng cùng một ngôn ngữ.\n"
        f"Nội dung:\n{text}"
    )
    resp = model.generate_content([prompt])
    return resp.text.strip()

def call_gemini_generate_mcqs(text: str, num_questions: int = 5, model_name: str = "gemini-2.5-flash") -> list:
    model = genai_old.GenerativeModel(model_name, tools=[])
    # Yêu cầu model phát hiện ngôn ngữ đầu vào và sinh câu hỏi trắc nghiệm bằng cùng ngôn ngữ
    prompt = (
        f"Hãy phát hiện ngôn ngữ của nội dung dưới đây. Sau đó sinh ra {num_questions} "
        "câu hỏi trắc nghiệm dựa trên nội dung, bằng cùng ngôn ngữ của nội dung. "
        "Trả về CHÍNH XÁC một mảng JSON (chỉ JSON) với các trường sau: "
        "\n- context: tóm tắt ngắn bối cảnh của câu hỏi "
        "\n- question: câu hỏi "
        "\n- options: mảng 4 phần tử, mỗi phần tử là lựa chọn bắt đầu bằng 'A.', 'B.', 'C.' hoặc 'D.' "
        "\n- answer: chứa  đáp án đúng bắt đầu bằng('A.', 'B.', 'C.' hoặc 'D.') "
        "\nKHÔNG bao gồm lời giải thích hoặc nội dung khác ngoài JSON.\n"
        f"Nội dung:\n{text}"
    )

    resp = model.generate_content([prompt])
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

# tools.py
def extract_text_from_audio_with_gemini(file_path: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Dùng Gemini 2.5 (SDK mới) để tóm tắt audio mà không lỗi ragStoreName.
    """
    prompt = (
        "Hãy nghe kỹ nội dung trong đoạn ghi âm này, "
        "phát hiện ngôn ngữ được nói và tóm tắt lại ngắn gọn, rõ ràng, đầy đủ bằng chính ngôn ngữ đó."
    )
    try:
        uploaded_file = client.files.upload(file=file_path)
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, uploaded_file]
        )
        return response.text.strip()
    except Exception as e:
        return f"[Lỗi Gemini audio summarize] {str(e)}"


def extract_transcript_from_audio_with_gemini(file_path: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Dùng Gemini 2.5 (SDK mới) để chép lại transcript audio mà không lỗi ragStoreName.
    """
    prompt = (
        "Hãy nghe toàn bộ đoạn ghi âm này và chép lại nguyên văn nội dung nói ra, "
        "giữ nguyên ngôn ngữ gốc của người nói, không tóm tắt, không bỏ sót chi tiết."
    )
    try:
        uploaded_file = client.files.upload(file=file_path)
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, uploaded_file]
        )
        return response.text.strip()
    except Exception as e:
        return f"[Lỗi Gemini transcript] {str(e)}"

def save_json_to_disk(obj, filename: str) -> str:
    out_path = os.path.join(tempfile.gettempdir(), filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path
