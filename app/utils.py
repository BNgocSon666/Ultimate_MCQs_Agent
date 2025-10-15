import re
import os
from typing import Tuple

def clean_text(raw_text: str) -> str:
    """Clean control characters and collapse whitespace."""
    if not raw_text:
        return ""
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', raw_text)
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\t', ' ')
    text = ' '.join(text.split())
    return text

def check_file_size_bytes(content_bytes: bytes, max_mb: int) -> Tuple[bool, str]:
    max_bytes = max_mb * 1024 * 1024
    if len(content_bytes) > max_bytes:
        return False, f"File quá lớn (> {max_mb}MB)."
    return True, "OK"

def safe_filename(filename: str) -> str:
    return os.path.basename(filename)
