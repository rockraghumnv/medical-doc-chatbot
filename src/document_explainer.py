import base64
import io
import re
from typing import Tuple

from PIL import Image
from langchain_core.messages import HumanMessage

from src.document_prompt import document_explainer_prompt


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 6 * 1024 * 1024
MAX_EXPLANATION_CHARS = 1800


def validate_uploaded_image(file_storage) -> Tuple[bytes, str]:
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValueError("Please upload an image file (jpg, jpeg, png, webp).")

    filename = file_storage.filename.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError("Unsupported file type. Use jpg, jpeg, png, or webp.")

    content = file_storage.read()
    if not content:
        raise ValueError("Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Image too large. Please upload a file under 6 MB.")

    return content, filename


def _normalize_image_bytes(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def extract_text_from_medical_image(chat_model, image_bytes: bytes, question: str) -> str:
    normalized = _normalize_image_bytes(image_bytes)
    b64 = base64.b64encode(normalized).decode("utf-8")

    extraction_prompt = (
        "Extract all medically relevant text from this document image. "
        "Include test names, values, units, reference ranges, date, medicine names, dosage, "
        "and instructions if present. If uncertain, mention [unclear]."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": extraction_prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
            {"type": "text", "text": f"User goal: {question}"},
        ]
    )

    result = chat_model.invoke([message])
    extracted_text = (result.content or "").strip()

    if not extracted_text:
        raise ValueError("I could not read text from the uploaded image. Please upload a clearer image.")

    return extracted_text


def _sanitize_plain_text(text: str) -> str:
    cleaned = text.replace("\r", "")

    sanitized_lines = []
    for line in cleaned.split("\n"):
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        sanitized_lines.append(line.strip())

    cleaned = "\n".join(sanitized_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _truncate_text(text: str, max_chars: int = MAX_EXPLANATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_sentence_end = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_sentence_end > int(max_chars * 0.7):
        truncated = truncated[:last_sentence_end + 1]

    return f"{truncated.strip()}\n\nNote: Response was shortened for readability."


def explain_document_simple(chat_model, retriever, extracted_text: str, question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else ""

    final_prompt = document_explainer_prompt.format(
        document_text=extracted_text,
        context=context,
        question=question,
    )

    response = chat_model.invoke(final_prompt)
    raw_text = (response.content or "").strip()
    plain_text = _sanitize_plain_text(raw_text)
    return _truncate_text(plain_text)
