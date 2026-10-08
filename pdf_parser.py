"""
DocTalk — PDF Parser
Extracts and chunks text from PDFs.
Fallback chain: pdfplumber → pytesseract OCR → Gemini Vision (for handwriting)
"""
 
import base64
import re
import tempfile
from pathlib import Path
from typing import List, Optional
 
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import google.generativeai as genai
 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
 
# Confidence threshold — if tesseract avg confidence is below this, use Gemini Vision
TESSERACT_CONFIDENCE_THRESHOLD = 50
 
 
def extract_text_by_page(pdf_path: str, gemini_api_key: Optional[str] = None) -> List[dict]:
    """
    Extract text from each page using a 3-tier fallback:
      1. pdfplumber   — digital PDFs (fast, perfect accuracy)
      2. pytesseract  — scanned/printed pages (decent for clean prints)
      3. Gemini Vision — handwritten or low-confidence pages (best for messy handwriting)
 
    Returns:
        [{"page": int, "text": str, "method": str}, ...]
    """
    pages = []
 
    with pdfplumber.open(pdf_path) as pdf:
        pil_images = None  # lazy-load only if needed
 
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
 
            # ── Tier 1: pdfplumber ──────────────────────────────────────
            text = page.extract_text() or ""
            text = _clean_text(text)
 
            if text.strip():
                pages.append({"page": page_num, "text": text, "method": "pdfplumber"})
                continue
 
            # ── Tier 2: pytesseract ─────────────────────────────────────
            if pil_images is None:
                pil_images = convert_from_path(pdf_path, dpi=300)
 
            pil_image = pil_images[i]
            tess_text, confidence = _tesseract_ocr(pil_image)
 
            if tess_text.strip() and confidence >= TESSERACT_CONFIDENCE_THRESHOLD:
                pages.append({"page": page_num, "text": tess_text, "method": "tesseract"})
                continue
 
            # ── Tier 3: Gemini Vision ───────────────────────────────────
            if gemini_api_key:
                print(f"Page {page_num}: low OCR confidence ({confidence:.0f}%) — trying Gemini Vision…")
                gemini_text = _gemini_vision_ocr(pil_image, gemini_api_key)
                if gemini_text.strip():
                    pages.append({"page": page_num, "text": gemini_text, "method": "gemini_vision"})
                    continue
 
            # All tiers failed — skip page
            print(f"Page {page_num}: no text extracted, skipping.")
 
    return pages

def chunk_pages(
        pages: List[dict],
        chunk_size: int= 500,
        chunk_overlap: int= 50,
) -> List[dict]:
    
    """
    splits page texts into overlapping chunks for embedding.

    Args:
        pages: output of extract_text_by_page()
        chunk_size: Target chunk size in words.
        chunk_overlap: number of words to overlap between consecutives chunks.

    Returns:
        [{"chunk_id": int, "page": int, "text":str}, ...]
    """
    chunks =[]
    chunk_id=0

    for page_dict in pages:
        words = page_dict["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                "chunk_id": chunk_id,
                "page": page_dict["page"],
                "text": chunk_text,
                }
            )
            chunk_id += 1
            start += chunk_size - chunk_overlap #side forward with overlap

    return chunks

def parse_pdf(
        pdf_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        gemini_api_key: Optional[str] = None,
) -> List[dict]:

    """
    Full pipeline: extract -> clean -> chunk.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Words per chunk.
        chunk_overlap: Words to overlap between consecutive chunks.
        gemini_api_key: Optional Gemini API key for Vision OCR fallback.

    Returns:
        List of chunk dicts with keys: chunk_id, page, text.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages = extract_text_by_page(
        pdf_path,
        gemini_api_key=gemini_api_key
    )

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned/image based."
        )

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return chunks

"""
----------------------------
Internal Helpers
---------------------------

"""
def _clean_text(text: str) -> str:
    """Normalise whitespace and rejoin hyphenated line-breaks."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text.strip()
 
 
def _tesseract_ocr(image: Image.Image) -> tuple[str, float]:
    """
    Run tesseract on a PIL image.
    Returns (text, avg_confidence). Confidence is 0-100.
    """
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        text = _clean_text(" ".join(
            word for word, conf in zip(data["text"], data["conf"])
            if str(conf).lstrip("-").isdigit() and int(conf) > 0 and word.strip()
        ))
        return text, avg_conf
    except Exception as e:
        print(f"Tesseract error: {e}")
        return "", 0
 
 
def _gemini_vision_ocr(image: Image.Image, api_key: str) -> str:
    """
    Send a page image to Gemini Vision and ask it to transcribe the text.
    Works well for handwritten content like prescriptions.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
 
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG", quality=95)
            tmp_path = tmp.name
 
        with open(tmp_path, "rb") as f:
            image_bytes = f.read()
 
        response = model.generate_content([
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            },
            (
                "Transcribe all the text in this image exactly as written. "
                "If it is a medical prescription, extract: patient name, doctor name, "
                "date, medications (name, dose, frequency), and any instructions. "
                "Return plain text only, no markdown."
            ),
        ])
        return _clean_text(response.text)
 
    except Exception as e:
        print(f"Gemini Vision error: {e}")
        return ""