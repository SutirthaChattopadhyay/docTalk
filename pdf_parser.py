
"""
DocTalk - PDF Parser

Extracts and chunks text from PDFs.

Fallback chain:
pdfplumber -> Tesseract OCR -> Gemini Vision OCR
"""

import io
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from PIL import Image
from google import genai
from google.genai import types


# --------------------------------------------------
# TESSERACT CONFIGURATION (CROSS-PLATFORM)
# --------------------------------------------------

# Priority:
# 1. TESSERACT_CMD environment variable
# 2. Tesseract executable available on PATH
# 3. Default Windows installation path

tesseract_cmd = (
    os.getenv("TESSERACT_CMD")
    or shutil.which("tesseract")
)

if not tesseract_cmd and os.name == "nt":
    windows_tesseract = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if Path(windows_tesseract).is_file():
        tesseract_cmd = windows_tesseract

if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# If Tesseract confidence is below this threshold,
# Gemini Vision is attempted when an API key is available.

TESSERACT_CONFIDENCE_THRESHOLD = 50


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------

def extract_text_by_page(
    pdf_path: str,
    gemini_api_key: Optional[str] = None
) -> List[dict]:

    pages = []
    pil_images = None

    with pdfplumber.open(pdf_path) as pdf:

        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # Tier 1: Digital PDF extraction
            text = _clean_text(page.extract_text() or "")

            if len(text.strip()) >= 100:
                pages.append({
                    "page": page_num,
                    "text": text,
                    "method": "pdfplumber"
                })
                continue

            # Tier 2: Local Tesseract OCR
            if pil_images is None:
                pil_images = convert_from_path(
                    pdf_path,
                    dpi=300
                )

            pil_image = pil_images[i]

            tess_text, confidence = _tesseract_ocr(
                pil_image
            )

            if (
                tess_text.strip()
                and confidence >= TESSERACT_CONFIDENCE_THRESHOLD
            ):
                pages.append({
                    "page": page_num,
                    "text": tess_text,
                    "method": "tesseract",
                    "confidence": confidence
                })
                continue

            # Tier 3: Gemini Vision OCR fallback
            if gemini_api_key:

                print(
                    f"Page {page_num}: OCR confidence "
                    f"{confidence:.0f}% - trying Gemini Vision..."
                )

                gemini_text = _gemini_vision_ocr(
                    pil_image,
                    gemini_api_key
                )

                if gemini_text.strip():
                    pages.append({
                        "page": page_num,
                        "text": gemini_text,
                        "method": "gemini_vision"
                    })
                    continue

            # Retain digital text if OCR failed
            if text.strip():
                pages.append({
                    "page": page_num,
                    "text": text,
                    "method": "pdfplumber"
                })
            else:
                print(
                    f"Page {page_num}: No text extracted, skipping."
                )

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def chunk_pages(
    pages: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[dict]:

    """
    Split page text into overlapping word chunks.

    Each chunk retains its page number and extraction method.
    Whitespace and line breaks are normalized.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    chunk_id = 0

    for page_dict in pages:

        text = page_dict["text"].strip()

        if not text:
            continue

        words = text.split()
        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_dict["page"],
                "text": chunk_text,
                "method": page_dict.get("method", "unknown"),
            })

            chunk_id += 1
            start += chunk_size - chunk_overlap

    return chunks


# --------------------------------------------------
# COMPLETE PDF PIPELINE
# --------------------------------------------------

def parse_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    gemini_api_key: Optional[str] = None,
) -> List[dict]:

    """
    Full pipeline: validate -> extract -> chunk.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.
        gemini_api_key: Optional Gemini API key for Vision OCR.

    Returns:
        List of dictionaries containing chunk_id, page,
        text and extraction method.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got: {path.suffix}"
        )

    pages = extract_text_by_page(
        str(path),
        gemini_api_key=gemini_api_key
    )

    if not pages:
        raise ValueError(
            "No extractable text found. "
            "The PDF may be scanned or image-based."
        )

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return chunks


# --------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------

def _clean_text(text: str) -> str:

    """Normalize whitespace and rejoin hyphenated words."""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    return text.strip()


def _tesseract_ocr(
    image: Image.Image
) -> tuple[str, float]:

    """
    Run Tesseract OCR using PSM 3.

    Returns:
        Extracted text and average OCR confidence (0-100).
    """

    try:

        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 3"
        )

        words_by_line = {}
        confidences = []

        for word, conf, block, par, line in zip(
            data["text"],
            data["conf"],
            data["block_num"],
            data["par_num"],
            data["line_num"]
        ):

            if (
                str(conf).lstrip("-").isdigit()
                and int(conf) > 0
                and word.strip()
            ):

                key = (block, par, line)

                words_by_line.setdefault(
                    key, []
                ).append(word)

                confidences.append(int(conf))

        avg_conf = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        text = "\n".join(
            " ".join(words)
            for words in words_by_line.values()
        )

        return text, avg_conf

    except Exception as e:

        print(f"Tesseract error: {e}")

        return "", 0


def _gemini_vision_ocr(
    image: Image.Image,
    api_key: str
) -> str:

    """
    Use Gemini Vision to transcribe scanned documents.

    Unreadable content should be marked as [unclear].
    """

    try:

        client = genai.Client(api_key=api_key)

        image_buffer = io.BytesIO()

        image.save(
            image_buffer,
            format="JPEG",
            quality=95
        )

        image_bytes = image_buffer.getvalue()

        prompt = """
        Transcribe all readable text in this image accurately.

        Preserve:
        - Original wording
        - Names and dates
        - Numbers and measurements
        - Medication names and dosages
        - Instructions and abbreviations

        If something is unreadable, mark it as [unclear].
        Do not guess missing information.

        Return plain text only.
        """

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                prompt
            ]
        )

        return _clean_text(
            response.text or ""
        )

    except Exception as e:

        print(f"Gemini Vision error: {e}")

        return ""
