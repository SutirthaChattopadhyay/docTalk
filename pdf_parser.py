"""
DocTalk - PDF parser [extracts and chunks text from uploaded PDF files]
"""

import re
from pathlib import Path
from typing import List

import pdfplumber

def extract_text_by_page(pdf_path: str) -> List[dict]:
    """
    extract text frpm each page of a pdf, returning the list of page dicts.

    Returns:
        [{"page": int, "text": str},...]
    """

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = _clean_text(text)
            if text.strip():
                pages.append({"page": i + 1, "text": text})
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
) -> List[dict]:

    """
    Full pipeline: extract->clean->chunk.

    Args:
        pdf_path: Path to the pdf file.
        chunk_size: words per chunk.
        chunk_overlap: overlap between consecutive chunks

    Returns :
        list of chunk dicts with keys: chunk_id, page, text.

    """ 

    path = Path(pdf_path)
    if not path.exists ():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
    
    pages = extract_text_by_page(pdf_path)
    if not pages:
        raise ValueError("No extractable text found. The PDF maybe scanned/image based.")
    
    chunks = chunk_pages(pages, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return chunks

"""
----------------------------
Internal Helpers
---------------------------

"""
def _clean_text(text: str) -> str:
    """Normalise whitespace whitespace and stip common PDF extraction artefacts."""
    text = re.sub(r"\s+"," ", text)    #collapse runs of whitespace
    text = re.sub(r"(\w)-\s+(\w)",r"\1\2", text)  #rejoin hyphenated line breaks
    return text.strip()