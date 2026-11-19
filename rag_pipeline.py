
from pathlib import Path

from pdf_parser import parse_pdf
from embeddings import EmbeddingPipeline
from generator import AnswerGenerator


class DocTalkRAG:
    """Coordinates document ingestion, retrieval and answer generation."""

    def __init__(self, api_key: str, top_k: int = 3):
        self.api_key = api_key
        self.top_k = top_k

        self.embedding_pipeline = EmbeddingPipeline()
        self.answer_generator = AnswerGenerator(api_key)

        self.document_chunks = []
        self.document_path = None
        self.ready = False

    def load_document(
        self,
        pdf_path: str,
        use_gemini_vision_ocr: bool = False
    ) -> dict:

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print("Processing document...")

        # 1. Extract and chunk the PDF
        chunks = parse_pdf(
            str(path),
            gemini_api_key=(
                self.api_key if use_gemini_vision_ocr else None
            )
        )

        if not chunks:
            raise ValueError("No text chunks were extracted.")

        # 2. Build the vector index
        self.embedding_pipeline.build(chunks)

        self.document_chunks = chunks
        self.document_path = str(path)
        self.ready = True

        return {
            "filename": path.name,
            "total_chunks": len(chunks),
            "total_pages": len(set(c["page"] for c in chunks))
        }

    def ask(self, question: str) -> dict:

        if not self.ready:
            raise RuntimeError("Load a document before asking questions.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # 3. Retrieve relevant chunks
        results = self.embedding_pipeline.search(
            question,
            top_k=self.top_k
        )

        if not results:
            return {
                "answer": "No relevant document content was found.",
                "sources": []
            }

        # 4. Prepare context for Gemini
        context = "\n\n".join(
            f"Page {result['page']} "
            f"(extraction: {result.get('method', 'unknown')}):\n"
            f"{result['text']}"
            for result in results
        )

        # 5. Generate an answer
        answer = self.answer_generator.generate(
            question,
            context
        )

        return {
            "answer": answer,
            "sources": results
        }
