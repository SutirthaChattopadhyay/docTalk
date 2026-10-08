"""
DocTalk - embedding & FAISS Index
Encodes text chunks and stores/loads them in a FAISS vector index.
"""

import json
import os
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer


# ------------------------------
# Defaults
# ------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "doctalk.index"
META_FILE = "doctalk_meta.json"


# ------------------------------
# Public API
# ------------------------------

class EmbeddingPipeline:
    """Encode chunks -> FAISS index, with save/load support."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: List[dict] = []

    # --------------------------------
    # Building the Index
    # --------------------------------

    def build(
        self,
        chunks: List[dict],
        show_progress: bool = True
    ) -> None:
        """
        Encode chunks and populate the FAISS index.

        Args:
            chunks: Output of pdf_parser.parse_pdf()
            show_progress: Print a progress message while encoding.
        """

        if not chunks:
            raise ValueError("chunks list is empty - nothing to embed.")

        texts = [c["text"] for c in chunks]

        if show_progress:
            print(
                f"Encoding {len(texts)} chunks with "
                f"'{self.model.get_sentence_embedding_dimension()}-dim' embeddings.."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]

        # Normalized embeddings + inner product = cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        self.metadata = [
            {
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "text": c["text"],
            }
            for c in chunks
        ]

        print(
            f"Index built: {self.index.ntotal} vectors, dim={dim}"
        )

    # --------------------------------------------------
    # Querying
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns:
            List of dictionaries containing:
            chunk_id, page, text, score
        """

        if self.index is None:
            raise RuntimeError(
                "Index is empty. Call build() or load() first."
            )

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_vec,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = dict(self.metadata[idx])
            result["score"] = float(score)

            results.append(result)

        return results

    # ----------------------------------------------
    # Persistence
    # ----------------------------------------------

    def save(self, directory: str = ".") -> None:
        """Persist the FAISS index and metadata to disk."""

        if self.index is None:
            raise RuntimeError(
                "Nothing to save - index is empty."
            )

        Path(directory).mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            os.path.join(directory, INDEX_FILE)
        )

        with open(
            os.path.join(directory, META_FILE),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Saved index + metadata to '{directory}/'"
        )

    def load(self, directory: str = ".") -> None:
        """Load a previously saved FAISS index and metadata."""

        index_path = os.path.join(
            directory,
            INDEX_FILE
        )

        meta_path = os.path.join(
            directory,
            META_FILE
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index file not found: {index_path}"
            )

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}"
            )

        self.index = faiss.read_index(index_path)

        with open(
            meta_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.metadata = json.load(f)

        print(
            f"Loaded index: {self.index.ntotal} "
            f"vectors from '{directory}/'"
        )