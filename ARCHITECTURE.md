

\# DocTalk — Technical Architecture



\## 1. Overview



DocTalk is a Retrieval-Augmented Generation (RAG) application that answers questions about uploaded PDF documents.



Instead of sending an entire document to an LLM, DocTalk extracts text, divides it into chunks, converts those chunks into embeddings, retrieves relevant passages, and supplies those passages as context to Gemini.



\## 2. High-Level Architecture



```text

&#x20;                 USER

&#x20;                   |

&#x20;                   v

&#x20;             STREAMLIT UI

&#x20;                   |

&#x20;                   v

&#x20;              PDF UPLOAD

&#x20;                   |

&#x20;                   v

&#x20;             PDF PARSER

&#x20;            /           \\

&#x20;           v             v

&#x20;      pdfplumber       Tesseract

&#x20;           \\             /

&#x20;            v           v

&#x20;             TEXT EXTRACTION

&#x20;                   |

&#x20;                   v

&#x20;              CHUNKING

&#x20;                   |

&#x20;                   v

&#x20;         SENTENCE TRANSFORMERS

&#x20;                   |

&#x20;                   v

&#x20;             EMBEDDINGS

&#x20;                   |

&#x20;                   v

&#x20;              FAISS INDEX

&#x20;                   |

&#x20;                   v

&#x20;            USER QUESTION

&#x20;                   |

&#x20;                   v

&#x20;           QUERY EMBEDDING

&#x20;                   |

&#x20;                   v

&#x20;            SIMILARITY SEARCH

&#x20;                   |

&#x20;                   v

&#x20;           RETRIEVED CONTEXT

&#x20;                   |

&#x20;                   v

&#x20;             GEMINI API

&#x20;                   |

&#x20;                   v

&#x20;             FINAL ANSWER

```



\## 3. Core Components



\### 3.1 Streamlit — app.py



Responsibilities:

\- Accept PDF uploads.

\- Collect the Gemini API key.

\- Configure chunking and retrieval parameters.

\- Display conversation history.

\- Display generated answers and source passages.



The current Streamlit implementation handles retrieval and generation directly.



\### 3.2 PDF Parser — pdf\_parser.py



Responsibilities:

\- Extract text from PDF pages.

\- Use pdfplumber for digital PDFs.

\- Use Tesseract for scanned pages.

\- Optionally use Gemini Vision OCR.

\- Divide extracted text into chunks.

\- Attach page and extraction-method metadata.



Important limitation:



OCR output is not guaranteed to be accurate. Medical values and medication names must be checked against the original document.



\### 3.3 Embedding Pipeline — embeddings.py



Model: all-MiniLM-L6-v2.



Responsibilities:

\- Convert text chunks into numerical vectors.

\- Normalize embeddings.

\- Create a FAISS index.

\- Retrieve the most relevant chunks for a query.

\- Maintain metadata associated with each vector.



The model produces 384-dimensional embeddings.



\### 3.4 Vector Search — FAISS



DocTalk uses FAISS IndexFlatIP.



IndexFlatIP performs exact inner-product similarity search.



Because the document embeddings and query embeddings are normalized, inner product is equivalent to cosine similarity:



cosine\_similarity(A, B) = A · B



The index returns the top-k most similar vectors.



\### 3.5 Answer Generation — generator.py



Responsibilities:

\- Construct a prompt containing retrieved context.

\- Send the prompt to Gemini.

\- Require answers to remain grounded in the supplied context.

\- Handle API errors and empty responses.



The generator is also used by the separate RAG orchestration module.



\### 3.6 RAG Orchestration — rag\_pipeline.py



The DocTalkRAG class coordinates:



1\. PDF ingestion.

2\. Text extraction and chunking.

3\. Embedding generation.

4\. FAISS indexing.

5\. Query retrieval.

6\. Context construction.

7\. Gemini answer generation.



This is a separate reusable pipeline; the current Streamlit app does not yet delegate its workflow to this class.



\## 4. Data Flow



\### Document ingestion



1\. User uploads a PDF.

2\. The parser extracts page text.

3\. OCR is used when required.

4\. Extracted text is split into chunks.

5\. Each chunk retains its page number and extraction method.

6\. The embedding model converts each chunk into a vector.

7\. FAISS stores the vectors for similarity search.



\### Question answering



1\. User enters a question.

2\. The question is converted into an embedding.

3\. FAISS searches for the top-k similar chunks.

4\. Retrieved chunks are combined into context.

5\. Gemini receives the question and context.

6\. The application displays the answer and its source passages.



\## 5. Important Design Decisions



\### Why chunking?



Sending an entire document can be inefficient and may exceed model context limits.



Chunking makes retrieval more targeted and allows the system to supply only relevant passages.



\### Why embeddings?



Keyword search can miss semantically similar wording.



Embeddings represent text as numerical vectors, allowing similarity search based on meaning.



\### Why FAISS?



FAISS provides efficient vector similarity search and can run locally without a hosted vector database.



\### Why normalized embeddings?



Normalization allows inner-product search to behave like cosine similarity.



\### Why OCR?



Many PDFs contain scanned images rather than selectable text. OCR makes those pages searchable.



\## 6. Failure Modes



| Component | Possible failure |

|---|---|

| PDF parser | Corrupted or unsupported PDF |

| OCR | Incorrect text recognition |

| Chunking | Important context split across chunks |

| Embeddings | Semantically weak representations |

| FAISS | Irrelevant passages retrieved |

| Gemini | Rate limits, service errors, hallucinations |

| Streamlit | Invalid session state or upload errors |



\## 7. Evaluation Plan



Retrieval should be evaluated independently from answer generation.



Possible metrics:



\- Recall@K: whether relevant passages appear in the top K results.

\- Precision@K: how many retrieved passages are relevant.

\- Mean Reciprocal Rank: how highly the first relevant result appears.

\- Answer faithfulness: whether the generated answer is supported by retrieved context.



OCR should also be evaluated against manually verified text.



\## 8. Security Considerations



\- Never commit API keys.

\- Do not commit private PDFs or patient records.

\- Use environment variables or a secrets manager for deployment.

\- Validate uploaded file types and sizes.

\- Avoid logging sensitive document contents.

\- Apply authentication and access controls before public deployment.



\## 9. Future Architecture Improvements



\- Integrate DocTalkRAG into the Streamlit application.

\- Add automated tests.

\- Add retrieval evaluation datasets.

\- Improve mixed digital/scanned page detection.

\- Add Docker support.

\- Deploy using AWS.

\- Add secure document storage and authentication.

