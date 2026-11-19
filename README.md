

\# DocTalk — Ask Questions About Your Documents



DocTalk is a Python-based document question-answering application. It allows users to upload PDF documents, extract their text, retrieve relevant passages, and ask questions using a Retrieval-Augmented Generation (RAG) pipeline.



\## Features



\- Upload and process PDF documents.

\- Extract text from digital PDFs using pdfplumber.

\- Use Tesseract OCR for scanned document pages.

\- Convert text chunks into semantic embeddings.

\- Retrieve relevant passages using FAISS.

\- Generate context-grounded answers using Google's Gemini API.

\- Display retrieved source passages alongside answers.



\## Architecture



```text

PDF Upload

&#x20;   |

&#x20;   v

PDF Parser

&#x20;   |

&#x20;   +---- pdfplumber (digital PDFs)

&#x20;   |

&#x20;   +---- Tesseract OCR (scanned pages)

&#x20;   |

&#x20;   v

Text Chunking

&#x20;   |

&#x20;   v

Sentence Transformers

&#x20;   |

&#x20;   v

Vector Embeddings

&#x20;   |

&#x20;   v

FAISS Similarity Search

&#x20;   |

&#x20;   v

Relevant Context

&#x20;   |

&#x20;   v

Gemini Answer Generation

&#x20;   |

&#x20;   v

Answer + Source Passages

```



\## Technology Stack



| Component | Technology |

|---|---|

| User interface | Streamlit |

| PDF extraction | pdfplumber |

| OCR | Tesseract |

| PDF image conversion | pdf2image |

| Embeddings | all-MiniLM-L6-v2 |

| Vector search | FAISS |

| Generation | Google Gemini |

| Language | Python |



\## Project Structure



```text

docTalk/

├── app.py

├── pdf\_parser.py

├── embeddings.py

├── generator.py

├── rag\_pipeline.py

├── requirements.txt

├── README.md

├── .gitignore

└── test\_\*.py

```



\## Setup



\### 1. Clone the repository



```bash

git clone https://github.com/SutirthaChattopadhyay/docTalk.git

cd docTalk

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Install system dependencies



Install Tesseract OCR and Poppler for Windows. Ensure their executable paths are configured correctly.



\### 5. Run the application



```bash

streamlit run app.py

```



Enter your Gemini API key in the application when prompted.



\## How It Works



1\. A user uploads a PDF.

2\. The parser extracts text from the document.

3\. Scanned pages can be processed using OCR.

4\. Extracted text is split into smaller chunks.

5\. Sentence Transformers converts chunks into vector embeddings.

6\. FAISS retrieves relevant chunks for a user question.

7\. Gemini generates an answer using the retrieved context.

8\. The application displays the answer and source passages.



\## Limitations



\- OCR accuracy depends on document quality.

\- Poor scans, handwriting, and unusual layouts can produce extraction errors.

\- Retrieval quality depends on chunking and embedding choices.

\- Gemini availability and rate limits can affect answer generation.

\- The application does not independently verify extracted facts.



\## Important Disclaimer



DocTalk is an educational document-question-answering project. It is not a medical diagnostic system. Extracted medical information may contain OCR errors and must be verified against the original document by a qualified professional.



\## Future Improvements



\- Docker-based deployment.

\- AWS deployment.

\- Automated evaluation of retrieval quality.

\- Better OCR confidence handling.

\- Improved document processing and error handling.

\- Authentication and secure document storage.



\## Author



Built as a learning project to explore Python, document processing, embeddings, vector databases, and Retrieval-Augmented Generation.