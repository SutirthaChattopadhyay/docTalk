
"""
DocTalk - Streamlit App

Upload a PDF -> Ask questions -> Get answers grounded in the document.

Powered by:
- pdfplumber
- Tesseract OCR
- Gemini Vision OCR (optional fallback)
- Sentence Transformers
- FAISS
- Google Gemini
"""

import os
import tempfile

import streamlit as st
from google import genai

from embeddings import EmbeddingPipeline
from pdf_parser import parse_pdf


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocTalk",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 DocTalk")
st.caption("Upload a PDF and ask questions about its contents.")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🗝️ Gemini API Key")

    api_key = st.text_input(
        "Enter your Gemini API key",
        type="password",
        placeholder="Enter API key...",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.divider()

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
    )

    st.divider()

    st.header("⚙️ Retrieval Settings")

    chunk_size = st.slider(
        "Chunk size (words)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk overlap (words)",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
    )

    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=10,
        value=5,
    )

    use_gemini_vision = st.checkbox(
        "Enable Gemini Vision OCR fallback",
        value=False,
        help="Uses Gemini when local OCR does not extract sufficient text.",
    )

    st.divider()

    process_button = st.button(
        "🚀 Process PDF",
        type="primary",
        use_container_width=True,
    )


    # --------------------------------------------------
    # PDF PROCESSING
    # --------------------------------------------------

    if process_button:

        if not api_key:
            st.error("Please enter your Gemini API key.")

        elif not uploaded_file:
            st.error("Please upload a PDF.")

        else:

            tmp_path = None

            try:

                with st.spinner(
                    "Parsing document and generating embeddings..."
                ):

                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as tmp:

                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # Extract and chunk document
                    chunks = parse_pdf(
                        tmp_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        gemini_api_key=(
                            api_key if use_gemini_vision else None
                        ),
                    )

                    if not chunks:
                        raise ValueError(
                            "No readable content was extracted from the PDF."
                        )

                    # Generate embeddings and build FAISS index
                    pipeline = EmbeddingPipeline()

                    pipeline.build(
                        chunks,
                        show_progress=False,
                    )

                    # Update session state only after successful processing
                    st.session_state.pipeline = pipeline
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.api_key = api_key
                    st.session_state.messages = []

                st.success(
                    f"Successfully indexed {len(chunks)} chunks "
                    f"from {uploaded_file.name}."
                )

                st.write("**Total chunks:**", len(chunks))
                st.write(
                    "**Total pages:**",
                    len(set(chunk["page"] for chunk in chunks)),
                )

            except Exception as e:

                st.error(f"Document processing failed: {e}")

            finally:

                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


    # Active document information
    if st.session_state.doc_name:

        st.divider()

        st.success(
            f"Active document: {st.session_state.doc_name}"
        )


# --------------------------------------------------
# APPLICATION GUARDS
# --------------------------------------------------

if not api_key:

    st.info(
        "Enter your Gemini API key in the sidebar to get started."
    )

    st.stop()


if st.session_state.pipeline is None:

    st.info(
        "Upload and process a PDF to start chatting with your document."
    )

    st.stop()


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

try:

    gemini = genai.Client(api_key=api_key)

except Exception as e:

    st.error(f"Could not initialize Gemini: {e}")
    st.stop()


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

st.subheader("💬 Chat with your document")


# Display conversation history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        # Display sources for assistant responses
        if msg["role"] == "assistant" and msg.get("sources"):

            with st.expander("📚 View source passages"):

                for source in msg["sources"]:

                    st.markdown(
                        f"**Page {source['page']}** "
                        f"| Similarity: {source['score']:.3f}"
                    )

                    st.caption(
                        f"Extraction method: "
                        f"{source.get('method', 'unknown')}"
                    )

                    st.write(source["text"])

                    st.divider()


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

if query := st.chat_input(
    "Ask something about your document..."
):

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # --------------------------------------------------
    # RETRIEVAL + GENERATION
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching document and generating answer..."):

            try:

                # 1. Retrieve relevant chunks from FAISS
                results = st.session_state.pipeline.search(
                    query,
                    top_k=top_k,
                )

                if not results:

                    answer = (
                        "I couldn't find relevant passages "
                        "in the document."
                    )

                else:

                    # 2. Build context from retrieved passages
                    context = "\n\n".join(
                        f"[Page {result['page']}]\n"
                        f"{result['text']}"
                        for result in results
                    )

                    # 3. Construct grounded prompt
                    prompt = f"""
You are DocTalk, a document question-answering assistant.

Answer the user's question using ONLY the supplied document passages.

Rules:
1. Never invent or assume information.
2. If the answer is missing, explicitly say it was not found.
3. Preserve medical values, units, names and medication dosages exactly.
4. OCR text may contain errors. Do not silently correct uncertain values.
5. Do not provide medical diagnoses.
6. Keep answers concise and clear.
7. Mention the relevant page number when possible.

DOCUMENT PASSAGES:

{context}

USER QUESTION:

{query}

ANSWER:
"""

                    # 4. Generate answer using Gemini
                    response = gemini.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                    )

                    answer = (
                        response.text
                        if response.text
                        else "Gemini returned an empty response."
                    )


                # Display answer
                st.markdown(answer)


                # Display retrieved sources
                if results:

                    with st.expander("📚 Source passages used"):

                        for result in results:

                            st.markdown(
                                f"**Page {result['page']}** "
                                f"| Similarity: {result['score']:.3f}"
                            )

                            st.caption(
                                f"Extraction method: "
                                f"{result.get('method', 'unknown')}"
                            )

                            st.write(result["text"])

                            st.divider()


                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                    }
                )


            except Exception as e:

                st.error(
                    "Something went wrong while generating the answer. "
                    "Please try again."
                )

                st.caption(str(e))
