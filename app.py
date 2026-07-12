"""
DocTalk - Streamlit App
upload a PDF -> ask questions -> get answers grounded in the document.
Powered by FAISS + sentence-transformer + Google Gemini. 
"""

import os
import tempfile

import google.generativeai as genai
import streamlit as st

from embeddings import EmbeddingPipeline
from pdf_parser import parse_pdf

# ------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="DocTalk",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 DocTalk")
st.caption("Upload a PDF and ask question about it.")

# ------------------------------------------------------------------------------
# Session Page
# ------------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------   

with st.sidebar:
    st.header("🗝️ Gemini API Key")
    api_key= st.text_input(
        "put the api key",
        type="password",
        placeholder="AI initiating.....",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.divider()

    st.header("Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    chunk_size = st.slider("chunk size (words)", 100, 1000, 500, step=50)
    chunk_overlap = st.slider("chunk overlap (words)", 0, 200, 50, step=10)
    top_k = st.slider("passages to retrieve", 1, 10, 5)

    if uploaded_file and st.button("Process PDF", type="primary"):
        with st.spinner("Parsing and embedding - this takes a moment.."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                chunks = parse_pdf(
                    tmp_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )    

                pipeline = EmbeddingPipeline()
                pipeline.build(chunks, show_progress=False)

                st.session_state.pipeline = pipeline
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    f"{len(chunks)} chunks indexed from **{uploaded_file.name}**"
                )
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
    if st.session_state.doc_name:
        st.info(f"Active: **{st.session_state.doc_name}**")

# ------------------------------------------------------------------------------
# Gaurds
# ------------------------------------------------------------------------------        

if not api_key:
    st.info("Paste your Gemini API key in the sidebar to get started.")
    st.stop()

if st.session_state.pipeline is None:
    st.info("Upload and process a PDF to start chatting")
    st.stop

# ------------------------------------------------------------------------------
# Gemini client (initialised once per key)
# ------------------------------------------------------------------------------

genai.configure(api_key=api_key)
gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    system_instruction=(
        "You are DocTalk, a helpful assistant that answers question strictly "
        "based on the document passages provided. "
        "If the answer is not in the passages, say so clearly. "
        "Always mention the page number(s) your answer comes from."
    ),
)



# ------------------------------------------------------------------------------
# Chat interface
# ------------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask something about the document..."):
    st.session_state.message.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #1. Retrieve relevant chunks from FAISS
            results = st.session_state.pipeline.search(query, top_k=top_k)

            if not results:
                answer = "I couldn't find relevant passages in the document."
            else:
                # 2. Build context block from retrieved chunks
                context = "\n\n".join(
                    f"[Page {r['Page']}]\n{r['text']}" for r in results
                )                

                #3. Send to Gemini
                prompt = (
                    f"Document passages:\n\n{context}\n\n"
                    f"Question: {query}"
                )
                try:
                    response = gemini.generate_content(prompt)
                    answer = response.text
                except Exception as e:
                    answer = f"Gemini error: {e}"


        st.markdown(answer)

        if results:
            with st.expander("Source passages used"):
                for r in results:
                    st.markdown(
                        f"**Page {r['page']}** . relevance score '{r['score']:.3f}'"
                    )                
                    st.caption(r["text"])
                    st.divider()

    st.session_state.message.append({"role": "assistant", "content":answer})
