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
    st.caption("Upload a PDF and ask question about it.")

if "messages" not in st.session_state:
    st.session_state.messages: list = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name: str = ""

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
# Chat Interface
# ------------------------------------------------------------------------------

for msg in st.session_state.messages:
    with



