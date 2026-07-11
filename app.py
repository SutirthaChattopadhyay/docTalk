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
    upload_file = st.file_uploader("Choose a PDF", type="pdf")

    chunk_size = st.slider("chunk size (words)", 100, 1000, 500, step=50)
    chunk_overlap = st.slider("chunk overlap ()")