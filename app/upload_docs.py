import logging
import os
import time

import streamlit as st
from PyPDF2 import PdfReader

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE
from css import uploading_style

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE
from src.embeddings import SentenceEmbeddings
from src.ingestion import Document_Ingestion
from src.opensearch import OpenSearchRetriever
from src.utils import TextProcessor, setup_logging

from functions import (
    Existing_docs,
    Upload_docs,
    documents_displayer,
)

setup_logging()  
logger = logging.getLogger(__name__)


st.set_page_config(page_title="Upload Documents", page_icon="📂")
st.markdown(uploading_style, unsafe_allow_html=True)
st.sidebar.markdown(
    "<h2 style='text-align: center;'>Retrieval Augmented Generation</h2>", 
    unsafe_allow_html=True
    )

st.sidebar.markdown(
    "<h4 style='text-align: center;'>Your Document Assistant</h4>",
    unsafe_allow_html=True
    )

def render_upload_page() -> None:
    """
    Renders the document upload page for users to upload and manage PDFs.
    Shows only the documents that are present in the OpenSearch index.
    """

    st.title("Upload Documents")

    if "embedding_models_loaded" not in st.session_state:
        with st.spinner("Loading models for document processing..."):
            embedding_model = SentenceEmbeddings.get_embedding_model()
            st.session_state["embedding_models_loaded"] = True
        logger.info("Embedding models loaded.")
    
    # Initialize OpenSearch client
    with st.spinner("Connecting to OpenSearch..."):
        open_search_client = OpenSearchRetriever.get_opensearch_client()
    index_name = OPENSEARCH_INDEX

    # Get the list of existing documents from OpenSearch
    document_names = Existing_docs(open_search_client, index_name)

    UPLOAD_DIR = "uploaded_files"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if document_names:
        for document_name in document_names:
            file_path = os.path.join(UPLOAD_DIR, document_name)
            if os.path.exists(file_path):
                reader = PdfReader(file_path)
                text = "".join([page.extract_text() for page in reader.pages])
                st.session_state["documents"].append(
                    {"filename": document_name, "content": text, "file_path": file_path}
                )
            else:
                st.session_state["documents"].append(
                    {"filename": document_name, "content": "", "file_path": None}
                )
                logger.warning(f"File '{document_name}' does not exist locally.")


    # Uploading 
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Uploading and processing documents. Please wait..."):
            Upload_docs(uploaded_files, document_names, 
                        open_search_client, embedding_model)
        st.success("Files uploaded and indexed successfully!")

    # Displaying documents
    if st.session_state["documents"]:
        st.markdown("### Uploaded Documents")
        with st.expander("Manage Uploaded Documents", expanded=True):
            documents_displayer



if __name__ == "__main__":
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
    render_upload_page()
   

    



