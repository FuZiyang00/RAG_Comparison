
import os
import streamlit as st
from opensearchpy import OpenSearch, helpers
import time

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE
from src.embeddings import SentenceEmbeddings
from src.ingestion import Document_Ingestion
from src.opensearch import OpenSearchRetriever
from src.ocr import OCR
from src.utils import TextProcessor, setup_logging
import logging

setup_logging()  
logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file) -> str:  # type: ignore
    """
    Saves an uploaded file to the local file system.

    Args:
        uploaded_file: The uploaded file to save.

    Returns:
        str: The file path where the uploaded file is saved.
    """
    UPLOAD_DIR = "uploaded_files"
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File '{uploaded_file.name}' saved to '{file_path}'.")
    return file_path


def Existing_docs(client: OpenSearchRetriever, index_name: str) -> list: 
    """
    Constructs an OpenSearch query to retrieve unique document names.
    Extracts document names from the response and logs the result.
    """

    # Ensure the index exists
    indexer = Document_Ingestion(client)
    indexer.create_index()

    # Query OpenSearch to get the list of unique document names
    query = {
        "size": 0,
        "aggs": {"unique_docs": {"terms": {"field": "document_name", "size": 10000}}},
    }
    response = client.search(index=index_name, body=query)
    buckets = response["aggregations"]["unique_docs"]["buckets"]
    document_names = [bucket["key"] for bucket in buckets]
    if not document_names:
        logger.info("No documents found in OpenSearch.")
        
    logger.info("Retrieved document names from OpenSearch.")

    return document_names

def Upload_docs(uploaded_files:list, 
                document_names: list,
                client: OpenSearch, 
                embedding_model) -> None:
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name in document_names:
            st.warning(f"The file '{uploaded_file.name}' already exists in the index.")
            continue
        
        file_path = save_uploaded_file(uploaded_file)
        reader = OCR(file_path)
        text = reader.extract_text_from_pdf()
        
        # cleaning and chunking 
        cleaner = TextProcessor(text)
        clean_text = cleaner.clean_text()
        chunks = TextProcessor.chunk_text(clean_text)

        # Embeddings 
        embedder = SentenceEmbeddings(chunks)
        embeddings = embedder.generate_embeddings(embedding_model)

        documents_to_index = [
                    {
                        "doc_id": f"{uploaded_file.name}_{i}",
                        "text": chunk,
                        "embedding": embedding,
                        "document_name": uploaded_file.name,
                    }
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                        ]
    
        indexer = Document_Ingestion(client)
        indexer.bulk_index_documents(documents_to_index)
        
        st.session_state["documents"].append(
            {
                "filename": uploaded_file.name,
                "content": text,
                "file_path": file_path,
            }
        )

    
def documents_displayer(client: OpenSearch) -> None:

    indexer = Document_Ingestion(client)

    for idx, doc in enumerate(st.session_state["documents"], 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(
                f"{idx}. {doc['filename']} - {len(doc['content'])} characters extracted"
            )
        with col2:
            delete_button = st.button(
                "Delete",
                key=f"delete_{doc['filename']}_{idx}",
                help=f"Delete {doc['filename']}",
            )
            if delete_button:
                if doc["file_path"] and os.path.exists(doc["file_path"]):
                    try:
                        os.remove(doc["file_path"])
                        logger.info(
                            f"Deleted file '{doc['filename']}' from filesystem."
                        )
                    except FileNotFoundError:
                        st.error(
                            f"File '{doc['filename']}' not found in filesystem."
                        )
                        logger.error(
                            f"File '{doc['filename']}' not found during deletion."
                        )

                response = indexer.delete_documents_by_document_name(doc["filename"])
                st.session_state["documents"].pop(idx - 1)
                st.session_state["deleted_file"] = doc["filename"]
                time.sleep(0.5)
                st.rerun()

                st.write(f"Deleted document: {response}")