import logging
import os
import streamlit as st

from src.constants import (
    HYBRID_SEARCH,
    K_RESULTS,
    TEMPERATURE) 

from src.utils import setup_logging
from src.llm_engine import LLMEngine
from src.ingestion import Document_Ingestion
from src.opensearch import OpenSearchRetriever
from src.embeddings import SentenceEmbeddings
from src.llm_engine import LLMEngine

from app.css import chatbot_style
from app.functions import (inference_sidebar_slider,
                           llm_chat)

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)


# Set page configuration
st.set_page_config(page_title="Jam with AI - Chatbot", page_icon="🤖")

# Apply custom CSS
st.markdown(chatbot_style, unsafe_allow_html=True)
logger.info("Custom CSS applied.")

def render_chatbot_page() -> None:
    # Set up a placeholder at the very top of the main content area
    st.title("Jam with AI - Chatbot 🤖")

    # Set up sidebar slider 
    inference_sidebar_slider(hybrid_search=HYBRID_SEARCH, num_results=K_RESULTS, temperature=TEMPERATURE)


    # Initialize OpenSearch client
    with st.spinner("Connecting to OpenSearch..."):
        open_search_client = OpenSearchRetriever.get_opensearch_client()
    
    indexer = Document_Ingestion(open_search_client)
    indexer.create_index()

    # Loading embeddinds and Large Language Model
    embedding_model = None
    if "embedding_models_loaded" not in st.session_state:
        with st.spinner("Loading Embedding and Ollama models for Hybrid Search..."):
            embedding_model = SentenceEmbeddings.get_embedding_model()  
            LLMEngine.ensure_model_pulled()
            st.session_state["embedding_models_loaded"] = True
    else: 
        embedding_model = SentenceEmbeddings.get_embedding_model()
        LLMEngine.ensure_model_pulled()

    logger.info("Embedding model loaded.")

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    else: 
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Process user input and generate response

    model_settings = [st.session_state["use_hybrid_search"],
                      st.session_state["num_results"],
                      st.session_state["temperature"]]
    

    if prompt := st.chat_input("Type your message here..."):
        llm_chat(prompt, model_settings, embedding_model, open_search_client)



# Main execution
if __name__ == "__main__":
    render_chatbot_page()