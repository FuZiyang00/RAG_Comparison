import logging
from typing import Any, List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from .constants import EMBEDDING_MODEL_PATH, EMBEDDING_DIMENSION
from .utils import setup_logging

setup_logging()  
logger = logging.getLogger(__name__)

class SentenceEmbeddings:
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        
    @staticmethod
    @st.cache_resource(show_spinner=False) # Cache the model to avoid loading it multiple times 
    def get_embedding_model() -> SentenceTransformer:
        """
        Loads and caches the embedding model.

        Returns:
            SentenceTransformer: The loaded embedding model.
        """
        logger.info(f"Loading embedding model from path: {EMBEDDING_MODEL_PATH}")
        return SentenceTransformer(EMBEDDING_MODEL_PATH)

    def generate_embeddings(self, embedding_model: SentenceTransformer) -> List[np.ndarray[Any, Any]]:
        """
        Generates embeddings for a list of text chunks.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List[np.ndarray[Any, Any]]: List of embeddings as numpy arrays for each chunk.
        """

        embeddings = [np.array(embedding_model.encode(chunk)) for chunk in self.chunks]
        logger.info(f"Generated embeddings for {len(self.chunks)} text chunks.")
        return embeddings
