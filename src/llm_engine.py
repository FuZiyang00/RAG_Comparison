import logging
from typing import Dict, Iterable, List, Optional

from .constants import ASSYMETRIC_EMBEDDING
from .embeddings import SentenceEmbeddings
from opensearchpy import OpenSearch
from .opensearch import OpenSearchRetriever
from .utils import setup_logging
from openai import OpenAI

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        
    def run_chat_streaming(self, prompt: str, temp: float): 
        """
        Args:
            prompt (str): The prompt to send to the model.
            temperature (float): The response generation temperature.

        """
    
        try:
            response_stream = self.client.chat.completions.create(
                model= self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature= temp,
                stream = True
            )
        
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            return None
        
        return response_stream
    
    @staticmethod
    def prompt_template(query: str, context: str, history: List[Dict[str, str]]) -> str:
        """
        Builds the prompt with context, conversation history, and user query.

        Args:
            query (str): The user's query.
            context (str): Context text gathered from hybrid search.
            history (List[Dict[str, str]]): Conversation history to include in the prompt.

        Returns:
            str: Constructed prompt for Ollama model.
        """
        prompt = "You are a knowledgeable chatbot assistant. "
        if context:
            prompt += (
                "Use the following context to answer the question.\nContext:\n"
                + context
                + "\n"
            )
        else:
            prompt += "Answer questions to the best of your knowledge.\n"

        if history:
            prompt += "Conversation History:\n"
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                prompt += f"{role}: {content}\n"
            prompt += "\n"

        prompt += f"User: {query}\nAssistant:"
        logger.info("Prompt constructed with context and conversation history.")
        return prompt
    
    @staticmethod
    def generate_response_streaming(
        query: str,
        chat_setting: list,
        embedding_model: SentenceEmbeddings,
        op_client: OpenSearch,
        chat_history: Optional[List[Dict[str, str]]] = None
        ) -> Optional[Iterable[str]]:
        """
        Generates a chatbot response by performing hybrid search and incorporating conversation history.

        Args:
            query (str): The user's query.
            chat_setting (list): List of chat settings.
            chat_history (Optional[List[Dict[str, str]]]): List of chat history messages.

        Returns:
            Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.
        """
        chat_history = chat_history or []
        max_history_messages = 10
        history = chat_history[-max_history_messages:]
        context = ""

        # Include hybrid search results if enabled
        use_hybrid_search = chat_setting[0]
        if use_hybrid_search:
            logger.info("Performing hybrid search.")
            if ASSYMETRIC_EMBEDDING:
                prefixed_query = f"passage: {query}"
            else:
                prefixed_query = f"{query}"

            embedding_model = embedding_model
            query_embedding = embedding_model.encode(prefixed_query).tolist()  # Convert tensor to list of floats
            
            # Perform hybrid search
            client = op_client
            num_results = chat_setting[1]
            open_search_retriever = OpenSearchRetriever(query, query_embedding, client)
            search_results = open_search_retriever.hybrid_search(top_k=num_results)
            logger.info("Hybrid search completed.")

            # Collect text from search results
            for i, result in enumerate(search_results):
                context += f"Document {i}:\n{result['_source']['text']}\n\n"

        prompt = LLMEngine.prompt_template(query, context, history)

        return prompt