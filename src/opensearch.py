import logging
from typing import Any, Dict, List

from opensearchpy import OpenSearch

from .constants import OPENSEARCH_HOST, OPENSEARCH_INDEX, OPENSEARCH_PORT
from .utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

class OpenSearchRetriever:

    def __init__(self, plain_query: str, 
                 query_embedding, 
                 client: OpenSearch):
        
        self.plain_query = plain_query
        self.query_embedding = query_embedding
        self.client = client


    @staticmethod
    def get_opensearch_client() -> OpenSearch:
        """
        Initializes and returns an OpenSearch client.

        Returns:
            OpenSearch: Configured OpenSearch client instance.
        """
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        logger.info("OpenSearch client initialized.")
        return client


    def hybrid_search(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search combining text-based and vector-based queries.

        Args:
            query_text (str): The text query for text-based search.
            query_embedding (List[float]): Embedding vector for vector-based search.
            top_k (int, optional): Number of top results to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of search results from OpenSearch.
        """
        client = self.client
        query_body = {
            "_source": {"exclude": ["embedding"]},  # Exclude embeddings from the results: we don't need them to build the response
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text": {"query": self.plain_query}}},  # Text-based search
                        {
                            "knn": {
                                "embedding": {
                                    "vector": self.query_embedding, # Vector-based search
                                    "k": top_k,
                                }
                            }
                        },
                    ]
                }
            },
            "size": top_k,
        }

        response = client.search(
            index=OPENSEARCH_INDEX, body=query_body, search_pipeline="nlp-search-pipeline"
        )
        logger.info(f"Hybrid search completed for query '{self.plain_query}' with top_k={top_k}.")

        hits: List[Dict[str, Any]] = response["hits"]["hits"] # dictionary with retrieved documents and their metadata 
        return hits