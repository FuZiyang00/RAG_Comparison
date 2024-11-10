
EMBEDDING_MODEL_PATH = 'all-mpnet-base-v2' # Choose a model that produces embeddings that match the EMBEDDING_DIMENSION
EMBEDDING_DIMENSION = 768 
ASSYMETRIC_EMBEDDING = False  
TEXT_CHUNK_SIZE = 300  
OLLAMA_MODEL_NAME = "llama3.2:1b" 
HYBRID_SEARCH = True
K_RESULTS = 5
TEMPERATURE = 0.5



####################################################################################################
# Dont change the following settings
####################################################################################################

# Logging
LOG_FILE_PATH = "logs/app.log"  # File path for the application log file

# OpenSearch settings
OPENSEARCH_HOST = "localhost"  # Hostname for the OpenSearch instance
OPENSEARCH_PORT = 9200  # Port number for OpenSearch
OPENSEARCH_INDEX = "documents"  # Index name for storing documents in OpenSearch