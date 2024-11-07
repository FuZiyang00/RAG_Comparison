
EMBEDDING_MODEL_PATH = "microsoft/mpnet-base"  # choose a model HuggingFace
ASSYMETRIC_EMBEDDING = False 
EMBEDDING_DIMENSION = 768  
TEXT_CHUNK_SIZE = 300  
OLLAMA_MODEL_NAME = "llama3.2:1b"  



####################################################################################################
# Dont change the following settings
####################################################################################################

# Logging
LOG_FILE_PATH = "logs/app.log"  # File path for the application log file

# OpenSearch settings
OPENSEARCH_HOST = "localhost"  # Hostname for the OpenSearch instance
OPENSEARCH_PORT = 9200  # Port number for OpenSearch
OPENSEARCH_INDEX = "documents"  # Index name for storing documents in OpenSearch