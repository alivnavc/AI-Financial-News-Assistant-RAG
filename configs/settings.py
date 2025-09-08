import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

class Settings:
    """
    Centralized configuration for the RAG Finance pipeline.

    This class reads environment variables and provides defaults for:
      - Qdrant vector database
      - OpenAI API usage
      - LangChain tracing
      - Miscellaneous settings
    """

    # ---------------- Qdrant Settings ----------------
    # Hostname for Qdrant (default: localhost)
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    # REST API port for Qdrant (default: 6333)
    QDRANT_REST_PORT = int(os.getenv("QDRANT_REST_PORT", 6333))
    # gRPC port for Qdrant (default: 6334)
    QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
    # Default collection name for storing news embeddings
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "news_articles")

    # ---------------- OpenAI Settings ----------------
    # API key for OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # LLM model for chat/completion tasks
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    # Embedding model for vector representations
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    # ---------------- LangChain Settings ----------------
    # Enable LangChain tracing (v2) for debugging/observability
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    # LangChain API key for logging/tracing
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    # Project name for LangChain tracing
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-finance")

    # ---------------- Miscellaneous Settings ----------------
    # Similarity threshold for semantic search filtering
    SIMILARITY_THRESHOLD = 0.4
    # Log file path
    LOG_FILE = "rag_finance.log"
