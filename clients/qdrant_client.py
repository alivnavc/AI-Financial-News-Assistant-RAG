from qdrant_client import QdrantClient
from configs.settings import Settings

"""
This module sets up a singleton Qdrant client to interact with a Qdrant vector database.

We are using gRPC for communication because it is faster and more efficient
than REST for large-scale vector operations. gRPC supports streaming,
binary serialization, and lower latency, making it ideal for embeddings search.
"""

# Singleton connection pooling
qdrant_client = QdrantClient(
    host=Settings.QDRANT_HOST,
    port=Settings.QDRANT_REST_PORT,
    grpc_port=Settings.QDRANT_GRPC_PORT
)
