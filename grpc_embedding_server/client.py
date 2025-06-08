import grpc
import embeddings_pb2
import embeddings_pb2_grpc
import numpy as np
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_embeddings(sentences, server_address=None):
    if server_address is None:
        grpc_host = os.getenv("GRPC_SERVER_HOST", "localhost")
        grpc_port = os.getenv("GRPC_SERVER_PORT", "50051")
        server_address = f"{grpc_host}:{grpc_port}"
    """
    Generate embeddings for a list of sentences using a gRPC server.
    
    Args:
        sentences (list[str]): List of sentences to generate embeddings for.
        server_address (str): Address of the gRPC server (e.g., 'localhost:50051').
    
    Returns:
        list[numpy.ndarray] or None: List of embedding vectors as numpy arrays,
                                   or None if an error occurred.
    
    Raises:
        Logs gRPC errors and general exceptions but returns None instead of raising.
    """
    try:
        with grpc.insecure_channel(server_address) as channel:
            stub = embeddings_pb2_grpc.EmbeddingsServiceStub(channel)
            request = embeddings_pb2.EmbeddingsRequest(sentences=sentences)
            response = stub.GenerateEmbeddings(request)
            if response.embeddings: 
                embeddings = [np.frombuffer(emb, dtype=np.float32) for emb in response.embeddings]
                return embeddings
            else:
                logging.error("Server returned an empty embeddings response.")
                return None

    except grpc.RpcError as e:
        logging.error(f"gRPC error: {e.details()}, code: {e.code()}")
        return None

    except Exception as e:
        logging.error(f"Client error: {e}")
        return None

if __name__ == '__main__':
    sentences = ["This is a test sentence.", "Another sentence for embedding."]
    embeddings = generate_embeddings(sentences)
    if embeddings is not None:
        print(embeddings)