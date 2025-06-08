import grpc
import embeddings_pb2
import embeddings_pb2_grpc
from concurrent import futures
from sentence_transformers import SentenceTransformer
import logging
import time
import torch

print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "snowflake-arctic-embed:335m"

try:
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    logging.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

class EmbeddingsServicer(embeddings_pb2_grpc.EmbeddingsServiceServicer):
    """
    gRPC service implementation for generating sentence embeddings.
    
    This servicer handles requests to generate embeddings using a pre-loaded
    SentenceTransformer model and returns the results as serialized numpy arrays.
    """
    def GenerateEmbeddings(self, request, context):
        """
        Generate embeddings for sentences provided in the gRPC request.
        
        Args:
            request (embeddings_pb2.EmbeddingsRequest): The gRPC request containing
                                                      a list of sentences to embed.
            context (grpc.ServicerContext): The gRPC context for the request.
        
        Returns:
            embeddings_pb2.EmbeddingsResponse: Response containing serialized embeddings
                                              as bytes, or empty response on error.
        """
        try:
            start_time = time.time()
            sentences = request.sentences
            embeddings = model.encode(sentences)
            serialized_embeddings = [emb.tobytes() for emb in embeddings]
            end_time = time.time()
            logging.info(f"Generated embeddings for {len(sentences)} sentences in {end_time - start_time:.2f} seconds.")
            return embeddings_pb2.EmbeddingsResponse(embeddings=serialized_embeddings)
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("An internal error occurred.")
            return embeddings_pb2.EmbeddingsResponse() 

def serve():
    """
    Start and run the gRPC embeddings server.
    
    Creates a gRPC server with the EmbeddingsServicer, binds it to localhost:50051,
    and starts listening for requests. 
    
    Raises:
        Logs any server errors but does not re-raise exceptions.
    """
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4)) 
        embeddings_pb2_grpc.add_EmbeddingsServiceServicer_to_server(EmbeddingsServicer(), server)
        server.add_insecure_port('localhost:50051') 
        server.start()
        logging.info("Server started on localhost:50051") 
        server.wait_for_termination()
    except Exception as e:
        logging.error(f"Server error: {e}")

if __name__ == '__main__':
    serve()