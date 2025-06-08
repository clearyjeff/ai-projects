from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
import pandas as pd
import grpc
import embeddings_pb2
import embeddings_pb2_grpc
import numpy as np
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "snowflake-arctic-embed:335m"

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        try:
            grpc_host = os.getenv("GRPC_SERVER_HOST", "localhost")
            grpc_port = os.getenv("GRPC_SERVER_PORT", "50051")
            server_address = f"{grpc_host}:{grpc_port}"
            with grpc.insecure_channel(server_address) as channel:
                stub = embeddings_pb2_grpc.EmbeddingsServiceStub(channel)
                request = embeddings_pb2.EmbeddingsRequest(sentences=input)
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

def create_product_text(row):
    """Combines relevant text fields from a DataFrame row."""
    return f"{row['Item desc']} {row['Category']} {row['SubCategory']} {row['Finish Description']}"

def embed_csv_to_chroma(csv_file_path, collection_name="product_data", db_path=None):
    if db_path is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    """Embeds data from a CSV file into a ChromaDB collection."""

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_file_path}")
        return

    logging.info(f"Successfully loaded {len(df)} rows from CSV file.")
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=MyEmbeddingFunction())

    logging.info(f"Starting embedding at {pd.Timestamp.now()}")
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]

        documents = []
        ids = []
        metadatas = []

        for index, row in batch_df.iterrows():
            documents.append(create_product_text(row)) 
            ids.append(row["SKU"])
            metadata = {k: v for k, v in row.items() if k != "Item desc"}
            metadatas.append(metadata)

        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        #print(f"Current batch embedded {i:i + batch_size} documents.")
    
    logging.info(f"Finished embedding at {pd.Timestamp.now()}")

def query_chroma(query_text, collection_name="product_data", db_path=None):
    if db_path is None:
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    """Queries a ChromaDB collection."""
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=MyEmbeddingFunction())

    results = collection.query(
        query_texts=[query_text],
        n_results=10,
    )

    return results

def generate_embeddings(sentences, server_address=None):
    if server_address is None:
        grpc_host = os.getenv("GRPC_SERVER_HOST", "localhost")
        grpc_port = os.getenv("GRPC_SERVER_PORT", "50051")
        server_address = f"{grpc_host}:{grpc_port}"
    
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

csv_file_path = "../docs/all_products.csv" 
embed_csv_to_chroma(csv_file_path, 'all_products', './all_products.db')

example_query = "Glynn-Johnson Overhead Stop 903S 630"
query_results = query_chroma(example_query, 'all_products', './all_products.db')
print(query_results)

