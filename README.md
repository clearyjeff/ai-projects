# AI Projects

A collection of AI-powered tools for document processing, embeddings generation, and semantic search. This project includes three main components: a document processor for extracting structured data from PDFs using LLMs, a file embedder for creating and storing vector embeddings in ChromaDB, and a gRPC embedding server for scalable embedding generation.

## Components

### 1. Document Processor (`document_processor/`)

An intelligent document processing system that uses AI models (Gemini and Ollama) to extract structured data from documents, specifically designed for hardware estimation documents.

**Features:**
- PDF, DOCX, and TXT document processing
- AI-powered structured data extraction using Gemini 2.0 Flash or Ollama models
- Configurable extraction patterns and prompts
- Hardware-specific prompt for door hardware estimates
- JSON output with extracted document information

**Key Files:**
- `document_processor.py`: Main processor class with Gemini and Ollama integration
- Uses `../prompts/hardware_estimation_prompt.txt` for specialized hardware extraction

**Usage:**
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor("config.json")
result = processor.process_document("path/to/document.pdf")
```

### 2. File Embedder (`file_embedder/`)

A system for generating vector embeddings from CSV data and storing them in ChromaDB for semantic search and retrieval.

**Features:**
- CSV data ingestion and processing
- Remote gRPC embedding generation using Snowflake Arctic Embed model
- ChromaDB integration for persistent vector storage
- Batch processing for large datasets
- Semantic search and similarity queries

**Key Files:**
- `file_embedder.py`: Main embedding and ChromaDB integration logic
- `embeddings.proto`: Protocol buffer definitions for gRPC communication
- `embeddings_pb2.py` & `embeddings_pb2_grpc.py`: Generated gRPC client code

**Usage:**
```python
from file_embedder import embed_csv_to_chroma, query_chroma

# Embed CSV data
embed_csv_to_chroma("data.csv", "collection_name", "./db_path")

# Query embeddings
results = query_chroma("search query", "collection_name", "./db_path")
```

### 3. gRPC Embedding Server (`grpc_embedding_server/`)

A high-performance gRPC server for generating sentence embeddings using SentenceTransformers with GPU acceleration.

**Features:**
- gRPC-based embedding service
- CUDA GPU acceleration support
- Snowflake Arctic Embed model (335M parameters)
- Concurrent request handling
- Error handling and logging

**Key Files:**
- `server.py`: gRPC server implementation with SentenceTransformer model
- `client.py`: Example gRPC client for testing
- `embeddings.proto`: Service definitions
- `embeddings_pb2.py` & `embeddings_pb2_grpc.py`: Generated gRPC code

**Usage:**
```bash
# Start the server
python grpc_embedding_server/server.py

# Test with client
python grpc_embedding_server/client.py
```

## Configuration

The project uses `config.json` for configuration


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key in `config.json`

3. For gRPC functionality, ensure you have the protocol buffer compiler installed

## Dependencies

- **AI/ML**: `google-genai`, `ollama`, `sentence-transformers`, `chromadb`
- **Document Processing**: `pymupdf`, `python-docx`, `unstructured`, `PyPDF2`
- **Data Processing**: `pandas`, `numpy`, `spacy`
- **gRPC**: Generated from `embeddings.proto`
- **Other**: `boto3`, `requests`, `pydantic`, `langchain`

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Document       │    │  File Embedder   │    │  gRPC Server    │
│  Processor      │    │                  │    │                 │
│                 │    │  ┌─────────────┐ │    │  ┌────────────┐ │
│  PDF/DOCX ──────┼────┼─→│ ChromaDB    │ │◄───┼──│ Sentence   │ │
│  │              │    │  │ Vector DB   │ │    │  │ Transformer│ │
│  ▼              │    │  └─────────────┘ │    │  └────────────┘ │
│  Gemini/Ollama  │    │                  │    │                 │
│  │              │    │  CSV Data ───────┼────┼─→ Embeddings   │ │
│  ▼              │    │                  │    │                 │
│  Structured JSON│    │  gRPC Client ────┼────┼─→ gRPC Service │ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Example Workflows

### Hardware Document Processing
1. Place PDF document in `docs/` directory
2. Run document processor to extract structured data
3. Output saved as JSON for further processing

### Product Data Embedding
1. Prepare CSV with product data
2. Use file embedder to generate embeddings via gRPC server
3. Store embeddings in ChromaDB for semantic search
4. Query using natural language for product recommendations

## License

See LICENSE file for details.