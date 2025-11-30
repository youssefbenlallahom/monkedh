from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import ollama
import hashlib
import os
import math
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class QdrantVectorizer:
    """Handles document vectorization and Qdrant operations using Ollama embeddings."""
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embedding_model: str = "bge-m3",
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize the vectorizer with Qdrant and Ollama.
        """
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=120  # Increased timeout for large batch operations
        )
        
        self.embedding_model = embedding_model
        ollama_host = os.getenv("OLLAMA_HOST", ollama_host)
        self.ollama_client = ollama.Client(host=ollama_host)
        
        # Auto-detect embedding dimensions
        print(f"  Detecting embedding dimensions for {embedding_model}...")
        try:
            # Send single string for test to comply with client's Pydantic validation
            test_response = self.ollama_client.embeddings(
                model=embedding_model,
                prompt="test"
            )
            self.embedding_dimensions = len(test_response["embedding"])
            print(f"✓ Using Ollama: {embedding_model} ({self.embedding_dimensions}D)")
            print(f"  Host: {ollama_host}")
        except Exception as e:
            print(f"Error detecting dimensions from Ollama. Check if Ollama is running at {ollama_host}.")
            print(f"Details: {e}")
            self.embedding_dimensions = 768 # Fallback dimension


    def _create_payload_indexes(self, collection_name: str) -> None:
        """Helper to create necessary payload indexes."""
        # Index fields that will be used for filtering and search
        index_fields = {
            "rt_id": PayloadSchemaType.KEYWORD,
            "numero_rt": PayloadSchemaType.INTEGER,
            "section_title": PayloadSchemaType.TEXT,
        }
        
        print(f"Creating payload indexes for collection '{collection_name}'...")
        for field, schema_type in index_fields.items():
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception:
                # Index might already exist
                pass
    
    def create_collection(self, collection_name: str, recreate: bool = False) -> None:
        """
        Create a Qdrant collection for storing vectors and set up payload indexes.
        """
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if collection_exists:
            if recreate:
                print(f"Deleting existing collection: {collection_name}")
                self.qdrant_client.delete_collection(collection_name)
            else:
                print(f"Collection '{collection_name}' already exists")
                self._create_payload_indexes(collection_name)
                return
        
        print(f"Creating collection: {collection_name}")
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimensions,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created successfully")
        
        # Improvement: Create payload indexes for fast filtering
        self._create_payload_indexes(collection_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Ollama.
        
        FIX: Iterates through texts one-by-one to avoid Pydantic validation error 
        when passing a list to the 'prompt' argument.
        """
        embeddings = []
        total_texts = len(texts)
        
        print(f"  Generating embeddings for {total_texts} texts (one-by-one)...")

        # Iterate through texts and call the embeddings API for each one
        for idx, text in enumerate(texts):
            # Print progress every 10 texts, or for the first few
            if total_texts > 10 and (idx + 1) % 10 == 0 or idx < 5:
                # print(f"    Processing text {idx + 1}/{total_texts}...")
                pass # Suppress frequent progress updates here
                
            try:
                # Send individual text as a string to comply with client validation
                response = self.ollama_client.embeddings(
                    model=self.embedding_model,
                    prompt=text 
                )
                embeddings.append(response["embedding"])
                
            except Exception as e:
                # In case of an error, print a warning and append an empty list to maintain array length consistency
                print(f"  ✗ Error generating embedding for text {idx + 1}. Skipping. Error: {e}")
                embeddings.append([])
        
        # Filter out failed embeddings (empty lists)
        valid_embeddings = [e for e in embeddings if e and len(e) == self.embedding_dimensions]
        if len(valid_embeddings) != total_texts:
            print(f"  Warning: {total_texts - len(valid_embeddings)} embeddings failed to generate.")
            
        return valid_embeddings
    
    def upload_vectors(
        self,
        collection_name: str,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> None:
        """
        Generate embeddings and upload vectors to Qdrant.
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        if len(texts) != len(metadata):
            raise ValueError("Number of texts and metadata items must match")
        
        print(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings_list = self.generate_embeddings(texts)

        if len(texts) != len(embeddings_list):
            print(f"Warning: {len(texts)} texts, but only {len(embeddings_list)} embeddings generated. Aborting upload.")
            return
        
        print(f"Uploading {len(embeddings_list)} vectors to Qdrant (batch size: {batch_size})...")
        points = []
        
        for idx, (text, embedding, meta) in enumerate(zip(texts, embeddings_list, metadata)):
            
            # Improvement: Create a stable, hashed ID for idempotency
            unique_id_string = f"{text}|{meta.get('source', 'unknown')}|{meta.get('chunk_index', idx)}"
            # Convert SHA256 hash to an integer ID
            point_id = int(hashlib.sha256(unique_id_string.encode()).hexdigest(), 16) % (2**63 - 1)
            
            payload = {
                "text": text,
                **meta
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
            
            if len(points) >= batch_size or idx == len(texts) - 1:
                try:
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True
                    )
                    points = []
                except Exception as e:
                    print(f"  ✗ Critical Error uploading batch at index {idx}. Aborting. Error: {e}")
                    break
        
        print(f"  ✓ Successfully uploaded all vectors to collection '{collection_name}'")
    
    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.
        """
        # Generate embedding for query (the query must be contextualized by the caller)
        print(f"  Generating embedding for contextualized query: '{query[:50]}...'")
        
        # Only send the query string, not a list
        query_embedding_list = self.generate_embeddings([query])
        
        if not query_embedding_list:
            print("  ✗ Failed to generate query embedding.")
            return []
            
        query_vector = query_embedding_list[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            text_content = result.payload.pop("text", "")
            results.append({
                "text": text_content,
                "score": result.score,
                "metadata": result.payload
            })
        
        return results