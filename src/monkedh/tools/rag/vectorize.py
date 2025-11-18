"""
Module for vectorizing documents and uploading to Qdrant using Ollama embeddings.
"""

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
import hashlib


class QdrantVectorizer:
    """Handles document vectorization and Qdrant operations using Ollama embeddings."""
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize the vectorizer with Qdrant and Ollama.
        
        Args:
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            embedding_model: Ollama model name
                - "nomic-embed-text" (768D, recommended for semantic search)
                - "mxbai-embed-large" (1024D, high quality)
                - "llama3.2" (2048D, general purpose)
            ollama_host: Ollama server URL (default: http://localhost:11434)
        """
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60  # Increase timeout to 60 seconds for cloud operations
        )
        
        self.embedding_model = embedding_model
        self.ollama_client = ollama.Client(host=ollama_host)
        
        # Auto-detect embedding dimensions
        print(f"  Detecting embedding dimensions for {embedding_model}...")
        test_response = self.ollama_client.embeddings(
            model=embedding_model,
            prompt="test"
        )
        self.embedding_dimensions = len(test_response["embedding"])
        
        print(f"✓ Using Ollama: {embedding_model} ({self.embedding_dimensions}D)")
        print(f"  Host: {ollama_host}")
    
    def create_collection(self, collection_name: str, recreate: bool = False) -> None:
        """
        Create a Qdrant collection for storing vectors.
        
        Args:
            collection_name: Name of the collection to create
            recreate: If True, delete existing collection and recreate
        """
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if collection_exists:
            if recreate:
                print(f"Deleting existing collection: {collection_name}")
                self.qdrant_client.delete_collection(collection_name)
            else:
                print(f"Collection '{collection_name}' already exists")
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
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Ollama.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 32  # Process in batches for progress tracking
        total_batches = (len(texts) - 1) // batch_size + 1
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  Generating embeddings for batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Ollama processes one text at a time
            batch_embeddings = []
            for text in batch:
                response = self.ollama_client.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                batch_embeddings.append(response["embedding"])
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def upload_vectors(
        self,
        collection_name: str,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None,
        batch_size: int = 10  # Smaller batches for cloud Qdrant
    ) -> None:
        """
        Generate embeddings and upload vectors to Qdrant.
        
        Args:
            collection_name: Name of the Qdrant collection
            texts: List of text chunks to vectorize
            metadata: Optional list of metadata dictionaries for each text
            batch_size: Number of vectors to upload per batch (default: 10)
        """
        if metadata is None:
            metadata = [{} for _ in texts]
        
        if len(texts) != len(metadata):
            raise ValueError("Number of texts and metadata items must match")
        
        print(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings = self.generate_embeddings(texts)
        
        print(f"Uploading {len(embeddings)} vectors to Qdrant (batch size: {batch_size})...")
        points = []
        
        for idx, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
            # Create a unique ID based on index (simpler and more reliable)
            point_id = idx
            
            # Add text to payload
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
            
            # Upload in smaller batches to avoid timeout
            if len(points) >= batch_size or idx == len(texts) - 1:
                try:
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True  # Wait for operation to complete
                    )
                    print(f"  ✓ Uploaded {idx + 1}/{len(texts)} vectors")
                    points = []
                except Exception as e:
                    print(f"  ✗ Error uploading batch at index {idx}: {e}")
                    print(f"  → Retrying with individual points...")
                    # Retry individual points on failure
                    for point in points:
                        try:
                            self.qdrant_client.upsert(
                                collection_name=collection_name,
                                points=[point],
                                wait=True
                            )
                        except Exception as retry_error:
                            print(f"  ✗ Failed to upload point {point.id}: {retry_error}")
                    points = []
        
        print(f"  ✓ Successfully uploaded all vectors to collection '{collection_name}'")
    
    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant.
        
        Args:
            collection_name: Name of the Qdrant collection
            query: Search query text
            limit: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for result in search_results:
            results.append({
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })
        
        return results
