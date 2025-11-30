"""
ONE-TIME SETUP SCRIPT: Vectorize the first aid document and upload to Qdrant.
Run this once to populate the vector database.

Usage: python src/monkedh/tools/rag/vectorize_document.py
"""

import sys
from pathlib import Path
import os # Added for environment variables

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv() 

from monkedh.tools.rag.vectorize import QdrantVectorizer
from monkedh.tools.rag.chunker import DocumentChunker


def main():
    """Main function to process and vectorize the document."""
    
    # ========================================
    # CONFIGURATION (CRITICAL FIX: Load from Environment)
    # ========================================
    
    # Qdrant Configuration - Load from environment variables
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    COLLECTION_NAME = "first_aid_manual"
    
    # Ollama Configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "embeddinggemma:latest")
    
    # Path to the document
    doc_path = project_root / "tests" / "data.txt"
    
    # Verification checks
    if not doc_path.exists():
        print(f"Error: Document not found at {doc_path}")
        return
        
    if not QDRANT_API_KEY:
        print("Error: QDRANT_API_KEY environment variable not set. Aborting.")
        return
    
    print("=" * 60)
    print("First Aid Document Vectorization Pipeline")
    print("=" * 60)
    
    # Step 1: Initialize the vectorizer
    print(f"\n[1/4] Initializing Qdrant vectorizer with Ollama...")
    
    vectorizer = QdrantVectorizer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        embedding_model=OLLAMA_MODEL,
        ollama_host=OLLAMA_HOST
    )
    
    # Check existing collections
    print("\nExisting collections:")
    collections = vectorizer.qdrant_client.get_collections()
    for col in collections.collections:
        print(f"  - {col.name}")
    
    # Step 2: Create collection
    print(f"\n[2/4] Creating Qdrant collection: {COLLECTION_NAME}")
    vectorizer.create_collection(COLLECTION_NAME, recreate=True)
    
    # Step 3: Process the document
    print(f"\n[3/4] Processing text document: {doc_path}")
    
    # Create chunks from text structure
    chunker = DocumentChunker(
        chunk_size=350,  
        chunk_overlap=80,
    )
    
    # Use the chunker to process the document
    chunks = chunker.process_document(
        file_path=str(doc_path),
        chunk_by_section=True,
        format="rt_manual"
    )
    
    print(f"\n  - Total chunks created: {len(chunks)}")
    
    # Step 4: Upload vectors to Qdrant
    print(f"\n[4/4] Uploading vectors to Qdrant...")
    
    # Enrichment/Contextualization used for INGESTION
    texts = []
    for chunk in chunks:
        # INGESTION Contextualization: Adding section title to the text being embedded
        contextualized_text = f"Document: {chunk['metadata']['section_title']}\nContenu: {chunk['text']}"
        texts.append(contextualized_text)
    
    metadata = [chunk["metadata"] for chunk in chunks]
    
    vectorizer.upload_vectors(
        collection_name=COLLECTION_NAME,
        texts=texts,
        metadata=metadata
    )
    
    # Test search
    print(f"\n{'=' * 60}")
    print("Testing Search...")
    print(f"{'=' * 60}")
    
    test_queries = [
            "Quels sont les signes d'un arrêt cardiaque chez l'adulte ?",
            "Comment administrer les premiers secours en cas de brûlure chimique ?",
            "Que faire si quelqu'un s'évanouit sans raison apparente ?",
            "Comment gérer une hémorragie externe abondante ?",
            "Quelles sont les étapes pour dégager les voies aériennes d'un bébé ?",
            "Comment reconnaître et traiter une crise d'épilepsie ?",
            "Que faire en cas d'intoxication alimentaire suspectée ?",
            "Comment immobiliser une fracture ouverte ?",
            "Quelles mesures prendre pour une victime en état de choc ?",
            "Comment effectuer un massage cardiaque sur un enfant ?"
        ]
    
    for test_query in test_queries:
        print(f"\nQuery: {test_query}")
        
        # Improvement: Contextualize the search query to match the ingestion format
        # QUERY Contextualization: Must match the format used during ingestion to align vector spaces
        contextualized_query = f"Contenu: {test_query}"
        
        results = vectorizer.search(
            collection_name=COLLECTION_NAME,
            query=contextualized_query,
            limit=2
        )
        
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n  Result {i+1} (score: {result['score']:.4f}):")
            print(f"    Section: {result['metadata'].get('section_title', 'N/A')}")
            print(f"    RT ID: {result['metadata'].get('rt_id', 'N/A')}")
            print(f"    Text: {result['text'][:150]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()