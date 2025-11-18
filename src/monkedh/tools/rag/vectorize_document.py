"""
ONE-TIME SETUP SCRIPT: Vectorize the first aid document and upload to Qdrant.
Run this once to populate the vector database.

Usage: python src/monkedh/tools/rag/vectorize_document.py
"""

import sys
import json
from pathlib import Path

# Add the project root to the path
# __file__ is in: src/monkedh/tools/rag/vectorize_document.py
# So we need to go up 4 levels to get to project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from monkedh.tools.rag.vectorize import QdrantVectorizer
from monkedh.tools.rag.chunker import DocumentChunker


def main():
    """Main function to process and vectorize the document."""
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Qdrant Configuration
    QDRANT_URL = "https://d86967eb-8e90-4dad-94e9-0e60dcf6bd26.europe-west3-0.gcp.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.N42GdPLDSAGwz4kJtMedZy8_YDIYh2-Fcyq_dIkjMhI"
    COLLECTION_NAME = "first_aid_manual"
    
    # Ollama Configuration
    OLLAMA_HOST = "http://localhost:11434"
    # Recommended models:
    # - "nomic-embed-text" (768D, best for semantic search)
    # - "mxbai-embed-large" (1024D, high quality)
    # - "llama3.2" (2048D, general purpose)
    OLLAMA_MODEL = "bge-m3"
    
    # Path to the document (tests folder is at project root, not in src)
    doc_path = project_root / "tests" / "output_unstructured.json"
    
    # Verify the file exists
    if not doc_path.exists():
        print(f"Error: Document not found at {doc_path}")
        print(f"Project root: {project_root}")
        print(f"Looking for: tests/output_unstructured.json")
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
    print(f"\n[3/4] Processing JSON document: {doc_path}")
    
    # Load JSON document
    with open(doc_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"  - Total RT sections: {len(json_data['content_chunks'])}")
    
    # Create chunks from JSON structure
    chunker = DocumentChunker(
        chunk_size=1000,  # 1000 characters per chunk
        chunk_overlap=200  # 200 characters overlap
    )
    
    def format_value(key, value, indent=0):
        """Recursively format any JSON value into readable text."""
        prefix = "  " * indent
        
        if isinstance(value, str):
            return f"{prefix}{value}\n"
        elif isinstance(value, list):
            if not value:
                return ""
            # Check if list of primitives or objects
            if isinstance(value[0], (str, int, float)):
                return "".join([f"{prefix}• {item}\n" for item in value])
            else:
                # List of objects - format each
                result = ""
                for item in value:
                    result += format_value(key, item, indent)
                return result
        elif isinstance(value, dict):
            result = ""
            for k, v in value.items():
                if k in ['step', 'number', 'name', 'case', 'title', 'action']:
                    # Header fields
                    result += f"\n{prefix}{v}:\n"
                elif isinstance(v, (str, int, float)):
                    result += f"{prefix}{k.replace('_', ' ').title()}: {v}\n"
                elif isinstance(v, list):
                    result += f"{prefix}{k.replace('_', ' ').title()}:\n"
                    result += format_value(k, v, indent + 1)
                elif isinstance(v, dict):
                    result += f"{prefix}{k.replace('_', ' ').title()}:\n"
                    result += format_value(k, v, indent + 1)
            return result
        else:
            return f"{prefix}{value}\n"
    
    def create_chunks_from_section(rt_section):
        """Create comprehensive chunks from an RT section."""
        rt_id = rt_section.get('id', '')
        rt_number = rt_section.get('rt_number', 0)
        rt_title = rt_section.get('title', '')
        rt_category = rt_section.get('category', 'N/A')
        keywords = ', '.join(rt_section.get('keywords', []))
        
        section_chunks = []
        
        # 1. Introduction chunk (overview)
        intro_parts = []
        intro_parts.append(f"# {rt_title}\n")
        
        if rt_section.get('situation'):
            intro_parts.append(f"\n**Situation**: {rt_section['situation']}\n")
        
        if rt_section.get('definition'):
            intro_parts.append(f"\n**Définition**: {rt_section['definition']}\n")
        
        if rt_section.get('objectives'):
            intro_parts.append("\n**Objectifs**:\n")
            for obj in rt_section['objectives']:
                intro_parts.append(f"• {obj}\n")
        
        if rt_section.get('key_concepts'):
            intro_parts.append("\n**Concepts clés**:\n")
            for concept in rt_section['key_concepts']:
                intro_parts.append(f"• {concept}\n")
        
        if rt_section.get('causes'):
            intro_parts.append("\n**Causes**:\n")
            for cause in rt_section['causes']:
                intro_parts.append(f"• {cause}\n")
        
        if rt_section.get('risks'):
            intro_parts.append("\n**Risques**:\n")
            for risk in rt_section['risks']:
                intro_parts.append(f"• {risk}\n")
        
        if rt_section.get('recognition_signs'):
            intro_parts.append("\n**Signes de reconnaissance**:\n")
            for sign in rt_section['recognition_signs']:
                intro_parts.append(f"• {sign}\n")
        
        if rt_section.get('serious_signs'):
            intro_parts.append("\n**Signes de gravité**:\n")
            for sign in rt_section['serious_signs']:
                intro_parts.append(f"• {sign}\n")
        
        intro_text = "".join(intro_parts)
        
        if intro_text.strip():
            for chunk_idx, chunk_text in enumerate(chunker.chunk_text(intro_text)):
                section_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section_title": f"{rt_id}: {rt_title}",
                        "rt_id": rt_id,
                        "rt_number": str(rt_number),
                        "category": rt_category,
                        "keywords": keywords,
                        "content_type": "introduction",
                        "chunk_index": chunk_idx,
                        "source": str(doc_path)
                    }
                })
        
        # 2. Procedure chunks (all procedure types)
        procedure_fields = ['procedure', 'procedure_adult_child', 'procedure_infant', 
                           'procedure_general', 'procedure_grave', 'procedure_simple']
        
        for proc_field in procedure_fields:
            if rt_section.get(proc_field):
                proc_title = proc_field.replace('_', ' ').title()
                proc_text = f"# {rt_title} - {proc_title}\n\n"
                proc_text += format_value(proc_field, rt_section[proc_field])
                
                for chunk_idx, chunk_text in enumerate(chunker.chunk_text(proc_text)):
                    section_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section_title": f"{rt_id}: {rt_title}",
                            "rt_id": rt_id,
                            "rt_number": str(rt_number),
                            "category": rt_category,
                            "keywords": keywords,
                            "content_type": f"procedure_{proc_field}",
                            "chunk_index": chunk_idx,
                            "source": str(doc_path)
                        }
                    })
        
        # 3. Techniques chunks
        if rt_section.get('techniques'):
            tech_text = f"# {rt_title} - Techniques\n\n"
            tech_text += format_value('techniques', rt_section['techniques'])
            
            for chunk_idx, chunk_text in enumerate(chunker.chunk_text(tech_text)):
                section_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section_title": f"{rt_id}: {rt_title}",
                        "rt_id": rt_id,
                        "rt_number": str(rt_number),
                        "category": rt_category,
                        "keywords": keywords,
                        "content_type": "techniques",
                        "chunk_index": chunk_idx,
                        "source": str(doc_path)
                    }
                })
        
        # 4. Compression/Ventilation techniques (RT4, RT6)
        for tech_field in ['compression_techniques', 'ventilation_techniques']:
            if rt_section.get(tech_field):
                tech_text = f"# {rt_title} - {tech_field.replace('_', ' ').title()}\n\n"
                tech_text += format_value(tech_field, rt_section[tech_field])
                
                for chunk_idx, chunk_text in enumerate(chunker.chunk_text(tech_text)):
                    section_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section_title": f"{rt_id}: {rt_title}",
                            "rt_id": rt_id,
                            "rt_number": str(rt_number),
                            "category": rt_category,
                            "keywords": keywords,
                            "content_type": tech_field,
                            "chunk_index": chunk_idx,
                            "source": str(doc_path)
                        }
                    })
        
        # 5. Special rules/structures
        special_fields = ['pls_technique', 'tourniquet_rules', 'message_structure', 
                         'emergency_numbers', 'efficacy_criteria']
        
        for field in special_fields:
            if rt_section.get(field):
                field_text = f"# {rt_title} - {field.replace('_', ' ').title()}\n\n"
                field_text += format_value(field, rt_section[field])
                
                for chunk_idx, chunk_text in enumerate(chunker.chunk_text(field_text)):
                    section_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section_title": f"{rt_id}: {rt_title}",
                            "rt_id": rt_id,
                            "rt_number": str(rt_number),
                            "category": rt_category,
                            "keywords": keywords,
                            "content_type": field,
                            "chunk_index": chunk_idx,
                            "source": str(doc_path)
                        }
                    })
        
        # 6. Special cases
        if rt_section.get('special_cases'):
            for case in rt_section['special_cases']:
                case_name = case.get('case', 'Cas particulier')
                case_text = f"# {rt_title} - Cas particulier: {case_name}\n\n"
                case_text += format_value('special_case', case)
                
                for chunk_idx, chunk_text in enumerate(chunker.chunk_text(case_text)):
                    section_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section_title": f"{rt_id}: {rt_title}",
                            "rt_id": rt_id,
                            "rt_number": str(rt_number),
                            "category": rt_category,
                            "keywords": keywords,
                            "content_type": "special_case",
                            "case_name": case_name,
                            "chunk_index": chunk_idx,
                            "source": str(doc_path)
                        }
                    })
        
        # 7. Sub-sections (RT8 - Traumatisme)
        if rt_section.get('sub_sections'):
            for sub in rt_section['sub_sections']:
                sub_title = sub.get('title', 'Section')
                sub_text = f"# {rt_title} - {sub_title}\n\n"
                sub_text += format_value('sub_section', sub)
                
                for chunk_idx, chunk_text in enumerate(chunker.chunk_text(sub_text)):
                    section_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section_title": f"{rt_id}: {rt_title}",
                            "rt_id": rt_id,
                            "rt_number": str(rt_number),
                            "category": rt_category,
                            "keywords": keywords,
                            "content_type": "sub_section",
                            "sub_title": sub_title,
                            "chunk_index": chunk_idx,
                            "source": str(doc_path)
                        }
                    })
        
        # 8. Precautions and hygiene
        for field in ['precautions', 'hygiene_safety', 'hygiene_prevention']:
            if rt_section.get(field):
                field_text = f"# {rt_title} - {field.replace('_', ' ').title()}\n\n"
                for item in rt_section[field]:
                    field_text += f"• {item}\n"
                
                for chunk_idx, chunk_text in enumerate(chunker.chunk_text(field_text)):
                    section_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "section_title": f"{rt_id}: {rt_title}",
                            "rt_id": rt_id,
                            "rt_number": str(rt_number),
                            "category": rt_category,
                            "keywords": keywords,
                            "content_type": field,
                            "chunk_index": chunk_idx,
                            "source": str(doc_path)
                        }
                    })
        
        return section_chunks
    
    chunks = []
    
    for rt_section in json_data['content_chunks']:
        rt_id = rt_section.get('id', '')
        rt_title = rt_section.get('title', '')
        print(f"\n  Processing {rt_id}: {rt_title}")
        
        section_chunks = create_chunks_from_section(rt_section)
        chunks.extend(section_chunks)
        print(f"    Created {len(section_chunks)} chunks")
    
    print(f"\n  - Total chunks created: {len(chunks)}")
    
    # Display some sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i+1}:")
        print(f"    Section: {chunk['metadata']['section_title']}")
        print(f"    Type: {chunk['metadata']['content_type']}")
        print(f"    Text preview: {chunk['text'][:100]}...")
    
    # Step 4: Upload vectors to Qdrant
    print(f"\n[4/4] Uploading vectors to Qdrant...")
    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]
    
    vectorizer.upload_vectors(
        collection_name=COLLECTION_NAME,
        texts=texts,
        metadata=metadata
    )
    
    # Verify upload
    collection_info = vectorizer.qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\n{'=' * 60}")
    print("Vectorization Complete!")
    print(f"{'=' * 60}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors stored: {collection_info.points_count}")
    print(f"Vector dimensions: {collection_info.config.params.vectors.size}")
    
    # Test search
    print(f"\n{'=' * 60}")
    print("Testing Search...")
    print(f"{'=' * 60}")
    
    test_queries = [
        "Comment faire une réanimation cardio-pulmonaire?",
        "Que faire en cas d'étouffement?",
        "Comment protéger une victime d'accident?"
    ]
    
    for test_query in test_queries:
        print(f"\nQuery: {test_query}")
        
        results = vectorizer.search(
            collection_name=COLLECTION_NAME,
            query=test_query,
            limit=2
        )
        
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n  Result {i+1} (score: {result['score']:.4f}):")
            print(f"    Section: {result['metadata'].get('section_title', 'N/A')}")
            print(f"    Text: {result['text'][:150]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
