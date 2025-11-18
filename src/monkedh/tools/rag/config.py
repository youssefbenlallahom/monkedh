"""
Configuration for RAG tool integration with CrewAI agents.
"""

# Qdrant Configuration
QDRANT_URL = "https://d86967eb-8e90-4dad-94e9-0e60dcf6bd26.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.N42GdPLDSAGwz4kJtMedZy8_YDIYh2-Fcyq_dIkjMhI"
COLLECTION_NAME = "first_aid_manual"

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
