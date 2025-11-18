# RAG System - First Aid Manual

Semantic search over the first aid manual using Qdrant + Ollama.

## Setup

1. Install Ollama and pull model:
   ```bash
   ollama pull nomic-embed-text
   ```

2. Vectorize the document:
   ```bash
   python src/monkedh/tools/rag/vectorize_document.py
   ```

## Usage in CrewAI

```python
from monkedh.tools.rag import create_first_aid_search_tool
from monkedh.tools.rag.config import QDRANT_URL, QDRANT_API_KEY

# Create tool
rag_tool = create_first_aid_search_tool(
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)

# Add to agent
agent = Agent(
    role="First Aid Expert",
    tools=[rag_tool],
    ...
)
```

The agent will automatically use `search_first_aid_manual` when needed.
