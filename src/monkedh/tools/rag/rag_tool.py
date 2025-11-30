"""
CrewAI custom tool for RAG-based search in the first aid manual.
"""

from crewai.tools import BaseTool
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from .vectorize import QdrantVectorizer


class FirstAidSearchInput(BaseModel):
    """Input schema for FirstAidSearchTool."""
    query: str = Field(
        ...,
        description="The search query in French. Ask about first aid procedures, emergency protocols, or medical interventions."
    )
    top_k: int = Field(
        default=5,
        description="Number of relevant results to return (default: 5)",
        ge=1,
        le=20
    )


class FirstAidSearchTool(BaseTool):
    """
    Tool for searching the first aid manual using semantic search.
    
    This tool allows agents to query a vectorized first aid manual to find
    relevant information about emergency procedures, medical interventions,
    and safety protocols.
    """
    
    name: str = "search_first_aid_manual"
    description: str = (
        "Search the comprehensive first aid manual (Manuel des Premiers Secours) "
        "to find relevant information about emergency procedures, medical interventions, "
        "CPR techniques, choking protocols, alerting procedures, victim protection, "
        "and other first aid topics. "
        "Use this tool when you need specific information about how to respond to "
        "medical emergencies or when looking for official first aid procedures. "
        "Query in French for best results."
    )
    args_schema: Type[BaseModel] = FirstAidSearchInput
    
    # Configuration (will be set during initialization)
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "first_aid_manual"
    embedding_model: str = "embeddinggemma:latest"
    ollama_host: str = "http://localhost:11434"
    
    # Cached vectorizer instance
    _vectorizer: Optional[QdrantVectorizer] = None
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "first_aid_manual",
        embedding_model: str = "embeddinggemma:latest",
        ollama_host: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize the FirstAidSearchTool.
        
        Args:
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            collection_name: Name of the collection (default: "first_aid_manual")
            embedding_model: Ollama model for embeddings (default: "nomic-embed-text")
            ollama_host: Ollama server URL (default: "http://localhost:11434")
        """
        super().__init__(**kwargs)
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host
    
    def _get_vectorizer(self) -> QdrantVectorizer:
        """Get or create the vectorizer instance (lazy loading)."""
        if self._vectorizer is None:
            self._vectorizer = QdrantVectorizer(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                embedding_model=self.embedding_model,
                ollama_host=self.ollama_host
            )
        return self._vectorizer
    
    def _run(
        self,
        query: str,
        top_k: int = 2,
        **kwargs: Any
    ) -> str:
        """
        Execute the search and return formatted results.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            Formatted string with search results
        """
        try:
            # Get vectorizer
            vectorizer = self._get_vectorizer()
            
            # Perform search
            results = vectorizer.search(
                collection_name=self.collection_name,
                query=query,
                limit=top_k
            )
            
            if not results:
                return (
                    f"No results found for query: '{query}'. "
                    "Try rephrasing your question or using different keywords."
                )
            
            # Format results
            formatted_output = []
            formatted_output.append(f"🔍 Search Results for: '{query}'\n")
            formatted_output.append(f"Found {len(results)} relevant sections:\n")
            formatted_output.append("=" * 80)
            
            for i, result in enumerate(results, 1):
                score = result['score']
                text = result['text']
                metadata = result['metadata']
                
                formatted_output.append(f"\n[Result {i}] Relevance: {score:.2%}")
                
                # Add section information if available
                if 'section_title' in metadata:
                    formatted_output.append(f"📖 Section: {metadata['section_title']}")
                
                if 'section_level' in metadata:
                    formatted_output.append(f"   Level: H{metadata['section_level']}")
                
                # Add the text content
                formatted_output.append(f"\n📝 Content:\n{text}")
                formatted_output.append("\n" + "-" * 80)
            
            return "\n".join(formatted_output)
            
        except Exception as e:
            return (
                f"❌ Error searching first aid manual: {str(e)}\n"
                "Please ensure Ollama is running and the Qdrant collection exists."
            )


# Factory function to create the tool easily
def create_first_aid_search_tool(
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = "first_aid_manual",
    embedding_model: str = "embeddinggemma:latest",
    ollama_host: str = "http://localhost:11434"
) -> FirstAidSearchTool:
    """
    Factory function to create a FirstAidSearchTool instance.
    
    Args:
        qdrant_url: Qdrant cluster URL
        qdrant_api_key: Qdrant API key
        collection_name: Name of the collection (default: "first_aid_manual")
        embedding_model: Ollama model for embeddings (default: "nomic-embed-text")
        ollama_host: Ollama server URL (default: "http://localhost:11434")
        
    Returns:
        Configured FirstAidSearchTool instance
        
    Example:
        ```python
        from monkedh.tools.rag.rag_tool import create_first_aid_search_tool
        
        rag_tool = create_first_aid_search_tool(
            qdrant_url="https://your-cluster.qdrant.io",
            qdrant_api_key="your-api-key"
        )
        
        # Use in agent
        agent = Agent(
            role="First Aid Expert",
            goal="Provide accurate first aid guidance",
            tools=[rag_tool]
        )
        ```
    """
    return FirstAidSearchTool(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_host=ollama_host
    )
