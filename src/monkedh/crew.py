from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List
import os
from crewai import LLM
from .tools.redis_storage import RedisStorage
from crewai.memory.short_term.short_term_memory import ShortTermMemory
import dotenv
from monkedh.tools.rag import create_first_aid_search_tool
from monkedh.tools.rag.config import QDRANT_URL, QDRANT_API_KEY
from .tools.image_suggestion import search_emergency_image


dotenv.load_dotenv()

# Disable SSL warnings for TokenFactory
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

embedder_config = {
            "provider": "openai",
            "api_type": "azure",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": os.getenv('AZURE_API_BASE'),
            "api_version": "2024-12-01-preview",
            "deployment_id": "text-embedding-3-small",
            "model_name": "text-embedding-3-small",
            "dimensions": 1536
        }

# Azure OpenAI LLM
llm = LLM(
    model=os.getenv("model"),
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv('AZURE_API_BASE'),
    api_version=os.getenv("AZURE_API_VERSION"),
    stream=False,
)

# TokenFactory LLM (Llama-3.1-70B) - without http_client
llm_tokenfactory = LLM(
    model="openai/hosted_vllm/Llama-3.1-70B-Instruct",
    api_key="sk-3b38b0ddab444c4d9b3ff5e14ef9654d",
    base_url="https://tokenfactory.esprit.tn/api",
    temperature=0.1,
    top_p=0.9
)

@CrewBase
class Monkedh():
    """Monkedh crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    
    serper_tool = SerperDevTool(
        country="fr",
        locale="fr",
        n_results=2,
    )
    webscraper_tool = ScrapeWebsiteTool()
    rag_tool = create_first_aid_search_tool(
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
    )
    image_tool = search_emergency_image
    
    # ============================================
    # AGENT UNIQUE : ASSISTANT MÉDICAL COMPLET
    # ============================================
    @agent
    def assistant_urgence_medical(self) -> Agent:
        """
        Agent unique qui gère tout : détection d'urgence, guidage 
        et images illustratives.
        Réduit la latence en éliminant la délégation inter-agents.
        """
        return Agent(
            config=self.agents_config['assistant_urgence_medical'],
            tools=[
                self.image_tool,         # Images premiers secours (PRIORITAIRE)
                self.rag_tool,           # Protocoles médicaux
                self.serper_tool,        # Recherche web (pharmacies, hôpitaux)
                self.webscraper_tool,    # Scraping de pages
            ],
            llm=llm,
            max_iter=3,
            cache=False,  # Désactivé pour forcer l'appel des outils
            verbose=True,
        )

    # ============================================
    # TÂCHE UNIQUE : ASSISTANCE MÉDICALE COMPLÈTE
    # ============================================
    @task
    def assistance_medicale_complete(self) -> Task:
        """
        Tâche unique qui combine détection, guidage et notification.
        """
        return Task(
            config=self.tasks_config['assistance_medicale_complete'],
            output_file='protocols_urgences.json'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Monkedh crew"""
        redis_storage = RedisStorage(
            host=os.getenv("REDIS_HOST", "redis-13350.c339.eu-west-3-1.ec2.redns.redis-cloud.com"),
            port=int(os.getenv("REDIS_PORT", 13350)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD","YoLErdUztvwgDQvhAr1Fgbp0NUdekrRm"),
            namespace="monkedh",
        )
        
        short_term_memory = ShortTermMemory(storage=redis_storage)
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            short_term_memory=short_term_memory,
            function_calling_llm=llm,
            verbose=True,
            output_log_file="monkedh_crew.log",
            cache=True,
        )
