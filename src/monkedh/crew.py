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
from .tools.samu_notification_tool import SAMUNotificationTool
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
    # AGENT 1 : COLLECTEUR DE DONNÉES MÉDICALES
    # ============================================
    @agent
    def guideur_urgence_samu(self) -> Agent:
        return Agent(
            config=self.agents_config['guideur_urgence_samu'],
            tools=[self.rag_tool, self.serper_tool, self.webscraper_tool, self.image_tool],
            llm=llm,
            max_iter=3,
            cache=False,
            verbose=False,
        )
    @agent
    def notificateur_samu(self) -> Agent:
        return Agent(
            config=self.agents_config['notificateur_samu'],
            tools=[SAMUNotificationTool()],
            llm=llm,
            max_iter=1,
            cache=False,
            verbose=True,
        )
    

    # ============================================
    # TÂCHES DE L'AGENT COLLECTEUR
    # ============================================
    @task
    def creation_notification_urgence(self) -> Task:
        return Task(
            config=self.tasks_config['creation_notification_urgence'],
            output_file='notification.json'
        )
    @task
    def guidage_urgence_temps_reel(self) -> Task:
        return Task(
            config=self.tasks_config['guidage_urgence_temps_reel'],
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
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            short_term_memory=short_term_memory,
            function_calling_llm=llm,
            verbose=True,
            output_log_file="monkedh_crew.log",
            cache=False,
        )
