#!/usr/bin/env python
import sys
import warnings
import uuid

from monkedh.crew import Monkedh
from monkedh.tools.redis_storage import redis_memory

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Launch an interactive terminal chat with the CPR assistant.
    """
    print("Assistant RCP virtuel - saisissez vos questions (tapez 'q' pour quitter).")
    crew_factory = Monkedh()
    channel_id = "default_channel"  # Doit correspondre à la mémoire RedisStorage
    user_id = str(uuid.uuid4())
    username = "Temoin"

    while True:
        try:
            question = input("\nVotre question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession terminee.")
            break

        if not question:
            continue

        if question.lower() in {"q", "quit", "exit"}:
            print("Fermeture de l'assistant. Prenez soin de vous.")
            break

        # Récupérer les 10 derniers messages de l'historique Redis
        conversation_history = redis_memory.get_conversation_pairs(
            channel_id=channel_id,
            limit=10
        )

        # Construire le contexte de conversation
        conversation_context = redis_memory.build_conversation_context(conversation_history)

        print(f"\n📚 Contexte récupéré: {len(conversation_history)} messages antérieurs\n")
        
        # Ajouter le contexte et la question actuelle aux inputs
        # IMPORTANT: Ces variables seront injectées dans les prompts des tasks
        inputs = {
            "question": question,
            "conversation_history": conversation_context if conversation_context else "Aucun historique précédent. C'est le début de la conversation."
        }

        try:
            crew = crew_factory.crew()
            result = crew.kickoff(inputs=inputs)
            output = getattr(result, "raw", result)
            print(f"\n{output}\n")
            
            # Store the conversation pair in Redis for future reference
            redis_memory.store_conversation_pair(
                channel_id=channel_id,
                user_id=user_id,
                user_query=question,
                bot_response=output,
                username=username
            )
        except Exception as exc:
            print(f"\nUne erreur est survenue pendant l'execution: {exc}\n")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "question": (
            "Je suis a la salle de sport, un homme de 35 ans vient de s'effondrer, il ne respire plus."
            " J'ai un telephone sur haut-parleur et une trousse de secours. Que faire maintenant ?"
        )
    }
    try:
        Monkedh().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Monkedh().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "question": (
            "Dans une gare, un adolescent respire a peine, j'ai un DEA automatique et une couverture."
            " Les secours arrivent dans cinq minutes, comment assurer la RCP en attendant ?"
        )
    }
    
    try:
        Monkedh().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
