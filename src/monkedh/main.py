#!/usr/bin/env python
import sys
import warnings
import uuid
import argparse

from monkedh.crew import Monkedh
from monkedh.tools.redis_storage import redis_memory

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def process_question(crew_factory, channel_id, user_id, username, question):
    """Process a question through the CrewAI medical agents."""
    # Get conversation history
    conversation_history = redis_memory.get_conversation_pairs(
        channel_id=channel_id,
        limit=10
    )
    conversation_context = redis_memory.build_conversation_context(conversation_history)
    
    print(f"\n📚 Contexte récupéré: {len(conversation_history)} messages antérieurs\n")
    
    inputs = {
        "question": question,
        "conversation_history": conversation_context if conversation_context else "Aucun historique précédent. C'est le début de la conversation."
    }
    
    try:
        crew = crew_factory.crew()
        result = crew.kickoff(inputs=inputs)
        output = getattr(result, "raw", str(result))
        
        # Store conversation
        redis_memory.store_conversation_pair(
            channel_id=channel_id,
            user_id=user_id,
            user_query=question,
            bot_response=output,
            username=username
        )
        
        return output
        
    except Exception as exc:
        return f"Une erreur est survenue: {exc}"


def run_text_mode():
    """Run the chatbot in text mode (original behavior)."""
    print("Assistant RCP virtuel - saisissez vos questions (tapez 'q' pour quitter).")
    crew_factory = Monkedh()
    channel_id = "default_channel"
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

        output = process_question(crew_factory, channel_id, user_id, username, question)
        print(f"\n{output}\n")


def run_voice_mode(voice_type: str = "shimmer"):
    """Run the chatbot in voice mode using GPT-Realtime."""
    import time
    import re
    
    print("\n" + "="*60)
    print("🚑 ASSISTANT D'URGENCE VOCAL - CREWAI + GPT-REALTIME")
    print("="*60)
    print("🎤 Voice: Azure GPT-Realtime (STT + TTS)")
    print("🧠 Brain: CrewAI Medical Agents (RAG, Protocols, Images)")
    print("="*60 + "\n")
    
    try:
        from monkedh.tools.voice.gpt_realtime import GPTRealtimeVoice
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("   Installez: pip install pyaudio websockets")
        return
    
    # Initialize voice with selected voice type
    try:
        voice = GPTRealtimeVoice(voice=voice_type)
    except Exception as e:
        print(f"❌ Erreur initialisation voice: {e}")
        return
    
    if not voice.is_available():
        print("❌ Dépendances manquantes.")
        print("   Installez: pip install pyaudio websockets")
        return
    
    # Initialize CrewAI
    print("🔧 Initialisation CrewAI...")
    crew_factory = Monkedh()
    channel_id = "voice_channel"
    user_id = str(uuid.uuid4())
    username = "Temoin_Vocal"
    
    def clean_for_speech(text: str) -> str:
        """Clean text for TTS."""
        # Remove markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Remove image paths
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'Image suggérée:.*?\.png', '', text)
        text = re.sub(r'📷.*?\.png', '', text)
        # Clean whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    print("\n" + "="*50)
    print("🎤 MODE VOCAL ACTIVÉ")
    print("="*50)
    print("Parlez naturellement - l'assistant vous répondra")
    print("Dites 'quitter' ou 'au revoir' pour arrêter")
    print("="*50 + "\n")
    
    # Welcome message
    welcome = "Bonjour, je suis l'assistant d'urgence du SAMU. Décrivez votre situation."
    print(f"🤖 Assistant: {welcome}")
    voice.speak(welcome)
    
    exit_words = ["quitter", "au revoir", "stop", "arrêter", "fin", "bye"]
    
    while True:
        try:
            print("\n" + "-"*50)
            print("🎤 C'EST VOTRE TOUR DE PARLER...")
            print("-"*50)
            
            user_text = voice.listen_once(timeout_seconds=15)
            
            if not user_text:
                retry_msg = "Je n'ai pas compris. Pouvez-vous répéter?"
                print(f"🤖 Assistant: {retry_msg}")
                voice.speak(retry_msg)
                continue
            
            print(f"\n👤 Vous: {user_text}")
            
            # Check for exit
            if any(word in user_text.lower() for word in exit_words):
                goodbye = "Au revoir. Prenez soin de vous et n'hésitez pas à rappeler."
                print(f"🤖 Assistant: {goodbye}")
                voice.speak(goodbye)
                break
            
            # Process through CrewAI
            print("\n🔄 Traitement par CrewAI...")
            response = process_question(crew_factory, channel_id, user_id, username, user_text)
            
            clean_response = clean_for_speech(response)
            print(f"\n🤖 Assistant: {clean_response[:300]}...")
            
            voice.speak(clean_response)
            time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            voice.speak("Une erreur s'est produite. Veuillez réessayer.")


def run():
    """
    Launch the chatbot. Use --voice flag for voice mode.
    """
    parser = argparse.ArgumentParser(description="Assistant d'urgence médical CrewAI")
    parser.add_argument('--voice', '-v', action='store_true', 
                        help='Activer le mode vocal (GPT-Realtime)')
    parser.add_argument('--voice-type', '-vt', type=str, default='nova',
                        choices=['shimmer', 'nova', 'alloy', 'echo', 'fable', 'onyx'],
                        help='Choisir la voix: shimmer (femme), nova (femme pro), alloy (neutre), echo (homme), fable (homme UK), onyx (homme grave)')
    
    # Parse only known args to avoid conflicts with other entry points
    args, _ = parser.parse_known_args()
    
    if args.voice:
        run_voice_mode(voice_type=args.voice_type)
    else:
        run_text_mode()


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


# Direct entry point for voice command
def run_voice_entry():
    """Direct entry point for voice mode."""
    run_voice_mode()


if __name__ == "__main__":
    run()
