from crewai import Agent, Task, Crew
from crewai.tools import tool
from .clip_retriever import EmergencyImageRetriever
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize CLIP retriever (singleton for efficiency)
_retriever = None

def get_retriever():
    """Get or initialize the CLIP retriever singleton"""
    global _retriever
    if _retriever is None:
        _retriever = EmergencyImageRetriever()
    return _retriever


@tool("Search Emergency Image Database")
def search_emergency_image(query: str) -> str:
    """
    Search for the most relevant emergency medical instruction image based on the query.
    Use this tool to find visual guidance for emergency situations like CPR, choking, 
    recovery position, etc. The tool uses CLIP-based semantic search to find the best 
    matching image from a curated medical emergency database.
    
    Args:
        query: A description of the emergency situation (e.g., "baby choking", 
               "adult CPR", "pregnant woman choking", "unconscious person breathing")
    
    Returns:
        Information about the best matching image including filename, category, 
        description, and relevance score.
        
    IMPORTANT: When using this image in your response, you MUST:
    1. Include the EXACT image path
    2. DESCRIBE what the image shows using the "Description" field
    3. Only use the image if it matches the situation (relevance > 50%)
    """
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=1)
    
    if results:
        best_match = results[0]
        relevance_pct = best_match['similarity'] * 100
        
        # Warn if relevance is low
        relevance_warning = ""
        if relevance_pct < 50:
            relevance_warning = """
⚠️ LOW RELEVANCE WARNING: This image may not match the situation well.
   Consider NOT including this image in your response, or search with different keywords.
"""
        
        return f"""
═══════════════════════════════════════════════════════════════════
📷 IMAGE TROUVÉE
═══════════════════════════════════════════════════════════════════
{relevance_warning}
📁 CHEMIN IMAGE (à copier tel quel) :
{best_match['filename']}

📂 CATÉGORIE : {best_match['category']} - {best_match['subcategory']}

📝 CE QUE MONTRE L'IMAGE (à décrire dans ta réponse) :
{best_match['caption']}

🏷️ MOTS-CLÉS : {', '.join(best_match['keywords'][:8])}

📊 PERTINENCE : {relevance_pct:.0f}%

═══════════════════════════════════════════════════════════════════
⚠️ INSTRUCTIONS OBLIGATOIRES :
═══════════════════════════════════════════════════════════════════
Quand tu utilises cette image dans ta réponse :
1. COPIE le chemin EXACT ci-dessus
2. DÉCRIS ce que montre l'image en utilisant la description ci-dessus
3. Si pertinence < 50%, NE PAS utiliser l'image

EXEMPLE DE FORMAT À UTILISER :
📷 GUIDE VISUEL : [chemin image]
   Cette image montre : [description de ce que montre l'image]
═══════════════════════════════════════════════════════════════════
"""
    return "❌ Aucune image correspondante trouvée. Ne mentionne pas d'image dans ta réponse."


@tool("Browse Emergency Categories")
def browse_emergency_categories(category: str) -> str:
    """
    Browse all images available in a specific emergency category.
    Use this tool when you want to see all available images for a category
    rather than searching for a specific situation.
    
    Args:
        category: The category to browse. Available categories are:
                  - "CPR" (includes adult and infant)
                  - "Choking" (includes adult, child, infant, pregnant)
                  - "Recovery Position"
                  - "Log Roll" (turning casualty face up)
    
    Returns:
        A list of all images in the specified category with their descriptions.
    """
    retriever = get_retriever()
    
    # Map user-friendly names to actual category names
    category_map = {
        'cpr': 'CPR',
        'choking': 'First aid for choking',
        'recovery': 'Recovery Position',
        'recovery position': 'Recovery Position',
        'log roll': 'How to turn a casualty face up',
        'turn casualty': 'How to turn a casualty face up'
    }
    
    mapped_category = category_map.get(category.lower(), category)
    images = retriever.search_by_category(mapped_category)
    
    if not images:
        # Try partial match
        for img in retriever.metadata:
            if category.lower() in img['category'].lower():
                images.append(img)
    
    if images:
        result = f"📁 Found {len(images)} images in '{mapped_category}':\n\n"
        for i, img in enumerate(images, 1):
            result += f"{i}. {img['filename']}\n"
            result += f"   Subcategory: {img['subcategory']}\n"
            result += f"   Description: {img['caption']}\n\n"
        return result
    
    return f"No images found in category '{category}'. Available categories: CPR, Choking, Recovery Position, Log Roll"


class EmergencyResponseAgent:
    def __init__(self):
        # Define the tools available to the agent
        self.tools = [search_emergency_image, browse_emergency_categories]
        
        self.agent = Agent(
            role="Emergency Medical Response Coordinator",
            goal="Analyze emergency situations, find the most relevant visual instructions using the image search tool, and provide clear step-by-step guidance",
            backstory="""You are an expert emergency medical response coordinator with extensive 
            training in first aid, CPR, and emergency procedures. You have access to a database 
            of validated emergency medical instruction images. You ALWAYS use the search tool 
            to find the most relevant image before providing instructions. You excel at quickly 
            assessing emergency situations and providing clear, actionable instructions while 
            referencing visual aids. You always prioritize safety and accuracy in your guidance.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools  # Attach tools to the agent
        )
    
    def create_response_task(self, emergency_situation):
        """Create a task to respond to an emergency situation"""
        return Task(
            description=f"""
            Analyze this emergency situation and provide comprehensive guidance:
            
            SITUATION: {emergency_situation}
            
            INSTRUCTIONS:
            1. FIRST, use the "Search Emergency Image Database" tool to find the most relevant 
               instructional image for this emergency situation. Pass a clear query describing 
               the emergency (e.g., "baby choking", "adult CPR", "pregnant woman choking").
            
            2. Based on the image found, provide your response with:
               - A brief assessment of the situation (2-3 sentences)
               - Reference the image found and explain it will help guide the procedure
               - Clear, numbered step-by-step instructions (minimum 4 steps)
               - Critical safety warnings if applicable
               - When to call emergency services (911/112)
            
            Format your response clearly with sections:
            - ASSESSMENT
            - VISUAL GUIDE (reference the image found)
            - STEP-BY-STEP INSTRUCTIONS
            - CRITICAL WARNINGS (if any)
            - WHEN TO CALL EMERGENCY SERVICES
            """,
            agent=self.agent,
            expected_output="A comprehensive emergency response that references the found image, includes assessment, detailed step-by-step instructions, and safety information"
        )
    
    def run(self, emergency_situation):
        """Execute the emergency response workflow"""
        task = self.create_response_task(emergency_situation)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        return result


if __name__ == "__main__":
    agent = EmergencyResponseAgent()
    
    # Test with a sample query
    print("\n" + "="*60)
    print("🚨 Testing Emergency Response Agent with Tools")
    print("="*60 + "\n")
    
    result = agent.run("A baby is choking and turning blue")
    print("\n" + "="*60)
    print("📋 FINAL RESPONSE:")
    print("="*60)
    print(result)

