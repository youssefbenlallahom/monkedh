# 🚑 Monkedh - AI-Powered Medical Emergency Assistant

An intelligent medical emergency chatbot designed for the **Tunisian healthcare context**, powered by [CrewAI](https://github.com/joaomdmoura/crewAI). This system provides real-time emergency guidance, first aid instructions with visual aids, and automatic SAMU (emergency services) notification.

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Features

- **🤖 Multi-Agent AI System**: Two specialized agents working together
  - **Emergency Guide Agent**: Detects emergencies, provides first aid guidance
  - **SAMU Notifier Agent**: Silently sends structured alerts to emergency services

- **📚 RAG-Based Knowledge**: Semantic search through official first aid manuals (French)

- **🖼️ Visual Guidance**: CLIP-powered image search for emergency procedures (CPR, choking, recovery position)

- **💬 Conversation Memory**: Redis-based memory for contextual conversations

- **🔔 Real-Time Dashboard**: Streamlit frontend for SAMU operators to monitor alerts

- **🇹🇳 Tunisia-Specific**: Knows local emergency numbers (190, 198, 197) and healthcare resources

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (CLI/API)                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CrewAI Multi-Agent System                    │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │  guideur_urgence_samu   │───▶│     notificateur_samu       │ │
│  │  (Emergency Guide)      │    │   (SAMU Notifier - Silent)  │ │
│  │                         │    │                             │ │
│  │  Tools:                 │    │  Tools:                     │ │
│  │  • RAG Search           │    │  • SAMU Notification API    │ │
│  │  • Web Search (Serper)  │    │                             │ │
│  │  • Web Scraper          │    │                             │ │
│  │  • Image Search (CLIP)  │    │                             │ │
│  └─────────────────────────┘    └──────────────┬──────────────┘ │
└────────────────────────────────────────────────┼────────────────┘
                                                 │
              ┌──────────────────────────────────┼──────────────────┐
              │                                  │                  │
              ▼                                  ▼                  ▼
    ┌─────────────────┐              ┌─────────────────┐   ┌───────────────┐
    │  Qdrant Cloud   │              │  FastAPI Backend│   │ Redis Cloud   │
    │  (Vector DB)    │              │  (Port 8000)    │   │ (Memory)      │
    │                 │              └────────┬────────┘   └───────────────┘
    │  First Aid      │                       │
    │  Manual Chunks  │                       ▼
    └─────────────────┘              ┌─────────────────┐
                                     │  Streamlit UI   │
                                     │  (Port 8501)    │
                                     │  SAMU Dashboard │
                                     └─────────────────┘
```

---

## 📁 Project Structure

```
monkedh/
├── backend/
│   └── main.py                 # FastAPI server for SAMU notifications
├── frontend/
│   └── app.py                  # Streamlit dashboard for SAMU operators
├── src/monkedh/
│   ├── main.py                 # CLI entry point
│   ├── crew.py                 # CrewAI agents & tasks configuration
│   ├── config/
│   │   ├── agents.yaml         # Agent definitions
│   │   └── tasks.yaml          # Task definitions
│   └── tools/
│       ├── redis_storage.py    # Conversation memory
│       ├── samu_notification_tool.py  # SAMU alert tool
│       ├── rag/                # RAG tool for first aid manual
│       │   ├── rag_tool.py
│       │   ├── vectorize.py
│       │   └── config.py
│       └── image_suggestion/   # CLIP-based image search
│           ├── clip_retriever.py
│           ├── image_metadata.json
│           └── emergency_image_db/  # First aid images
├── knowledge/
│   └── user_preference.txt     # User preferences
├── pyproject.toml              # Project configuration
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (create this)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 - 3.12**
- **Ollama** (for local embeddings) - [Install Ollama](https://ollama.ai)
- **Azure OpenAI API** credentials (or modify to use other LLMs)

### 1. Clone & Setup Environment

```powershell
# Clone the repository
git clone https://github.com/youssefbenlallahom/monkedh.git
cd monkedh

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_openai_api_key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-12-01-preview
model=azure/your-deployment-name

# Redis (pre-configured cloud instance included)
REDIS_HOST=redis-13350.c339.eu-west-3-1.ec2.redns.redis-cloud.com
REDIS_PORT=13350
REDIS_PASSWORD=YoLErdUztvwgDQvhAr1Fgbp0NUdekrRm

# Optional: Serper API for web search
SERPER_API_KEY=your_serper_api_key
```

### 3. Start Ollama (for embeddings)

```powershell
# Pull the embedding model
ollama pull embeddinggemma:latest

# Ollama should be running on http://localhost:11434
```

### 4. Run the Application

#### Option A: CLI Chatbot Only

```powershell
python -m monkedh.main
```

#### Option B: Full Stack (Backend + Frontend + CLI)

**Terminal 1 - Backend API:**
```powershell
python backend/main.py
# Runs on http://localhost:8000
```

**Terminal 2 - Frontend Dashboard:**
```powershell
streamlit run frontend/app.py
# Runs on http://localhost:8501
```

**Terminal 3 - Chatbot:**
```powershell
python -m monkedh.main
```

---

## 💬 Usage Examples

### Emergency Scenario

```
Votre question : Un homme de 50 ans s'est effondré dans la rue, il ne respire plus !

🚨 ALERTE : C'est une urgence vitale probable.

J'ai transmis votre alerte au SAMU (190). Ils sont prévenus.

MAINTENANT, ÉCOUTEZ-MOI BIEN :
1. Placez la victime sur le dos sur une surface dure
2. Basculez sa tête en arrière, soulevez son menton
3. Commencez les compressions thoraciques : 30 compressions, 2 insufflations

📷 GUIDE VISUEL : [Image CPR adulte]

Est-ce que la victime a repris sa respiration ?
```

### Information Query

```
Votre question : Comment mettre quelqu'un en PLS ?

Voici les étapes pour la Position Latérale de Sécurité (PLS) :

1. Placez le bras de la victime le plus proche de vous à angle droit
2. Amenez l'autre bras sur la poitrine, main contre la joue
3. Pliez le genou le plus éloigné
4. Faites rouler la victime vers vous
5. Ajustez la jambe pour stabiliser la position

📷 GUIDE VISUEL : [Image PLS]

En Tunisie, si la victime ne reprend pas conscience, appelez le 190 (SAMU).
```

---

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| **RAG Search** | Semantic search in first aid manual (Qdrant + Ollama) |
| **Serper Search** | Web search for local healthcare info |
| **Web Scraper** | Extract info from healthcare websites |
| **Image Search** | CLIP-powered emergency image retrieval |
| **SAMU Notification** | Send structured alerts to backend API |

---

## 📊 API Endpoints

### Backend (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/notifications` | Create emergency notification |
| `GET` | `/notifications` | List all notifications |

### Notification Schema

```json
{
  "niveau_risque": "Critique",
  "contenu": "Arrêt cardiaque, homme 50 ans",
  "patient_nom": "Inconnu",
  "patient_age": 50,
  "localisation": "Rue principale, Tunis",
  "contact_urgence": "+216 XX XXX XXX",
  "details_supplementaires": "RCP en cours par témoin"
}
```

---

## 🇹🇳 Tunisia Emergency Numbers

| Service | Number |
|---------|--------|
| **SAMU** (Medical Emergency) | 190 |
| **Protection Civile** (Fire/Accidents) | 198 |
| **Police Secours** | 197 |
| **Centre Antipoison** (Tunis) | 71 335 500 |

---

## 🧪 Testing

```powershell
# Run RAG tool test
.\run_test.ps1

# Or manually
python tests/test_rag_tool.py
```

---

## 🔧 Configuration

### Changing the LLM

Edit `src/monkedh/crew.py`:

```python
# For OpenAI
llm = LLM(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# For local Ollama
llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
)
```

### Adding New Emergency Images

1. Add images to `src/monkedh/tools/image_suggestion/emergency_image_db/`
2. Update `image_metadata.json` with image details
3. Delete `image_embeddings.npz` to regenerate embeddings

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 👨‍💻 Author

**Youssef Benlallahom**

---

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent framework
- [OpenAI CLIP](https://github.com/openai/CLIP) - Image-text matching
- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM hosting
