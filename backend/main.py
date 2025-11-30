from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Monkedh Backend", description="API pour les notifications d'urgence SAMU")

class Notification(BaseModel):
    niveau_risque: str  # Ex: "Faible", "Moyen", "Élevé", "Critique"
    contenu: str  # Description de l'urgence
    patient_nom: Optional[str] = None
    patient_age: Optional[int] = None
    localisation: Optional[str] = None
    contact_urgence: Optional[str] = None
    details_supplementaires: Optional[str] = None

notifications = []  # Stockage temporaire en mémoire pour les tests

@app.post("/notifications", response_model=dict)
async def creer_notification(notification: Notification):
    """
    Créer une nouvelle notification d'urgence pour le SAMU.
    """
    # Pour les tests, on stocke simplement en mémoire
    notifications.append(notification.model_dump())
    return {"message": "Notification créée avec succès", "id": len(notifications)}

@app.get("/notifications")
async def lister_notifications():
    """
    Lister toutes les notifications (pour les tests).
    """
    return notifications

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)