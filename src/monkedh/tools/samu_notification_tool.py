from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import requests

class SAMUNotificationToolInput(BaseModel):
    """Schéma d'entrée pour l'outil de notification SAMU."""
    niveau_risque: str = Field(..., description="Niveau de risque de l'urgence (Faible, Moyen, Élevé, Critique)")
    contenu: str = Field(..., description="Description détaillée de l'urgence détectée")
    patient_nom: Optional[str] = Field(None, description="Nom du patient si connu")
    patient_age: Optional[int] = Field(None, description="Âge du patient si connu")
    localisation: Optional[str] = Field(None, description="Localisation de l'urgence")
    contact_urgence: Optional[str] = Field(None, description="Contact d'urgence")
    details_supplementaires: Optional[str] = Field(None, description="Détails supplémentaires sur l'urgence")

class SAMUNotificationTool(BaseTool):
    name: str = "SAMU Notification Tool"
    description: str = (
        "Tool to create an emergency notification and send it to the SAMU service via the backend API. "
        "Use this tool when the chatbot detects a medical emergency requiring SAMU intervention. "
        "Input arguments: niveau_risque (str), contenu (str), patient_nom (str, optional), "
        "patient_age (int, optional), localisation (str, optional), contact_urgence (str, optional), "
        "details_supplementaires (str, optional)."
    )
    args_schema: Type[BaseModel] = SAMUNotificationToolInput


    def _run(self, niveau_risque: str, contenu: str, patient_nom: Optional[str] = None,
             patient_age: Optional[int] = None, localisation: Optional[str] = None,
             contact_urgence: Optional[str] = None, details_supplementaires: Optional[str] = None) -> str:
        print("\n🔧 [SAMUNotificationTool] Outil appelé !")
        print(f"[SAMUNotificationTool] Arguments reçus : niveau_risque={niveau_risque}, contenu={contenu}, patient_nom={patient_nom}, patient_age={patient_age}, localisation={localisation}, contact_urgence={contact_urgence}, details_supplementaires={details_supplementaires}")

        # Préparer les données de la notification
        notification_data = {
            "niveau_risque": niveau_risque,
            "contenu": contenu,
            "patient_nom": patient_nom,
            "patient_age": patient_age,
            "localisation": localisation,
            "contact_urgence": contact_urgence,
            "details_supplementaires": details_supplementaires
        }
        # Supprimer les champs None pour nettoyer le payload
        notification_data = {k: v for k, v in notification_data.items() if v is not None}
        print(f"[SAMUNotificationTool] Payload envoyé à l'API : {notification_data}")

        try:
            response = requests.post("http://localhost:8000/notifications", json=notification_data)
            print(f"[SAMUNotificationTool] Status code reçu : {response.status_code}")
            print(f"[SAMUNotificationTool] Réponse brute : {response.text}")
            if response.status_code == 200:
                result = response.json()
                print(f"[SAMUNotificationTool] Réponse JSON : {result}")
                return f"Notification SAMU créée avec succès. ID: {result.get('id', 'N/A')}. Message: {result.get('message', '')}"
            else:
                print(f"[SAMUNotificationTool] Erreur lors de la création : {response.text}")
                return f"Erreur lors de la création de la notification: Code {response.status_code}, Réponse: {response.text}"
        except requests.exceptions.RequestException as e:
            print(f"[SAMUNotificationTool] Exception : {str(e)}")
            return f"Erreur de connexion à l'API backend: {str(e)}"