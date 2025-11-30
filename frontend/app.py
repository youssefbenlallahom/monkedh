import streamlit as st
import requests
import time
from typing import List, Dict

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Dashboard SAMU - Notifications d'Urgence", page_icon="🚑", layout="wide")

st.title("🚑 Dashboard SAMU - Notifications d'Urgence en Temps Réel")

st.markdown("""
Ce dashboard affiche les notifications d'urgence détectées par le chatbot.
Les notifications sont mises à jour automatiquement toutes les 5 secondes.
""")

# Fonction pour récupérer les notifications depuis l'API
def get_notifications() -> List[Dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/notifications")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des notifications: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de contacter l'API: {e}")
        return []

# Fonction pour afficher une notification
def display_notification(notif: Dict, index: int):
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader(f"Notification #{index + 1}")
            st.write(f"**Niveau de Risque:** {notif.get('niveau_risque', 'N/A')}")
            st.write(f"**Contenu:** {notif.get('contenu', 'N/A')}")

            if notif.get('patient_nom'):
                st.write(f"**Patient:** {notif['patient_nom']}")
            if notif.get('patient_age'):
                st.write(f"**Âge:** {notif['patient_age']} ans")
            if notif.get('localisation'):
                st.write(f"**Localisation:** {notif['localisation']}")
            if notif.get('contact_urgence'):
                st.write(f"**Contact Urgence:** {notif['contact_urgence']}")
            if notif.get('details_supplementaires'):
                st.write(f"**Détails Supplémentaires:** {notif['details_supplementaires']}")

        with col2:
            # Bouton pour marquer comme traité
            if st.button(f"Marquer comme traité #{index + 1}", key=f"mark_{index}"):
                st.success("Notification marquée comme traitée !")
                # Ici, on pourrait ajouter une logique pour mettre à jour l'API

        with col3:
            # Indicateur de priorité basé sur le niveau de risque
            risk_level = notif.get('niveau_risque', '').lower()
            if 'critique' in risk_level or 'élevé' in risk_level:
                st.error("🔴 PRIORITÉ ÉLEVÉE")
            elif 'moyen' in risk_level:
                st.warning("🟡 PRIORITÉ MOYENNE")
            else:
                st.info("🟢 PRIORITÉ FAIBLE")

        st.divider()

# Zone principale
placeholder = st.empty()

# Boucle pour mise à jour en temps réel
while True:
    with placeholder.container():
        st.subheader("Notifications Récentes")

        notifications = get_notifications()

        if notifications:
            for i, notif in enumerate(reversed(notifications)):  # Afficher les plus récentes en premier
                display_notification(notif, len(notifications) - 1 - i)
        else:
            st.info("Aucune notification pour le moment.")

        st.caption("Dernière mise à jour: " + time.strftime("%H:%M:%S"))

    time.sleep(5)  # Mise à jour toutes les 5 secondes
    st.rerun()  # Redessiner l'interface