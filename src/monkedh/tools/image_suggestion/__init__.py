"""
Module de suggestion d'images pour les urgences médicales.
Utilise CLIP pour la recherche sémantique d'images de premiers secours.
"""

from .emergency_agent import search_emergency_image, browse_emergency_categories

__all__ = ["search_emergency_image", "browse_emergency_categories"]
