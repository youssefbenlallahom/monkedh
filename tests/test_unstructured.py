"""
Test d'extraction PDF avec Unstructured - Optimisé pour RAG
Avantages: Conçu pour RAG/LLM, chunking intelligent, détection de structure
Gère: Texte, titres, listes, tableaux, métadonnées riches pour RAG
"""

from unstructured.partition.pdf import partition_pdf
import pathlib
import json
import time
import re

def clean_text_for_rag(text: str) -> str:
    """
    Nettoie le texte extrait en supprimant les éléments inutiles pour le RAG
    
    Args:
        text: Texte brut extrait
    
    Returns:
        Texte nettoyé
    """
    # 1. Remplacer les codes CID par des bullet points
    text = re.sub(r'\(cid:\d+\)', '•', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Patterns à supprimer (en-têtes, pieds de page, répétitions)
    patterns_to_remove = [
        r'^page\s+\d+\s*$',  # "page 2", "page 10"
        r'^page\s*\|\s*$',  # "page |"
        r'^Formation aux premiers [Ss]ecours\s*$',  # Répétitions du titre
        r'^FORMATION AUX PREMIERS SECOURS\s*$',
        r'^Formation aux premiers secours\s+\d{2}/\d{2}/\d{4}\s*$',  # "Formation aux premiers secours 01/03/2006"
        r'^\d{2}/\d{2}/\d{4}\s*$',  # Dates "01/03/2006"
        r'^Centre d\'Enseignement des Soins d\'Urgence\s*(?:CESU\s*\d+)?\s*$',  # "Centre d'Enseignement des Soins d'Urgence CESU 03"
        r'^Centre d\'Enseignement des Soins d\'Urgence CESU\s*\d+\s*$',  # "Centre d'Enseignement des Soins d'Urgence CESU 03"
        r'^Service d\'Aide Médicale Urgente du Centre Est\s*(?:SAMU\s*\d+)?\s*$',  # Pied de page SAMU
        r'^Centre d\'Enseignement des Soins d\'Urgence CESU\s+\d+\s+page\s+\d+\s*$',  # "Centre d'Enseignement des Soins d'Urgence CESU 03 page 49"
        r'^CESU\s+\d+\s*$',  # "CESU 03"
    ]
    
    for line in lines:
        line_stripped = line.strip()
        
        # Ignorer les lignes vides
        if not line_stripped:
            cleaned_lines.append('')
            continue
        
        # Vérifier si la ligne correspond à un pattern à supprimer
        should_remove = False
        for pattern in patterns_to_remove:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                should_remove = True
                break
        
        if not should_remove:
            cleaned_lines.append(line)
    
    # Rejoindre et supprimer les lignes vides multiples
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remplacer les multiples sauts de ligne par maximum 2
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text

def extract_with_unstructured(pdf_path: str, output_path: str):
    """
    Extrait le contenu du PDF avec Unstructured (optimisé pour RAG)
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        output_path: Chemin de sortie pour le fichier markdown
    """
    print("=" * 60)
    print("Extraction avec Unstructured - Optimisé pour RAG")
    print("=" * 60)
    
    start_time = time.time()
    
    print("Partition du document (peut prendre du temps)...")
    
    # Extraction avec partition - stratégie FAST sans OCR
    # Pour PDFs normaux avec texte natif (pas de scan)
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",  # "fast" = extraction directe SANS OCR
        infer_table_structure=True,  # Détection intelligente des tableaux
        extract_images_in_pdf=False,  # Désactivé - pas d'extraction d'images
    )
    
    print(f"✓ {len(elements)} éléments extraits\n")
    
    text_content = []
    text_content.append(f"# Document PDF - {len(elements)} éléments détectés\n\n")
    
    # Compteurs par type d'élément
    element_types = {}
    
    for i, element in enumerate(elements):
        element_type = type(element).__name__
        element_types[element_type] = element_types.get(element_type, 0) + 1
        
        # Formater selon le type d'élément
        if "Title" in element_type:
            text_content.append(f"## {element.text}\n\n")
        elif "Table" in element_type:
            # Traiter les tableaux extraits en texte
            text_content.append(f"**[TABLEAU - Texte]**\n\n{element.text}\n\n")
        elif "List" in element_type:
            text_content.append(f"{element.text}\n\n")
        elif "Image" in element_type:
            # Ignorer les éléments Image - pas d'extraction ni de description
            pass
        else:
            text_content.append(f"{element.text}\n\n")
    
    full_text = "".join(text_content)
    
    # Nettoyer le texte pour le RAG
    print("\nNettoyage du texte (suppression en-têtes/pieds de page)...")
    full_text_cleaned = clean_text_for_rag(full_text)
    
    # Sauvegarde du texte nettoyé
    pathlib.Path(output_path).write_text(full_text_cleaned, encoding='utf-8')
    
    elapsed_time = time.time() - start_time
    
    # Statistiques
    stats = {
        "tool": "Unstructured",
        "strategy": "fast",
        "file_size_bytes": len(full_text_cleaned.encode()),
        "char_count": len(full_text_cleaned),
        "total_elements": len(elements),
        "element_types": element_types,
        "extraction_time_seconds": round(elapsed_time, 2),
    }
    
    print(f"\n✓ Extraction terminée en {elapsed_time:.2f}s")
    print(f"✓ Éléments extraits: {stats['total_elements']}")
    print(f"✓ Types d'éléments:")
    for elem_type, count in element_types.items():
        print(f"  - {elem_type}: {count}")
    print(f"✓ Taille du fichier: {stats['file_size_bytes']:,} bytes")
    print(f"\nFichier sauvegardé: {output_path}")
    
    return stats

if __name__ == "__main__":
    # Chemin relatif au script (pas au dossier d'exécution)
    script_dir = pathlib.Path(__file__).parent
    pdf_file = script_dir / "FPS.pdf"
    output_file = script_dir / "output_unstructured.md"
    
    print("\n" + "=" * 60)
    print("TEST EXTRACTION PDF AVEC UNSTRUCTURED")
    print("=" * 60)
    print("\nConfiguration:")
    print("  - Stratégie: fast (extraction directe SANS OCR)")
    print("  - Extraction images: Désactivée")
    print("  - Structure tableaux: Activée")
    print("  - PDF: Texte natif (pas de scan)")
    print("\n")
    
    try:
        stats = extract_with_unstructured(str(pdf_file), str(output_file))
        print("\n" + "=" * 60)
        print("✓ EXTRACTION RÉUSSIE")
        print("=" * 60)
        print(f"\nRésumé:")
        print(f"  - Temps: {stats['extraction_time_seconds']}s")
        print(f"  - Éléments: {stats['total_elements']}")
        print(f"  - Taille: {stats['file_size_bytes']:,} bytes")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
