import torch
import clip
import json
import os
from PIL import Image
import numpy as np
from pathlib import Path

class EmergencyImageRetriever:
    def __init__(self, metadata_path=None, embeddings_path=None):
        """Initialize CLIP model and load image metadata"""
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        
        if metadata_path is None:
            metadata_path = current_dir / "image_metadata.json"
        if embeddings_path is None:
            embeddings_path = current_dir / "image_embeddings.npz"
            
        print("🔧 Chargement du modèle CLIP...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        print("📚 Chargement des métadonnées images...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)['images']
        
        # Update image paths to be relative to current_dir
        for img in self.metadata:
            if not os.path.isabs(img['filename']):
                img['filename'] = str(current_dir / img['filename'])
        
        self.embeddings_path = str(embeddings_path)
        self.image_embeddings = None
        self.valid_indices = []  # Track which metadata indices have valid embeddings
        
        # Try to load pre-computed embeddings
        if os.path.exists(self.embeddings_path):
            print("✅ Chargement des embeddings pré-calculés...")
            self._load_embeddings()
        else:
            print("🎨 Calcul des embeddings images (première utilisation)...")
            self._compute_embeddings()
    
    def _compute_embeddings(self):
        """Compute CLIP embeddings for all images in metadata"""
        embeddings_list = []
        self.valid_indices = []  # Track which metadata entries have valid embeddings
        
        for idx, img_data in enumerate(self.metadata):
            img_path = img_data['filename']
            
            if not os.path.exists(img_path):
                print(f"⚠️ Attention : {img_path} non trouvé, ignoré...")
                continue
            
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Compute embedding
                with torch.no_grad():
                    embedding = self.model.encode_image(image_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
                
                embeddings_list.append(embedding.cpu().numpy())
                self.valid_indices.append(idx)  # Track this as a valid embedding
                print(f"✅ Embedding {len(embeddings_list)}/{len(self.metadata)}: {os.path.basename(img_path)}")
                
            except Exception as e:
                print(f"❌ Erreur embedding {img_path}: {e}")
        
        # Stack all embeddings
        self.image_embeddings = np.vstack(embeddings_list)
        
        # Save embeddings and valid indices
        np.savez(self.embeddings_path, embeddings=self.image_embeddings, valid_indices=np.array(self.valid_indices))
        print(f"💾 Embeddings sauvegardés dans {self.embeddings_path} ({len(self.valid_indices)} images)")
    
    def _load_embeddings(self):
        """Load pre-computed embeddings from disk"""
        data = np.load(self.embeddings_path)
        self.image_embeddings = data['embeddings']
        
        # Load valid indices if available, otherwise assume all are valid (backwards compatibility)
        if 'valid_indices' in data:
            self.valid_indices = data['valid_indices'].tolist()
        else:
            # Old format: assume all metadata has embeddings (may cause issues)
            self.valid_indices = list(range(len(self.image_embeddings)))
        
        print(f"✅ {len(self.image_embeddings)} embeddings chargés")
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve top-k most relevant images for a query (French or English)
        
        Args:
            query (str): Natural language emergency description
            top_k (int): Number of top results to return
            
        Returns:
            list: Top-k results with metadata and similarity scores
        """
        # Enhance query with metadata for better matching
        enhanced_query = f"{query}"
        
        # Encode query text
        text_tokens = clip.tokenize([enhanced_query]).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_text(text_tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        query_embedding = query_embedding.cpu().numpy()
        
        # Keywords for boosting (French + English)
        keywords_to_boost = [
            # French keywords
            'enceinte', 'grossesse', 'bébé', 'nourrisson', 'enfant', 'adulte',
            'étouffement', 'rcp', 'heimlich', 'massage cardiaque', 'compressions',
            'pls', 'position latérale', 'inconscient', 'respiration', 'comment faire',
            'technique', 'étapes', 'arrêt cardiaque', 'réanimation',
            # English keywords  
            'pregnant', 'pregnancy', 'baby', 'infant', 'child', 'adult',
            'choking', 'cpr', 'heimlich', 'chest compressions', 'back blows',
            'recovery', 'unconscious', 'breathing', 'how to', 'performing',
            'technique', 'steps', 'do cpr', 'cardiac arrest', 'resuscitation'
        ]
        
        query_lower = query.lower()
        
        # Compute cosine similarities
        similarities = (self.image_embeddings @ query_embedding.T).squeeze()
        
        # Apply keyword boosting - iterate over VALID indices only
        boosted_similarities = similarities.copy()
        for emb_idx, metadata_idx in enumerate(self.valid_indices):
            img_meta = self.metadata[metadata_idx]
            boost_factor = 0.0
            caption_lower = img_meta['caption'].lower()
            keywords_lower = [k.lower() for k in img_meta['keywords']]
            subcategory_lower = img_meta['subcategory'].lower()
            
            # CRITICAL: Age-specific matching - prioritize exact age group
            # Baby/infant queries should ONLY match infant images (FR + EN)
            if any(term in query_lower for term in ['baby', 'infant', 'newborn', 'bébé', 'nourrisson', 'nouveau-né']):
                if subcategory_lower in ['infant', 'nourrisson'] or any(k in keywords_lower for k in ['infant', 'baby', 'nourrisson', 'bébé']):
                    boost_factor += 0.5  # Strong boost for infant match
                elif subcategory_lower in ['adult', 'adulte']:
                    boost_factor -= 0.8  # Strong penalty for adult when query is about baby
            
            # Adult queries should prioritize adult images (FR + EN)
            elif any(term in query_lower for term in ['adult', 'adulte']):
                if subcategory_lower in ['adult', 'adulte']:
                    boost_factor += 0.5
                elif subcategory_lower in ['infant', 'nourrisson']:
                    boost_factor -= 0.8
            
            # Pregnant woman queries (FR + EN)
            if any(term in query_lower for term in ['pregnant', 'enceinte', 'grossesse', 'pregnancy']):
                if any(k in keywords_lower for k in ['pregnant', 'enceinte', 'grossesse']):
                    boost_factor += 0.6
            
            # Special boost for CPR queries (FR + EN) - ONLY if explicitly about CPR/not breathing
            cpr_terms = ['cpr', 'rcp', 'cardiac arrest', 'arrêt cardiaque', 'not breathing', 'ne respire pas', 'ne respire plus', 'heart stopped', 'massage cardiaque', 'compressions', 'réanimation', 'resuscitation']
            if any(term in query_lower for term in cpr_terms):
                if img_meta['category'].lower() in ['cpr', 'rcp']:
                    boost_factor += 0.5  # Strong boost for CPR category
            else:
                # If query is NOT about CPR, penalize CPR images
                if img_meta['category'].lower() in ['cpr', 'rcp']:
                    boost_factor -= 0.3
            
            # Special boost for choking queries (FR + EN)
            if any(term in query_lower for term in ['choking', 'étouffement', 'étouffe', "s'étouffe", 'heimlich']):
                if 'étouffement' in img_meta['category'].lower() or 'choking' in img_meta['category'].lower():
                    boost_factor += 0.3
            
            # Special boost for PLS / Recovery position (FR + EN)
            if any(term in query_lower for term in ['pls', 'position latérale', 'recovery position', 'inconscient respire', 'unconscious breathing']):
                if 'pls' in img_meta['category'].lower() or 'recovery' in img_meta['category'].lower() or 'latérale' in img_meta['category'].lower():
                    boost_factor += 0.3
            
            # Special boost for unconscious person queries (FR + EN) - should return assessment/primary survey image
            # ONLY when they don't mention breathing status or CPR
            unconscious_terms = ['unconscious', 'inconscient', 'inconscience', 'évanoui', 'ne répond pas', 'not responding', 'unresponsive']
            cpr_breathing_terms = ['not breathing', 'ne respire pas', 'ne respire plus', 'cpr', 'rcp', 'massage cardiaque', 'cardiac arrest', 'arrêt cardiaque']
            
            if any(term in query_lower for term in unconscious_terms):
                # If query is ONLY about unconsciousness (no CPR/breathing mentioned)
                if not any(term in query_lower for term in cpr_breathing_terms):
                    if any(k in keywords_lower for k in ['unconscious', 'inconscient', 'primary survey', 'bilan primaire', 'assessment', 'airway', 'voies aériennes']):
                        boost_factor += 0.8  # Very strong boost for assessment image
                    if 'évaluation' in img_meta['category'].lower() or 'assessment' in img_meta['category'].lower():
                        boost_factor += 0.5
                    # Also check if filename contains "unconsciousness"
                    if 'unconsciousness' in img_meta['filename'].lower():
                        boost_factor += 0.6
            
            # Check for exact keyword matches
            for keyword in keywords_to_boost:
                if keyword in query_lower:
                    # Boost if keyword appears in caption or metadata keywords
                    if keyword in caption_lower or any(keyword in k for k in keywords_lower):
                        boost_factor += 0.15
            
            # Extra boost for subcategory match
            if subcategory_lower in query_lower:
                boost_factor += 0.2
                
            boosted_similarities[emb_idx] += boost_factor
        
        # Get top-k indices (these are embedding indices, need to map back to metadata)
        top_emb_indices = np.argsort(boosted_similarities)[::-1][:top_k]
        
        # Prepare results - map embedding indices back to metadata indices
        results = []
        for emb_idx in top_emb_indices:
            metadata_idx = self.valid_indices[emb_idx]
            results.append({
                'filename': self.metadata[metadata_idx]['filename'],
                'category': self.metadata[metadata_idx]['category'],
                'subcategory': self.metadata[metadata_idx]['subcategory'],
                'caption': self.metadata[metadata_idx]['caption'],
                'keywords': self.metadata[metadata_idx]['keywords'],
                'similarity': float(boosted_similarities[emb_idx])
            })
        
        return results
    
    def search_by_category(self, category):
        """Get all images in a specific category"""
        return [img for img in self.metadata if img['category'].lower() == category.lower()]


if __name__ == "__main__":
    # Test the retriever
    retriever = EmergencyImageRetriever()
    
    # Test queries
    test_queries = [
        "pregnant woman choking emergency",
        "baby not breathing CPR",
        "adult cardiac arrest chest compressions",
        "unconscious person breathing recovery position",
        "infant choking back blows"
    ]
    
    print("\n" + "="*60)
    print("🔍 Testing Emergency Image Retrieval")
    print("="*60 + "\n")
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = retriever.retrieve(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['filename']}")
            print(f"     Category: {result['category']} - {result['subcategory']}")
            print(f"     Caption: {result['caption']}")
            print(f"     Similarity: {result['similarity']:.3f}")
