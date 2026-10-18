import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import torch

def process_knowledge_base(data_dir: str):
    """
    Process a PDF knowledge base by loading it, splitting it into chunks, and returning the chunks.

    Args:
        data_dir (str): The path to the directory containing the PDF files.

    Returns:
        list: A list of document chunks.
    """
    all_chunks = []
    
    if not os.path.exists(data_dir):
        print(f"Erreur : Le dossier '{data_dir}' n'existe pas.")
        print("Veuillez créer un dossier 'data/raw' et y glisser quelques PDF.")
        return []

    # Parcourir tous les fichiers du dossier
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not files:
        print(f"Aucun fichier PDF trouvé dans '{data_dir}'.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # chunk_size = nombre de caractères max par chunk (~200-300 mots)
        chunk_overlap=200,  # chunk_overlap = chevauchement pour ne pas couper le contexte entre deux chunks
        separators=["\n\n", "\n", " ", ""]
    )

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        print(f"Traitement en cours : {filename}")
        
        try:
            # 1. Extraction du texte page par page via PyMuPDF
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # 2. Enrichissement des métadonnées pour chaque page
            for doc in documents:
                doc.metadata["source_file"] = filename
                
            # 3. Chunking (Découpage sémantique par blocs)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"   -> {len(chunks)} chunks générés pour {filename}")
            
        except Exception as e:
            print(f"   ❌ Erreur lors du traitement de {filename}: {e}")

    print(f"\n✨ Ingestion terminée avec succès ! Total de chunks prêts : {len(all_chunks)}")
    return all_chunks





if __name__ == "__main__":
    # Dossier cible contenant vos PDF de Master IA
    DATA_DIR = "../data/raw"
    
    chunks = process_knowledge_base(DATA_DIR)
    vector_store = build_vector_store(chunks)
    # Affichage d'un exemple pour vérifier le résultat
    print(vector_store)
    if chunks:
        print("\n" + "="*40)
        print("Exemple du premier chunk généré :")
        print("="*40)
        print(f"Métadonnées : {chunks[0].metadata}")
        print(f"Contenu :\n{chunks[0].page_content[:400]}...")