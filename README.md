# University Knowledge Assistant

Assistant de questions-réponses sur les documents officiels d'Avignon Université (statuts, règlement intérieur, chartes, modalités de contrôle des connaissances…), basé sur une architecture **RAG** (*Retrieval-Augmented Generation*).

L'assistant répond **uniquement à partir des documents indexés**, cite ses sources (`[1]`, `[2]`… avec fichier et page) et indique quand l'information n'est pas disponible plutôt que d'inventer une réponse.

> Projet d'apprentissage : l'objectif est de construire un RAG pas à pas, d'abord dans une version simple, puis de l'améliorer avec des techniques de RAG avancé mesurées par une évaluation.

## Fonctionnement

```
                         INGESTION (une fois)
  PDF ──► extraction ──► chunking ──► embeddings ──► Qdrant
         (PyMuPDF)     (1000 car.,   (multilingual-  (vecteurs +
                        overlap 200)   e5-small)      texte + métadonnées)

                         QUESTION (à chaque requête)
  Question ──► embedding ──► recherche top-k ──► contexte numéroté ──► prompt ──► LLM ──► réponse + sources
               ("query: ")    (Qdrant, cosinus)   [1] (fichier, p.X)              (HF)
```

| Étape | Fichier | Fonction |
|---|---|---|
| Chargement et découpage des PDF | [src/pdf_loader.py](src/pdf_loader.py) | `process_knowledge_base` |
| Calcul des embeddings | [src/embedding.py](src/embedding.py) | `build_vector_store` |
| Stockage dans Qdrant | [src/embedding.py](src/embedding.py) | `store_embbedings_in_qdrant` |
| Recherche des passages pertinents | [src/embedding.py](src/embedding.py) | `search_in_qdrant` |
| Construction du contexte et du prompt | [src/rag.py](src/rag.py) | `build_context`, `build_messages` |
| Appel au LLM | [src/llm.py](src/llm.py) | `generate` |
| Pipeline complet | [src/rag.py](src/rag.py) | `answer` |
| Interface de chat | [app/streamlit_app.py](app/streamlit_app.py) | — |

## Stack technique

- **Extraction PDF** : PyMuPDF (via `langchain-community`)
- **Chunking** : `RecursiveCharacterTextSplitter` (`langchain-text-splitters`)
- **Embeddings** : [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (384 dimensions, multilingue, préfixes `passage:` / `query:`)
- **Base vectorielle** : [Qdrant](https://qdrant.tech/), distance cosinus
- **LLM** : modèle Hugging Face via les *Inference Providers* (`openai/gpt-oss-20b` par défaut), avec `ChatHuggingFace`
- **Interface** : Streamlit

## Structure du projet

```
.
├── app/
│   └── streamlit_app.py     # Interface de chat
├── data/
│   ├── raw/                 # PDF sources
│   └── processed/
├── evaluation/              # Jeu de test et scripts d'évaluation (à venir)
├── src/
│   ├── pdf_loader.py        # Chargement + chunking des PDF
│   ├── init_db.py           # Création de la collection Qdrant
│   ├── embedding.py         # Embeddings, stockage et recherche dans Qdrant
│   ├── rag.py               # Prompt, contexte et pipeline answer()
│   └── llm.py               # Point d'entrée unique vers le LLM
├── test_directory/          # Petit jeu de PDF pour les tests
├── tests/
├── test_qdrant.ipynb        # Notebook d'expérimentation (ingestion, recherche, RAG)
└── .env.example             # Modèle des variables d'environnement
```

## Installation

### 1. Environnement Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit qdrant-client pymupdf langchain langchain-community \
            langchain-text-splitters langchain-huggingface sentence-transformers \
            python-dotenv
```

> PyTorch est installé comme dépendance de `sentence-transformers`. Sur un GPU ancien (compute capability < 7.5), installer un build compatible, par exemple :
> `pip install torch --index-url https://download.pytorch.org/whl/cu126`. Le modèle d'embedding tourne par défaut sur CPU.

### 2. Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Le tableau de bord est disponible sur http://localhost:6333/dashboard.

### 3. Variables d'environnement

Copier `.env.example` en `.env` et le compléter :

| Variable | Rôle | Valeur par défaut |
|---|---|---|
| `LLM_PROVIDER` | Fournisseur du LLM utilisé par `generate()` | `hf` |
| `HUGGINGFACEHUB_API_TOKEN` | Token Hugging Face (permission *Make calls to Inference Providers*) | — |
| `HF_MODEL_ID` | Modèle appelé via les Inference Providers | `openai/gpt-oss-20b` |

Le fichier `.env` est exclu du dépôt par le `.gitignore`.

## Utilisation

### Indexer les documents

Placer les PDF dans `data/raw/`, puis, depuis la racine du projet (voir aussi `test_qdrant.ipynb`) :

```python
from src.pdf_loader import process_knowledge_base
from src.embedding import build_vector_store, store_embbedings_in_qdrant

chunks = process_knowledge_base("data/raw")
vectors = build_vector_store(chunks)
store_embbedings_in_qdrant(chunks, vectors)
```

> Après une modification du chunking ou du modèle d'embedding, supprimer la collection `knowledge_base` dans Qdrant avant de réindexer : une collection existante n'est pas recréée.

### Poser une question en Python

```python
from src.rag import answer

result = answer("Qui doit être le président d'une association étudiante ?", top_k=5)
print(result["answer"])
print(result["sources"])   # [{"number", "source_file", "page", "score"}, ...]
```

### Lancer l'interface

```bash
python -m streamlit run app/streamlit_app.py
```

Lancer avec `python -m` depuis la racine du projet, pour que le paquet `src` soit trouvé.

## Avancement

- [x] Définition du périmètre
- [x] Collecte des documents
- [x] Pipeline d'ingestion (extraction PDF)
- [x] Nettoyage des documents
- [x] Ajout des métadonnées (fichier source, page)
- [x] Chunking
- [x] Embeddings
- [x] Stockage dans Qdrant
- [x] Première version du RAG (recherche, prompt avec citations, génération, interface Streamlit)
- [ ] Mini-évaluation : jeu de questions de référence, Hit@k / MRR pour la recherche, notation des réponses
- [ ] RAG avancé (pistes : recherche hybride, reranking, réécriture de requête, prise en compte de l'historique)

## Limites actuelles

- L'assistant ne tient pas compte de l'historique de la conversation : chaque question est traitée indépendamment.
- Si Qdrant est indisponible, la recherche renvoie une liste vide et l'assistant répond qu'il ne trouve pas l'information au lieu de signaler l'erreur.
- La détection du « je ne sais pas » compare la réponse du LLM à une phrase exacte.
- Les dépendances ne sont pas encore figées dans un `requirements.txt`.
