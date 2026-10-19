# University Knowledge Assistant

Assistant de questions-réponses sur les documents officiels d'Avignon Université (statuts, règlement intérieur, chartes, modalités de contrôle des connaissances…), basé sur une architecture **RAG** (*Retrieval-Augmented Generation*).

L'assistant répond **uniquement à partir des documents indexés**, cite ses sources (`[1]`, `[2]`… avec fichier et page) et indique quand l'information n'est pas disponible plutôt que d'inventer une réponse.

> Projet d'apprentissage : l'objectif est de construire un RAG pas à pas, d'abord dans une version simple, puis de l'améliorer avec des techniques de RAG avancé mesurées par une évaluation.

## Fonctionnement

```
                         INGESTION (une fois)
  PDF ──► extraction ──► chunking ──┬─► embedding dense (multilingual-e5-small) ──┐
         (PyMuPDF)     (1000 car.,  │                                              ├─► Qdrant : 1 point = 2 vecteurs
                        overlap 200)└─► vecteur sparse BM25 (fastembed, français) ─┘   "dense" + "bm25" + texte + métadonnées

                         QUESTION (à chaque requête)
  Question + historique ──► routeur (LLM) ──┬─ hors_domaine ──► message de périmètre (sans recherche)
                            type + question  ├─ conversation ──► réponse courte (sans recherche)
                            autonome         └─ documentation
                                                    │
  ┌─────────────────────────────────────────────────┘
  └─► recherche hybride ──► 20 candidats ──► reranker ──► 5 chunks ──► prompt ──► LLM ──► réponse + sources
      dense ("query: ") + BM25              (cross-encoder)           [1] (fichier, p.X)   (HF)   ou « Je ne trouve pas… »
      fusion RRF dans Qdrant
```

Un **routeur** (un appel LLM) classe d'abord la question et la réécrit en question autonome à partir de l'historique (« Et qui doit composer le bureau ? » → « Qui doit composer le bureau d'une association étudiante ? »). Seules les questions de type `documentation` passent par la recherche.

La recherche se fait ensuite en deux étages : un premier étage **rapide** (recherche hybride) qui doit ne rien rater, puis un second **précis** (cross-encoder) qui relit chaque candidat avec la question pour remettre les plus pertinents en tête.

| Étape | Fichier | Fonction |
|---|---|---|
| Chargement et découpage des PDF | [src/pdf_loader.py](src/pdf_loader.py) | `process_knowledge_base` |
| Création de la collection (vecteurs nommés `dense` + `bm25`) | [src/init_db.py](src/init_db.py) | `init_collection` |
| Embeddings denses | [src/embedding.py](src/embedding.py) | `build_vector_store` |
| Vecteurs sparse BM25 | [src/embedding.py](src/embedding.py) | `build_sparse_vectors`, `build_sparse_query` |
| Stockage dans Qdrant | [src/embedding.py](src/embedding.py) | `store_embbedings_in_qdrant` |
| Recherche (`dense`, `bm25` ou `hybrid`) | [src/embedding.py](src/embedding.py) | `search_in_qdrant` |
| Reranking | [src/reranker.py](src/reranker.py) | `rerank` |
| Historique de conversation (derniers échanges, sans citations ni refus) | [src/rag.py](src/rag.py) | `format_history` |
| Routeur : type de question + question autonome | [src/rag.py](src/rag.py) | `route_question` |
| Réponse aux messages de conversation | [src/rag.py](src/rag.py) | `answer_conversation` |
| Construction du contexte et du prompt | [src/rag.py](src/rag.py) | `build_context`, `build_messages` |
| Détection d'un refus du LLM | [src/rag.py](src/rag.py) | `is_not_found` |
| Appel au LLM | [src/llm.py](src/llm.py) | `generate` |
| Pipeline complet | [src/rag.py](src/rag.py) | `answer` |
| Interface de chat | [app/streamlit_app.py](app/streamlit_app.py) | — |

## Stack technique

- **Extraction PDF** : PyMuPDF (via `langchain-community`)
- **Chunking** : `RecursiveCharacterTextSplitter` (`langchain-text-splitters`)
- **Embeddings denses** : [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (384 dimensions, multilingue, préfixes `passage:` / `query:`)
- **Recherche par mots-clés** : BM25 avec [`fastembed`](https://github.com/qdrant/fastembed) (`Qdrant/bm25`, stopwords et stemming français), IDF calculé par Qdrant
- **Base vectorielle** : [Qdrant](https://qdrant.tech/), vecteurs nommés `dense` (cosinus) et `bm25` (sparse), fusion **RRF** (*Reciprocal Rank Fusion*) dans une seule requête (`prefetch` + `FusionQuery`)
- **Reranker** : cross-encoder multilingue [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (`sentence-transformers`)
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
├── evaluation/
│   ├── compare_search_modes.py   # Comparaison des modes de recherche sur des questions de référence
│   ├── out_of_scope_questions.py # Évaluation des questions hors documentation (refus, faux refus, routeur)
│   └── results/                  # Réponses enregistrées par les évaluations (relecture manuelle)
├── src/
│   ├── pdf_loader.py        # Chargement + chunking des PDF
│   ├── init_db.py           # Création de la collection Qdrant (vecteurs dense + bm25)
│   ├── embedding.py         # Embeddings dense et BM25, stockage et recherche dans Qdrant
│   ├── reranker.py          # Reranking des candidats par cross-encoder
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
            fastembed python-dotenv
```

> PyTorch est installé comme dépendance de `sentence-transformers`. Sur un GPU ancien (compute capability < 7.5), installer un build compatible, par exemple :
> `pip install torch --index-url https://download.pytorch.org/whl/cu126`. L'embedding e5 et le reranker utilisent le GPU s'il est disponible, sinon le CPU.
>
> Au premier lancement, les modèles (e5, BM25, reranker) sont téléchargés depuis Hugging Face.

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
store_embbedings_in_qdrant(chunks, vectors)   # calcule aussi les vecteurs BM25
```

> Après une modification du chunking, du modèle d'embedding ou du schéma de la collection, supprimer la collection `knowledge_base` dans Qdrant avant de réindexer : une collection existante n'est pas recréée. C'est notamment nécessaire pour passer d'une collection à un seul vecteur (V1) à la collection hybride (`dense` + `bm25`).

### Poser une question en Python

```python
from src.rag import answer

result = answer("Qui doit être le président d'une association étudiante ?")
print(result["answer"])
print(result["sources"])             # [{"number", "source_file", "page", "score", "rerank_score"}, ...]
print(result["type"])                # "documentation", "conversation" ou "hors_domaine"
print(result["rewritten_question"])  # question réellement utilisée pour la recherche
```

Pour une question de suivi, passer l'historique :

```python
from src.rag import answer, format_history

messages = [
    {"role": "user", "content": "Qui peut présider une association étudiante ?"},
    {"role": "assistant", "content": result["answer"]},
]
answer("Et qui doit composer le bureau ?", history=format_history(messages))
```

Paramètres de `answer()` :

| Paramètre | Rôle | Défaut |
|---|---|---|
| `history` | Historique renvoyé par `format_history` (`None` pour la première question) | `None` |
| `top_k` | Nombre de chunks transmis au LLM | `5` |
| `mode` | Recherche `"hybrid"`, `"dense"` ou `"bm25"` | `"hybrid"` |
| `use_reranker` | Reclasse les candidats avec le cross-encoder | `True` |
| `candidates` | Nombre de chunks récupérés dans Qdrant puis transmis au reranker | `20` |
| `min_rerank_score` | Seuil sous lequel l'assistant répond « je ne trouve pas » sans appeler le LLM (`None` = désactivé, voir [Résultats](#résultats)) | `None` |

### Comparer les modes de recherche

```bash
python -m evaluation.compare_search_modes
```

Affiche, pour 11 questions de référence (mots-clés et reformulées), le top 3 de chaque mode, le Hit@3 par mode et des tests sur le stemming et l'échelle des scores.

### Évaluer les questions hors documentation

```bash
python -m evaluation.out_of_scope_questions --label <nom_du_run>
```

Pose 23 questions (hors domaine, non couvertes, partielles, conversation, tentatives de détournement, et questions couvertes pour détecter les faux refus), affiche les métriques et enregistre toutes les réponses dans `evaluation/results/` pour une relecture manuelle. Le script appelle le LLM une à deux fois par question.

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
- [x] Recherche hybride : dense + BM25, fusion RRF dans Qdrant
- [x] Reranker : cross-encoder multilingue, intégré à `answer()`, seuil de pertinence (désactivé par défaut)
- [x] Contexte conversationnel : réécriture de la question de suivi en question autonome à partir de l'historique (`format_history`, `route_question`)
- [x] Questions hors documentation : routeur (documentation / conversation / hors domaine), réponses adaptées, règles anti-déduction et réponses partielles dans le prompt, affichage des refus dans Streamlit
- [ ] Construire l'API FastAPI
- [ ] Créer un dataset d'évaluation
- [ ] Évaluation du Retriever
- [ ] Évaluation du la génération
- [ ] Pistes suivantes : voir [Axes d'amélioration](#axes-damélioration)

## Résultats

Mesures sur 11 questions de référence (6 « mots-clés » : numéros d'articles, chiffres, sigles ; 5 « sens » : questions reformulées). Un succès = un chunk de la bonne page du bon document dans les résultats.

**Recherche (top 3)**

| Mode | Mots-clés | Sens |
|---|---|---|
| dense | 5/6 | 5/5 |
| bm25 | 5/6 | 4/5 |
| hybrid | 5/6 | 5/5 |

**Reranker (20 candidats hybrides → 5)**

| | Hit@1 | Hit@3 | MRR@5 |
|---|---|---|---|
| hybrid | **9/11** | 10/11 | **0,85** |
| hybrid + rerank | 7/11 | **11/11** | 0,79 |

Observations :
- **BM25** classe nettement mieux les termes rares et exacts (« Loi EgAlim », « 30 ETCS ») mais échoue sur les reformulations ; l'**hybride** n'est jamais le pire des deux.
- Le **reranker** fait remonter des chunks absents du top 3 (la question sur l'article 23 n'est trouvée qu'avec lui) et écarte les pages de sommaire, mais place parfois le bon chunk en 2ᵉ ou 3ᵉ position. Le LLM lisant les 5 chunks, c'est le rappel qui a été privilégié (reranker activé par défaut).
- **Seuil de pertinence** : le meilleur score du reranker va de -4,5 à +4,8 pour des questions qui ont une réponse et de -6,9 à -0,7 pour des questions hors sujet. Les deux groupes se chevauchent (les questions reformulées obtiennent des scores aussi bas que les questions hors sujet), d'où le seuil désactivé par défaut. Vérifié à nouveau avec le jeu hors documentation : une question partielle légitime descend à -5,2, alors qu'aucune question non couverte ne passe sous -4,4 ; un seuil de sécurité à -6 n'apporterait donc rien.

**Questions hors documentation** (23 questions, `evaluation/out_of_scope_questions.py`)

| Version | Refus corrects | Faux refus | Conversation | Routeur | Latence moyenne |
|---|---|---|---|---|---|
| Avant (prompt seul) | 11/11 | 1/9 | 0/3 | — | 2,3 s |
| Routeur + réponses par type + prompt renforcé | **11/11** | **0/9** | **3/3** | **23/23** | **2,3 s** |

- *Refus corrects* : questions hors domaine, non couvertes ou de détournement qui reçoivent un refus. *Faux refus* : questions couvertes (entièrement ou partiellement) qui reçoivent un refus. *Conversation* : « Bonjour », « Merci », « Que sais-tu faire ? » reçoivent une vraie réponse, sans sources.
- Le `SYSTEM_PROMPT` gérait déjà bien les refus ; le gain vient surtout des messages de conversation (auparavant refusés ou accompagnés de sources sans rapport) et de la latence : les questions hors domaine ne déclenchent plus ni recherche ni génération, ce qui compense l'appel du routeur.
- Le LLM n'est pas déterministe (même à température 0,1) : une même question peut obtenir une réponse puis un refus d'un run à l'autre. Un écart d'une question entre deux runs n'est pas significatif.

## Limites actuelles

- **Sommaires indexés** : les tables des matières (règlement intérieur, statuts) sont découpées comme du contenu et remontent sur les requêtes du type « Article N ».
- **Chunks multi-articles** : un chunk de 1000 caractères peut contenir la fin d'un article et le début du suivant, ce qui gêne la recherche et le reranker.
- La *Charte pour l'égalité entre les femmes et les hommes* ne produit aucun chunk à l'ingestion : PyMuPDF n'en extrait pas de texte (probablement un PDF scanné, qui nécessiterait de l'OCR).
- **Historique non transmis à la génération** : l'historique sert uniquement à réécrire la question de suivi en question autonome ; le LLM qui rédige la réponse ne reçoit que cette question réécrite et les extraits. Il ne peut donc pas faire référence à ses réponses précédentes (« Comme indiqué précédemment… ») ni traiter une demande portant sur la conversation elle-même (« Résume ta réponse »).
- **Déductions à partir d'extraits proches** : malgré la règle « ne déduis pas une information absente », le LLM l'enfreint parfois. Exemple relevé : pour « Quelles sont les démarches pour faire un stage ? », il a appliqué au stage un délai de demande d'autorisation qui concerne un *événement*. Les métriques automatiques ne détectent pas ce type d'erreur, seule la relecture des réponses le montre.
- **Un appel LLM supplémentaire par question** : le routeur est appelé pour chaque question, y compris la première d'une conversation (latence et crédits Hugging Face).
- **Périmètre décrit à la main** : le message hors domaine et le prompt de conversation s'appuient sur la constante `DOCUMENTS_SCOPE`, à mettre à jour à chaque ajout de document.
- **Détection des refus** : `is_not_found` reconnaît un refus quand la réponse *commence* par « Je ne trouve pas cette information ». Un refus formulé autrement ne serait pas détecté (les sources resteraient alors affichées).
- **Jeux de test réduits** (11 et 23 questions) : les résultats sont indicatifs.
- Si Qdrant est indisponible, la recherche renvoie une liste vide et l'assistant répond qu'il ne trouve pas l'information au lieu de signaler l'erreur.
- Les dépendances ne sont pas encore figées dans un `requirements.txt`.

## Axes d'amélioration

- **Transmettre un historique court à la génération** : donner au LLM la question réécrite **et** les derniers échanges (insérés entre le message système et le message utilisateur dans `build_messages`), pour permettre des transitions naturelles (« Comme indiqué précédemment… ») et les demandes du type « Résume ta réponse ». Il faudra alors préciser dans `SYSTEM_PROMPT` que l'historique sert uniquement à comprendre la question et que les faits doivent venir exclusivement des extraits fournis, pour qu'une réponse précédente erronée ne soit jamais reprise comme source.
- **Données** : supprimer les pages de sommaire à l'ingestion et découper les documents par article (numéro d'article stocké en métadonnée).
- **Détecter les déductions abusives** : vérifier automatiquement que chaque affirmation de la réponse est appuyée par l'extrait cité (par exemple avec un second appel LLM « juge » ou le score du reranker entre la phrase et l'extrait).
- **Réduire le coût du routeur** : détecter les salutations simples sans LLM, ou utiliser un modèle plus petit pour le routage.
- **Évaluation du contexte conversationnel** : jeu de conversations de test (questions de suivi, changements de sujet, questions déjà autonomes) pour mesurer la qualité de la réécriture et comparer le Hit@3 avec la question brute et avec la question réécrite.
