# University Knowledge Assistant

Assistant de questions-réponses sur les documents officiels d'Avignon Université (statuts, règlement intérieur, chartes, modalités de contrôle des connaissances…), basé sur une architecture **RAG** (*Retrieval-Augmented Generation*).

L'assistant répond **uniquement à partir des documents indexés**, cite ses sources (`[1]`, `[2]`… avec fichier et page) et indique quand l'information n'est pas disponible plutôt que d'inventer une réponse.

> Projet d'apprentissage : l'objectif est de construire un RAG pas à pas, d'abord dans une version simple, puis de l'améliorer avec des techniques de RAG avancé mesurées par une évaluation.

## Fonctionnement

L'application est découpée en deux parties : une **API REST** (FastAPI) qui contient tout le moteur RAG, et une **interface** (Streamlit) qui ne fait qu'envoyer des requêtes HTTP à l'API. L'API est sans état : le client envoie l'historique de la conversation avec chaque question.

```
┌──────────────┐  POST /ask (question + historique)  ┌───────────────────────────────────┐
│  Streamlit   │ ──────────────────────────────────► │  API FastAPI                      │
│  (interface) │ ◄────────────────────────────────── │  routeur, recherche, reranker,    │──► Qdrant
└──────────────┘  réponse + sources + type           │  LLM (modèles chargés au démarrage)│──► LLM Hugging Face
   autres clients : curl, scripts…                    └───────────────────────────────────┘
```

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
| API REST (`POST /ask`, `GET /health`) | [api/main.py](api/main.py), [api/schemas.py](api/schemas.py) | `ask`, `health` |
| Interface de chat (cliente de l'API) | [app/streamlit_app.py](app/streamlit_app.py) | `ask_api` |

## Stack technique

- **Extraction PDF** : PyMuPDF (via `langchain-community`)
- **Chunking** : `RecursiveCharacterTextSplitter` (`langchain-text-splitters`)
- **Embeddings denses** : [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (384 dimensions, multilingue, préfixes `passage:` / `query:`)
- **Recherche par mots-clés** : BM25 avec [`fastembed`](https://github.com/qdrant/fastembed) (`Qdrant/bm25`, stopwords et stemming français), IDF calculé par Qdrant
- **Base vectorielle** : [Qdrant](https://qdrant.tech/), vecteurs nommés `dense` (cosinus) et `bm25` (sparse), fusion **RRF** (*Reciprocal Rank Fusion*) dans une seule requête (`prefetch` + `FusionQuery`)
- **Reranker** : cross-encoder multilingue [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (`sentence-transformers`)
- **LLM** : modèle Hugging Face via les *Inference Providers* (`openai/gpt-oss-20b` par défaut), avec `ChatHuggingFace`
- **API** : FastAPI + Uvicorn, schémas Pydantic, tests avec `pytest` et `TestClient`
- **Interface** : Streamlit (appelle l'API avec `requests`)

## Structure du projet

```
.
├── api/
│   ├── main.py              # API FastAPI : /ask, /health, chargement des modèles au démarrage
│   └── schemas.py           # Schémas Pydantic des requêtes et réponses
├── app/
│   └── streamlit_app.py     # Interface de chat (cliente de l'API)
├── data/
│   ├── raw/                 # PDF sources
│   └── processed/
├── evaluation/
│   ├── dataset/                  # Dataset d'évaluation versionné (eval_dataset_v1.jsonl, README, CHANGELOG, candidats)
│   ├── dataset.py                # Schéma d'une question (Pydantic) et load_dataset()
│   ├── validate_dataset.py       # Validation du dataset (schéma, cohérence, sources présentes dans Qdrant)
│   ├── generate_candidates.py    # Génération de questions candidates par le LLM, à relire
│   ├── assign_splits.py          # Découpage dev / test stratifié
│   ├── compare_search_modes.py   # Comparaison des modes de recherche (questions mots-clés et reformulées)
│   ├── out_of_scope_questions.py # Comportement : refus, faux refus, conversation, routeur
│   ├── metrics.py                # Métriques du retriever : Hit@k, Recall@k, reciprocal rank
│   ├── eval_retriever.py         # Évaluation du retriever (5 configurations, sans LLM sauf réécriture en cache)
│   ├── error_analysis.py         # Affichage des questions ratées d'un run, pour en classer les causes
│   ├── generation_metrics.py     # Métriques des réponses : faits clés, citations
│   ├── eval_generation.py        # Évaluation des réponses (génération sauvegardée, puis métriques automatiques)
│   └── results/                  # Runs enregistrés (JSON), cache des réécritures, analyse d'erreurs
├── src/
│   ├── pdf_loader.py        # Chargement + chunking des PDF
│   ├── init_db.py           # Création de la collection Qdrant (vecteurs dense + bm25)
│   ├── embedding.py         # Embeddings dense et BM25, stockage et recherche dans Qdrant
│   ├── reranker.py          # Reranking des candidats par cross-encoder
│   ├── rag.py               # Prompt, contexte et pipeline answer()
│   └── llm.py               # Point d'entrée unique vers le LLM
├── test_directory/          # Petit jeu de PDF pour les tests
├── tests/
│   ├── test_api.py          # Tests de l'API (sans Qdrant, modèles ni LLM)
│   ├── test_metrics.py      # Tests des métriques du retriever, sur des exemples calculés à la main
│   └── test_generation_metrics.py  # Tests des métriques des réponses (faits clés, citations)
├── test_qdrant.ipynb        # Notebook d'expérimentation (ingestion, recherche, RAG)
├── requirements.txt         # Dépendances (versions figées)
└── .env.example             # Modèle des variables d'environnement
```

## Installation

### 1. Environnement Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Sur un GPU ancien (compute capability < 7.5), installer le build CUDA compatible de PyTorch :
> `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`. L'embedding e5 et le reranker utilisent le GPU s'il est disponible, sinon le CPU.
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
| `API_URL` | URL de l'API appelée par l'interface Streamlit | `http://localhost:8000` |

Le fichier `.env` est exclu du dépôt par le `.gitignore`. Le token Hugging Face n'est utilisé que par l'API, jamais par l'interface.

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

### Lancer l'API

```bash
uvicorn api.main:app --reload
```

Depuis la racine du projet (pour que les paquets `src` et `api` soient trouvés), avec **un seul worker** : chaque worker charge ses propres modèles. Au démarrage, l'API vérifie que la collection Qdrant existe et charge les modèles (e5, BM25, reranker), puis affiche « Application startup complete ».

- Documentation interactive (Swagger) : http://localhost:8000/docs
- État de l'API et de Qdrant : `GET /health` → `{"status": "ok", "qdrant": "ok", "points": 235}` (503 si Qdrant est indisponible ou la collection vide)

Exemple de question, puis de question de suivi :

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Qui peut présider une association étudiante ?"}'

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{
  "question": "Et qui doit composer le bureau ?",
  "history": [
    {"role": "user", "content": "Qui peut présider une association étudiante ?"},
    {"role": "assistant", "content": "Un étudiant inscrit à l'\''Université d'\''Avignon."}
  ]
}'
```

Corps de la requête `POST /ask` : `question` (1 à 1000 caractères), `history` (messages `user` / `assistant`, 50 au maximum ; le serveur ne garde que les derniers échanges), `top_k` (1 à 10), `mode` (`hybrid`, `dense`, `bm25`), `use_reranker`. Une requête invalide reçoit une erreur **422**.

Réponse : `answer`, `type` (`documentation`, `conversation`, `hors_domaine`), `refused` (l'assistant n'a pas pu répondre), `rewritten_question` et `sources`. Si Qdrant ou le LLM est indisponible, l'API répond **503** (500 pour une erreur inattendue) ; le détail technique est écrit dans les logs du serveur, pas renvoyé au client.

### Lancer les tests

```bash
python -m pytest
```

Les tests de l'API remplacent `answer()` et Qdrant par des fonctions factices : ils ne nécessitent ni Qdrant, ni modèle, ni token Hugging Face.

### Poser une question en Python (sans l'API)

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

### Dataset d'évaluation

Toutes les évaluations utilisent le même dataset versionné, [evaluation/dataset/eval_dataset_v1.jsonl](evaluation/dataset/eval_dataset_v1.jsonl) : **68 questions** (factuelles, mots-clés, reformulées, multi-sources, partielles, non couvertes, hors domaine, conversation, suivi avec historique, détournement), chacune avec ses sources attendues (fichier + pages), une réponse de référence, des faits clés et un split `dev` (47) / `test` (21). Le détail du schéma, de la couverture et des règles d'utilisation est dans [evaluation/dataset/README.md](evaluation/dataset/README.md).

```bash
python -m evaluation.validate_dataset                          # schéma, cohérence, sources présentes dans Qdrant
python -m evaluation.generate_candidates --file STATUTS --n 8  # questions candidates générées par le LLM, à relire
python -m evaluation.assign_splits                             # split dev / test des nouvelles questions
```

> On règle le système sur le split `dev` et on annonce les scores sur le split `test`.

### Comparer les modes de recherche

```bash
python -m evaluation.compare_search_modes --split dev
```

Affiche, pour les questions « mots-clés » et « reformulées » du dataset, le top 3 de chaque mode (dense, BM25, hybride), le Hit@3 par mode et des tests sur le stemming et l'échelle des scores.

### Évaluer le retriever

```bash
python -m evaluation.eval_retriever --split dev --label <nom_du_run>
python -m evaluation.eval_retriever --split dev --configs hybrid hybrid+rerank20   # quelques configurations
python -m evaluation.error_analysis --config hybrid+rerank20 --metric hit@1        # questions ratées du dernier run
```

Évalue la recherche seule (sans génération) sur les questions du dataset qui ont des sources attendues. L'unité de pertinence est la page (fichier, page) : les chunks d'une même page ne comptent qu'une fois. Pour chaque configuration (`dense`, `bm25`, `hybrid`, `hybrid+rerank20`, `hybrid+rerank40`), le script calcule Hit@1/3/5/10 et le MRR sur un top 10, puis le détail par catégorie, document et difficulté, et la latence. Les questions de suivi sont cherchées deux fois, brute et réécrite. Les réécritures sont mises en cache dans `evaluation/results/rewrites_cache.json` pour que les runs soient reproductibles et sans appel au LLM (`--refresh-rewrites` pour les refaire). Chaque run est enregistré dans `evaluation/results/retriever_<label>_<date>.json` avec son contexte (commit git, modèles, configurations) et le détail par question.

### Évaluer les réponses

```bash
python -m evaluation.eval_generation --split dev --label <nom_du_run>
python -m evaluation.eval_generation --split dev --categories factuelle partielle --label essai
python -m evaluation.eval_generation --answers evaluation/results/generation_<label>_<date>.json   # recalcul sans LLM
```

Pose chaque question du dataset **une seule fois** à `answer()` (avec son historique) et enregistre les réponses, **avec le texte des extraits** transmis au LLM, dans `evaluation/results/generation_<label>_<date>.json`. Les métriques sont calculées à partir de ce fichier : on peut donc les corriger et les recalculer (`--answers`) sans régénérer. Métriques automatiques, sans LLM juge :
- **comportement** : refus corrects, faux refus, messages de conversation, routeur ;
- **faits clés** : part des `key_facts` présents dans la réponse (alternatives séparées par `|`, comparaison après normalisation des accents, apostrophes, tirets et espaces insécables ; les citations `[n]` sont retirées avant la comparaison) ;
- **citations** : réponses avec au moins une citation, citations `[n]` sans extrait correspondant (inventées), part des citations qui pointent vers une page attendue.

Le script appelle le LLM une à deux fois par question : limiter le run avec `--split` ou `--categories`.

### Évaluer le comportement de l'assistant

```bash
python -m evaluation.out_of_scope_questions --label <nom_du_run> --split dev
python -m evaluation.out_of_scope_questions --label <nom_du_run> --categories hors_domaine non_couverte conversation
```

Pose les questions du dataset (avec leur historique pour les questions de suivi), mesure les refus corrects, les faux refus, les réponses aux messages de conversation et la précision du routeur, et enregistre toutes les réponses dans `evaluation/results/` pour une relecture manuelle. Le script appelle le LLM une à deux fois par question : limiter le run avec `--split` ou `--categories`.

### Lancer l'interface

L'API doit être lancée (voir plus haut), puis dans un second terminal :

```bash
python -m streamlit run app/streamlit_app.py
```

L'interface envoie chaque question à l'API (`API_URL`) avec l'historique de la conversation. Si l'API ne répond pas, un message d'erreur indique comment la lancer.

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
- [x] API FastAPI : `POST /ask` et `GET /health`, schémas Pydantic, modèles chargés au démarrage, erreurs 503/500, tests `pytest`, interface Streamlit branchée sur l'API
- [x] Dataset d'évaluation v1 : 68 questions (62 manuelles, 6 synthétiques relues), sources vérifiées dans Qdrant, split dev / test, scripts d'évaluation branchés dessus
- [x] Évaluation du retriever : métriques testées (Hit@k, MRR), 5 configurations comparées sur dev, analyse d'erreurs, score final sur test, classement rendu déterministe (départage des ex aequo RRF)
- [x] Évaluation de la génération (version minimale) : réponses générées une fois et sauvegardées avec leurs extraits, métriques automatiques testées (comportement, faits clés, citations), score sur test. La fidélité aux extraits (LLM juge) est reportée dans les axes d'amélioration
- [ ] Pistes suivantes : voir [Axes d'amélioration](#axes-damélioration)

## Résultats

### Retriever (dataset v1)

Recherche seule, sans génération (`evaluation/eval_retriever.py`), sur les questions qui ont des sources attendues. Unité : la page (fichier, page), dédoublonnée. Pour les questions de suivi, c'est la question réécrite qui est cherchée, comme dans `answer()`.

**Choix de la configuration sur `dev` (37 questions)**

| Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Latence (s) |
|---|---|---|---|---|---|---|
| dense | 0.68 | 0.84 | 0.92 | 1.00 | 0.78 | 0.02 |
| bm25 | 0.65 | 0.86 | 0.86 | 0.92 | 0.75 | 0.00 |
| hybrid | 0.73 | 0.89 | 0.89 | 0.92 | 0.81 | 0.02 |
| **hybrid+rerank20** | **0.84** | 0.95 | **0.97** | 0.97 | **0.89** | 0.87 |
| hybrid+rerank40 | 0.78 | 0.97 | 0.97 | 0.97 | 0.87 | 1.61 |

Configuration retenue : **hybride + reranker sur 20 candidats**, déjà la configuration par défaut de `answer()`. Passer à 40 candidats n'apporte rien et double la latence. Sur dev, le reranker rattrape surtout les questions sur les Statuts (Hit@5 de 0,60 à 1,00) et les questions de suivi. BM25 seul est faible sur les questions reformulées (Hit@5 = 0,50).

**Score final sur `test` (16 questions, un seul run, aucun réglage après)**

| Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Latence (s) |
|---|---|---|---|---|---|---|
| dense | 0.69 | 0.81 | 0.81 | 0.94 | 0.77 | 0.02 |
| bm25 | 0.69 | 0.75 | 0.94 | 0.94 | 0.77 | 0.00 |
| hybrid | 0.81 | 0.81 | 0.94 | 0.94 | 0.84 | 0.02 |
| **hybrid+rerank20 (retenue)** | 0.69 | **1.00** | **1.00** | 1.00 | 0.82 | 0.91 |
| hybrid+rerank40 | 0.69 | 0.94 | 1.00 | 1.00 | 0.82 | 1.61 |

- Avec la configuration retenue, **la bonne page est toujours dans les 5 extraits transmis au LLM** (Hit@5 = 1,00), et toujours dans les 3 premiers.
- **Sur test, le reranker fait baisser le Hit@1** (0,69 contre 0,81 pour l'hybride seul) : il place la bonne page en 1ʳᵉ position pour 2 questions, mais la fait descendre au 2ᵉ ou 3ᵉ rang pour 4 autres, surtout des questions **mots-clés** (« Article 14 », « 30 ECTS »), que BM25 classe déjà très bien. C'est la même tendance que celle observée avant le dataset. Comme le LLM lit les 5 extraits, c'est le Hit@5 qui compte, et le choix est maintenu.
- **Petits effectifs** : avec 16 questions en test, une question vaut environ 6 points. L'écart de Hit@1 entre dev (0,84) et test (0,69) représente 2 à 3 questions et reste dans la marge d'incertitude.

**Analyse d'erreurs sur dev** ([evaluation/results/error_analysis_dev.md](evaluation/results/error_analysis_dev.md)) : sur les 6 questions où la bonne page n'est pas 1ʳᵉ, 4 sont des **pages concurrentes** (une autre page du même document traite du même sujet, typiquement dans les Statuts), et 2 sont des problèmes de **vocabulaire** (question reformulée sans les mots du document). Aucune erreur du dataset. Le seul vrai raté (`ref-plagiat`, page absente du top 10) cumule un vocabulaire absent, la page de sommaire arrivée 1ʳᵉ (le mot « article » de la question correspond aux « Article N » du sommaire) et un chunk qui couvre 3 articles.

**Réécriture des questions de suivi** : sur dev, la réécriture fait passer le Hit@5 du dense de 0,50 à 0,75 (sur 4 questions) et ne change rien pour les autres configurations. Deux questions ne sont pas réécrites par le routeur (« Et l'alcool ? », « Et pour les bouteilles en plastique ? »), mais restent bien retrouvées : le mot-clé suffit à la recherche. Ce défaut se mesurera sur la qualité de la réponse.

### Réponses (dataset v1)

Pipeline complet (`answer()` avec la configuration retenue : hybride + reranker 20 → 5 extraits), `evaluation/eval_generation.py`. Les faits clés et les citations sont mesurés sur les questions qui doivent recevoir une réponse.

| Split | Comportement correct | Refus corrects | Faux refus | Conversation | Routeur | Faits clés (moyenne) | Faits clés complets | Réponses citées | Citations inventées | Citations sur une page attendue | Latence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev (47) | 43/47 | 8/8 | 4/37 | 2/2 | 47/47 | 0,81 | 29/37 | 33/33 | 0 | 98 % | 2,7 s |
| **test (21)** | **21/21** | **4/4** | **0/16** | **1/1** | **21/21** | **0,81** | **12/16** | **16/16** | **0** | **100 %** | **2,9 s** |

- **Refus et routage sans erreur** : toutes les questions hors domaine, non couvertes et de détournement sont refusées, et le routeur classe correctement toutes les questions (68/68 sur dev + test).
- **Citations fiables** : chaque réponse cite ses extraits, aucune citation ne renvoie à un extrait inexistant, et les extraits cités sont (presque) toujours sur une page attendue.
- **Le point faible est la complétude**, surtout sur les questions mots-clés (faits clés 0,72 sur dev, 0,33 sur test) : la réponse est juste mais ne reprend pas le fait principal. Par exemple, pour « 30 ETCS », le LLM cite un exemple de calcul au lieu de la règle de pondération (0,75 / 0,25), alors que la bonne page est dans les extraits.
- **Relecture des réponses ratées sur dev** (9 réponses) : 2 viennent du **retriever** (le bon chunk n'est pas dans les 5 extraits, par exemple `ref-plagiat`, donc le refus est le bon comportement), 2 sont des **refus à tort** alors que la bonne page était fournie (`ref-enseignant-moqueur`, `part-stage` : la règle « ne déduis pas » rend le LLM trop prudent sur les reformulations), 2 sont des **réponses incomplètes**, 1 est **en anglais** malgré la règle « réponds dans la langue de la question » (`kw-ueo`), 1 est un **fait clé trop strict** (« proscrit » contre « proscrire »), et 1 était une **erreur de la métrique** (tiret insécable dans « Vice‑Président »), corrigée et couverte par un test.
- **À lire avec prudence** : les faits clés sont comparés par texte, donc une paraphrase juste compte comme un raté (c'est le cas de `suivi-bouteilles` sur test). À l'inverse, une réponse qui contient les faits clés peut quand même contenir une affirmation non appuyée : la **fidélité aux extraits n'est pas mesurée** (voir les axes d'amélioration). Avec 16 questions en test, une question vaut environ 6 points.

### Avant le dataset v1

> Les mesures de la recherche, du reranker et des questions hors documentation faites sur les anciennes listes de 11 et 23 questions sont remplacées par les résultats ci-dessus. Elles sont gardées pour l'historique des choix (seuil de pertinence, gain du routeur).

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

- **Sommaires indexés** : les tables des matières (règlement intérieur, statuts) sont découpées comme du contenu et remontent sur les requêtes du type « Article N » (cause du seul raté du retriever sur dev).
- **Le reranker dégrade parfois le classement des questions mots-clés** (« Article 14 », « 30 ECTS ») : la bonne page passe de la 1ʳᵉ à la 2ᵉ ou 3ᵉ position. Elle reste dans les 5 extraits lus par le LLM.
- **Réponses incomplètes et refus trop prudents** : sur les questions mots-clés, le LLM répond juste mais oublie souvent le fait principal ; sur certaines reformulations, il refuse alors que la bonne page est dans les extraits (effet de la règle « ne déduis pas »). Une réponse a aussi été rédigée en anglais à une question en français.
- **Fidélité non mesurée** : l'évaluation des réponses vérifie les faits clés et les citations, pas que chaque affirmation est appuyée par l'extrait cité. Les faits clés, comparés par texte, ne reconnaissent pas les paraphrases.
- **Routeur et questions de suivi elliptiques** : « Et l'alcool ? » ou « Et pour les bouteilles en plastique ? » ne sont pas réécrites en questions autonomes.
- **Chunks multi-articles** : un chunk de 1000 caractères peut contenir la fin d'un article et le début du suivant, ce qui gêne la recherche et le reranker.
- La *Charte pour l'égalité entre les femmes et les hommes* ne produit aucun chunk à l'ingestion : PyMuPDF n'en extrait pas de texte (probablement un PDF scanné, qui nécessiterait de l'OCR).
- **Historique non transmis à la génération** : l'historique sert uniquement à réécrire la question de suivi en question autonome ; le LLM qui rédige la réponse ne reçoit que cette question réécrite et les extraits. Il ne peut donc pas faire référence à ses réponses précédentes (« Comme indiqué précédemment… ») ni traiter une demande portant sur la conversation elle-même (« Résume ta réponse »).
- **Déductions à partir d'extraits proches** : malgré la règle « ne déduis pas une information absente », le LLM l'enfreint parfois. Exemple relevé : pour « Quelles sont les démarches pour faire un stage ? », il a appliqué au stage un délai de demande d'autorisation qui concerne un *événement*. Les métriques automatiques ne détectent pas ce type d'erreur, seule la relecture des réponses le montre.
- **Un appel LLM supplémentaire par question** : le routeur est appelé pour chaque question, y compris la première d'une conversation (latence et crédits Hugging Face).
- **Périmètre décrit à la main** : le message hors domaine et le prompt de conversation s'appuient sur la constante `DOCUMENTS_SCOPE`, à mettre à jour à chaque ajout de document.
- **Détection des refus** : `is_not_found` reconnaît un refus quand la réponse *commence* par « Je ne trouve pas cette information ». Un refus formulé autrement ne serait pas détecté (les sources resteraient alors affichées).
- **Dataset de taille modeste** (68 questions, 21 en test) : un écart d'une ou deux questions entre deux versions n'est pas significatif. Les réponses de référence ont été écrites par une seule personne et dépendent de la version des documents (RI de mars 2025, statuts du 6 février 2025…).
- **API sans authentification ni limite de débit** : n'importe quel client qui peut joindre l'API peut l'interroger (et consommer les crédits Hugging Face). À ne pas exposer publiquement en l'état.
- **Réponse non streamée** : l'API renvoie la réponse complète en une fois, après plusieurs secondes d'attente.
- **Un seul worker** : chaque worker Uvicorn charge ses propres modèles, ce qui limite le nombre de requêtes traitées en parallèle sur une petite machine.

## Axes d'amélioration

- **Transmettre un historique court à la génération** : donner au LLM la question réécrite **et** les derniers échanges (insérés entre le message système et le message utilisateur dans `build_messages`), pour permettre des transitions naturelles (« Comme indiqué précédemment… ») et les demandes du type « Résume ta réponse ». Il faudra alors préciser dans `SYSTEM_PROMPT` que l'historique sert uniquement à comprendre la question et que les faits doivent venir exclusivement des extraits fournis, pour qu'une réponse précédente erronée ne soit jamais reprise comme source.
- **Données** : supprimer les pages de sommaire à l'ingestion et découper les documents par article (numéro d'article stocké en métadonnée).
- **Évaluation de la génération, version complète** : LLM juge de la fidélité (découpage de la réponse en affirmations, chacune appuyée / non appuyée / contredite par les extraits) et de la complétude des réponses partielles, juge validé sur une quinzaine de réponses annotées à la main ; run « oracle » (extraits des pages attendues fournis directement) pour séparer les erreurs du retriever et du générateur ; comparaison avec RAGAS.
- **Prompt** : demander de reprendre en priorité la règle ou le chiffre qui répond à la question (complétude des questions mots-clés), assouplir la règle anti-déduction pour les reformulations (refus à tort), et rappeler la langue de réponse en fin de prompt. À mesurer sur dev avec `eval_generation`.
- **Détecter les déductions abusives** : vérifier automatiquement que chaque affirmation de la réponse est appuyée par l'extrait cité (par exemple avec un second appel LLM « juge » ou le score du reranker entre la phrase et l'extrait).
- **Réduire le coût du routeur** : détecter les salutations simples sans LLM, ou utiliser un modèle plus petit pour le routage.
- **API** : clé d'API dans un en-tête et limite de débit, streaming de la réponse (Server-Sent Events), CORS pour un client web, `Dockerfile` et `docker-compose` (API + Qdrant).
- **Retriever** : chunking par article et exclusion des sommaires, puis comparer sur dev avec `eval_retriever` ; tester une fusion qui garde la 1ʳᵉ place de BM25 quand le score BM25 est très net (questions mots-clés), ou un enrichissement de la requête (synonymes, HyDE) pour les questions reformulées ; agrandir le dataset (v2) pour réduire la marge d'incertitude.
- **Évaluation du contexte conversationnel** : jeu de conversations de test (questions de suivi, changements de sujet, questions déjà autonomes) pour mesurer la qualité de la réécriture et comparer le Hit@3 avec la question brute et avec la question réécrite.
