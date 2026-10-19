# Dataset d'évaluation

Source de vérité unique pour toutes les évaluations du projet (retriever, génération, comportement). Version actuelle : **v1** (`eval_dataset_v1.jsonl`), **gelée** : les corrections vont dans une v2 (voir [CHANGELOG.md](CHANGELOG.md)).

## Format

Un objet JSON par ligne (JSONL), validé par le modèle Pydantic `EvalItem` ([evaluation/dataset.py](../dataset.py)). Tous les scripts chargent le dataset avec `load_dataset(split=..., categories=...)`.

| Champ | Rôle |
|---|---|
| `id` | Identifiant stable (`fact-…`, `kw-…`, `ref-…`, `multi-…`, `part-…`, `nc-…`, `hd-…`, `conv-…`, `det-…`, `suivi-…`, `syn-…`) |
| `question`, `history` | La question, et les messages précédents pour les questions de suivi |
| `category` | Type de question (voir couverture ci-dessous) |
| `expected_behavior` | `reponse`, `reponse_partielle`, `refus` ou `conversation` |
| `expected_type` | Type attendu du routeur : `documentation`, `conversation`, `hors_domaine` |
| `expected_sources` | Fichier + pages **numérotées à partir de 1** qui contiennent la réponse (jamais les ids de chunks, qui changent à chaque réindexation) |
| `reference_answer` | Réponse de référence, exactement appuyée sur le document |
| `key_facts` | Faits courts que la réponse doit contenir ; alternatives séparées par `|` (`4 ans|quatre ans`) |
| `difficulty`, `origin`, `split`, `notes` | Difficulté, `manuelle` / `synthetique`, `dev` / `test`, remarques |

## Couverture de la v1 (68 questions)

| Catégorie | Plan | v1 | dev | test |
|---|---|---|---|---|
| factuelle | 15 | 21 | 15 | 6 |
| mots_cles | 8 | 9 | 6 | 3 |
| reformulee | 8 | 8 | 6 | 2 |
| multi_sources | 5 | 5 | 3 | 2 |
| partielle | 4 | 4 | 3 | 1 |
| non_couverte | 6 | 6 | 4 | 2 |
| hors_domaine | 4 | 4 | 3 | 1 |
| conversation | 3 | 3 | 2 | 1 |
| suivi | 5 | 6 | 4 | 2 |
| detournement | 2 | 2 | 1 | 1 |
| **total** | **60** | **68** | **47** | **21** |

Questions citant chaque document : règlement intérieur 17, statuts 12, charte des associations 10, charte alimentation 8, guide des MCC 6, charte SI 5. La *Charte pour l'égalité* n'a aucun chunk indexé (PDF sans texte extractible) : la seule question qui la concerne est classée `non_couverte`.

Origine : 62 questions manuelles, 6 synthétiques (8,8 %, maximum autorisé 30 %).

## Construction

1. **Questions manuelles** : écrites après lecture intégrale des PDF, formulées comme un étudiant, réponse et pages vérifiées dans le document. Les 34 questions des anciens scripts (`compare_search_modes.py`, `out_of_scope_questions.py`) ont été reprises et complétées.
2. **Questions synthétiques** : `python -m evaluation.generate_candidates --file <document> --n <nombre>` propose des questions à partir de chunks tirés au hasard (graine fixe) dans [candidates.jsonl](candidates.jsonl). Chaque candidat est relu puis gardé, corrigé ou rejeté ; la décision est enregistrée dans le fichier (`status`, `review_note`). v1 : 11 candidats, 6 retenus (dont 3 corrigés), 5 rejetés (biais lexical, doublon, réponse fausse, question incompréhensible sans l'extrait).
3. **Validation** : `python -m evaluation.validate_dataset` (schéma, ids et questions uniques, cohérence des champs, existence de chaque fichier et de chaque page dans Qdrant, statistiques et plan de couverture). Code retour 1 en cas de problème.
4. **Découpage** : `python -m evaluation.assign_splits` attribue `dev` / `test` (≈ 70 / 30, stratifié par catégorie, graine fixe) aux seules questions qui n'en ont pas encore.

## Règles d'utilisation

- **On règle sur `dev`, on annonce le score sur `test`.** Ajuster un prompt jusqu'à réussir les questions de test rend le score optimiste.
- **On ne modifie jamais une question pour améliorer un score.** Une question fausse ou ambiguë est corrigée dans une nouvelle version, avec une entrée dans le CHANGELOG.
- Un score n'a de sens qu'avec la version du dataset qui l'a produit (et la date des documents : RI de mars 2025, statuts du 6 février 2025…).
- La page de sommaire n'est jamais une source attendue : elle contient les mots, pas la réponse.
