# Analyse d'erreurs du retriever (dev, dataset v1)

Run : `retriever_baseline_20261019_122033.json`, après correction du départage des ex aequo RRF.
Commandes :
```bash
python -m evaluation.error_analysis                                  # hybrid+rerank20, Hit@1 = 0
python -m evaluation.error_analysis --config hybrid --metric hit@5   # ce que le reranker rattrape
```

## 1. Questions dont la bonne page n'est pas 1ʳᵉ avec `hybrid+rerank20` (6/37)

| id | Rang | Page arrivée 1ʳᵉ | Cause principale | Explication |
|---|---|---|---|---|
| fact-ca-membres | 5 | Statuts p.16 | page concurrente | « conseil d'administration » et « membres » figurent sur presque toutes les pages des Statuts. Le chiffre (« vingt-huit membres », art. 21) se trouve dans un chunk qui commence par la fin de l'article 20 (comité électoral) : c'est aussi un chunk multi-articles. |
| ref-mot-de-passe | 2 | RI p.18 | vocabulaire | « prêter à un ami » contre « cédés ou communiqués à un tiers ». Le reranker place devant un passage du RI sur la fraude, qui est sans rapport. Les 2 bonnes pages restent aux rangs 2 et 3. |
| ref-faille | 3 | RI p.14 | page concurrente | Le RI p.14 parle aussi de fichiers et de données personnelles, mais pas de signaler une faille, ce que fait la Charte SI p.3. |
| ref-plagiat | absente (top 10) | RI p.3 (sommaire) | vocabulaire | La question ne contient ni « plagiat » ni « citation ». Pire : le mot « article » (de revue) fait remonter les « Article N » du **sommaire** et des autres pages du RI. La bonne réponse (art. 26) est en plus dans un chunk qui couvre 3 articles (harcèlement, bizutage, plagiat). |
| part-cotisation | 2 | RI p.10 | page concurrente | Le RI p.10 parle des associations (domiciliation, bureau), sans dire qui peut présider. La 2ᵉ moitié de la question (cotisation) n'a pas de réponse et attire d'autres pages. |
| syn-epreuves-cc | 2 | Guide p.2 | page concurrente | Il s'agit de l'introduction du même guide (contrôle continu, crise sanitaire). Le nombre minimum d'épreuves est en p.3. |

**Comptage des causes principales :** page concurrente **4**, vocabulaire **2**, sommaire 0 (cause secondaire 1 fois), chunk multi-articles 0 (secondaire 2 fois), réécriture 0, erreur du dataset **0**.

## 2. Ce que `hybrid` sans reranker rate en Hit@5 (4/37)
`fact-ca-membres` et `suivi-etudiants-ca` (même page Statuts p.9, absente du top 10), `ref-faille` (rang 7) et `ref-plagiat` (absente).
Le reranker en rattrape **3 sur 4**. Toutes ces erreurs viennent de questions sur les Statuts, un document très répétitif, ou de questions reformulées.

## 3. Conclusions
- **Aucune erreur du dataset** : les pages attendues sont bien les seules qui répondent.
- **Aucune cause dominante que l'on puisse corriger simplement.** Les « pages concurrentes » viennent de l'unité de mesure (la page) et de documents qui répètent le même vocabulaire. Elles restent dans le top 5, donc le LLM reçoit quand même la bonne page.
- **Le seul vrai raté (`ref-plagiat`)** combine les 3 faiblesses connues : un vocabulaire absent, une page de sommaire et un chunk multi-articles. Corriger une seule de ces causes ne suffirait probablement pas.
- **La réécriture n'apparaît pas ici**, alors que « Et l'alcool ? » n'est pas réécrite : la question brute suffit à trouver l'article sur l'alcool. Le défaut du routeur se mesurera sur la **réponse**, pas sur le retriever.

**Décision :** garder `hybrid+rerank20`, qui est déjà la configuration par défaut de `answer()`. Les pistes suivantes vont dans les axes d'amélioration, pour être testées sur dev avec un dataset v2 plus grand :
1. un chunking par article (règle `fact-ca-membres` et `ref-plagiat`) ;
2. l'exclusion des pages de sommaire ;
3. l'enrichissement de la requête (synonymes ou HyDE) pour les questions reformulées.
