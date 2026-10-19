# Changelog du dataset d'évaluation

## v1 (gelée)

Première version : 68 questions (62 manuelles, 6 synthétiques), découpage 47 dev / 21 test.

Reprise des questions des anciens scripts, avec les corrections suivantes :
- « Que faire si un enseignant se moque de moi sans arrêt ? » : la page 3 du règlement intérieur (sommaire) a été retirée des sources attendues ; sources : RI p.6 et p.17.
- « Article 11 accès aux locaux » : ajout de la page 12 du RI, où l'article se poursuit.
- « Quelles sont les démarches pour faire un stage ? » : classée `partielle` (le guide des MCC traite des stages uniquement dans le contexte de la crise sanitaire), et non `non_couverte`.
- Les 5 questions « dans le corpus » de `out_of_scope_questions.py` n'avaient ni sources ni réponse de référence : elles ont été complétées.

## À prévoir pour la v2

Relevé pendant l'évaluation du retriever et des réponses (la v1 reste inchangée pour que les runs restent comparables) :
- Faits clés trop stricts : `proscrire` ne reconnaît pas « proscrit » / « proscrites » (`fact-alim-bouteilles`, `suivi-bouteilles`). Ajouter des alternatives (`proscrire|proscrit`).
- Agrandir les catégories (6 questions par catégorie en dev, 1 à 3 en test) pour réduire la marge d'incertitude.
