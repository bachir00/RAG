# Projet RAG — Notebook "bachir.ipynb"

Ce dépôt contient un notebook Jupyter (``bachir.ipynb``) qui illustre un workflow complet pour :

- l'extraction de texte depuis des PDF,
- la segmentation (chunking) et la génération automatique de paires Question/Réponse avec des LLMs locaux (via `ollama`),
- la comparaison d'outputs de plusieurs LLMs,
- l'évaluation automatique avec BERTScore,
- la mise en place d'un pipeline RAG (Simple avec ChromaDB et Avancé avec BM25) et
- une démonstration basique de fine-tuning (préparation des données et entraînement d'un Flan-T5).

## Fichier principal

- Notebook : `bachir.ipynb` (chemin : racine du dossier)

## Structure (principaux fichiers et dossiers générés)

- `bachir.ipynb` — Notebook central contenant tout le code et les étapes.
- `sortie.txt` — fichier de sortie contenant le texte extrait de l'ensemble des PDFs (généré par la cellule d'extraction).
- `dataset_llama2.txt` — dataset Q/R généré automatiquement (extractions via LLMs).
- `dataset_llama2.json` — version JSON (utile pour fine-tuning).
- `llm_comparaison.txt`, `llm_comparaison2.txt` — sorties de comparaison entre modèles.
- `resultats_llm.txt` — synthèse des résultats (moyennes, etc.).
- `bert_scores_llm_ragSimple_ragAvancee.csv` — résultats BERTScore pour les différentes méthodes.
- `chroma_db/` (ou `./chroma_db`) — base Chroma persistante créée par le notebook (index des chunks).
- `flan-t5-qa/` — répertoire de modèle sauvegardé si le fine-tuning est exécuté.

## Prérequis

- OS : Windows (instructions PowerShell fournies ci-dessous).
- Python 3.8+ (recommandé 3.9 / 3.10+).
- Accès à un noyau Jupyter / VS Code avec extension Jupyter.
- (Optionnel mais requis pour exécution des LLMs locaux) Installation et configuration d'Ollama si vous voulez utiliser `ollama.chat`.
- (Optionnel) Token Hugging Face si vous souhaitez pousser/charger des modèles depuis `huggingface.co`.

Pour faciliter l'installation des dépendances, un fichier `requirements.txt` est fourni.

## Installation (exemple PowerShell)

1. Créer et activer un environnement virtuel (PowerShell) :

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Mettre à jour pip et installer les dépendances :

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

Remarque : certaines bibliothèques listées peuvent nécessiter des composants système supplémentaires (CUDA pour accélération GPU, compilateurs, etc.).

## Exécution

1. Ouvrez `bachir.ipynb` dans VS Code ou Jupyter Notebook / Lab.
2. Exécutez les cellules séquentiellement. Beaucoup de cellules réalisent des installations (cellules commençant par `%pip install`) — vous pouvez les commenter si vous avez déjà installé les paquets.
3. Les étapes principales à exécuter dans l'ordre :

- Installation / Imports
- Extraction des PDF vers `sortie.txt`
- Chunking du texte extrait
- Génération de paires Q/R (via `ollama`) et écriture dans `dataset_llama2.txt`
- Comparaison des modèles (génère `llm_comparaison.txt`)
- Calcul BERTScore et sauvegarde dans `resultats_llm.txt` et `bert_scores_llm_ragSimple_ragAvancee.csv`
- Construction d'un index ChromaDB (RAG Simple)
- Requête / récupération de contexte et génération via LLMs (RAG Simple et RAG Avancé)
- Conversion dataset -> JSON et fine-tuning Flan-T5 (optionnel)

## Description détaillée des sections du notebook

- "Install" : cellules qui installent les dépendances via `%pip install` (utile pour démarrer rapidement dans un notebook).
- "Les 30 premiers documents" : extraction du texte depuis un dossier `./dataset` (PDF) vers `sortie.txt` en concaténant le contenu de chaque PDF.
- "Q/R Chunking" : segmentation du fichier `sortie.txt` en chunks (taille configurable) et génération de prompts pour obtenir jusqu'à 3 questions/réponses par chunk via `ollama.chat`.
- "Comparaison LLMs" : envoie les questions extraites à plusieurs modèles (`llama2`, `mistral`, `gemma`) et enregistre les réponses dans `llm_comparaison.txt`.
- "BertScore" : calcule BERTScore entre les références (dataset) et les réponses des différents modèles, puis sauvegarde les résultats.
- "RAG Simple" : création d'une collection ChromaDB, découpage en tokens avec `tiktoken`, indexation des chunks, puis recherche de contexte via `collection.query`.
- "RAG Avancé" : combinaison embeddings (sentence-transformers) et re-ranking BM25 (rank_bm25) pour améliorer la pertinence des contextes récupérés.
- "Étude comparative" : calcule et sauvegarde les scores BERTScore pour les 3 approches (LLM simple, LLM+RAG simple, LLM+RAG avancé) dans un fichier CSV.
- "RAG + fine tuning" : conversion du dataset textuel en JSON de paires Q/R, création d'un dataset Hugging Face, tokenisation et entraînement d'un Flan-T5 (exemples fournis dans le notebook).

## Variables et fichiers importants à vérifier

- `dossier_pdf` (dans le notebook) — chemin vers les PDF à extraire. Par défaut `./dataset`.
- `sortie.txt` — fichier agrégé d'extraction (vérifier présence et encodage UTF-8).
- `dataset_llama2.txt` / `dataset_llama2.json` — outputs utilisés comme références.
- `chroma_db/` — dossier database Chroma (ne pas supprimer si vous voulez réutiliser l'index).

## Conseils et points d'attention

- Ollama : les appels `ollama.chat(...)` supposent qu'un runtime Ollama local est installé et accessible. Si vous n'avez pas Ollama, commentez ces cellules ou adaptez-les à l'API LLM de votre choix.
- Versions : certaines bibliothèques (p. ex. `bert-score`, `chromadb`, `sentence-transformers`) évoluent vite. Si vous avez des erreurs, essayez d'utiliser les versions proches de l'époque de création du notebook.
- Performance : l'encodage et les embeddings peuvent être lents sur CPU. Pour de grandes bases, utilisez GPU et ajustez `batch_size`.
- Encodage : veillez à utiliser `encoding='utf-8'` lors de la lecture/écriture de fichiers pour éviter des erreurs de décodage.

## Limitations et hypothèses

- Le notebook suppose que les données de référence et les sorties des modèles partagent les mêmes clés de questions (alignement Q -> R). Si ce n'est pas le cas, il y a des filtres dans le notebook pour réaligner mais il faut vérifier manuellement.
- Le pipeline de fine-tuning est démonstratif (Flan-T5 small) et adapté à un petit jeu de données. Pour de vraies tâches, prévoir validation, test, et réglages d'hyperparamètres.

## Prochaines étapes recommandées

1. Extraire un sous-ensemble de documents et exécuter le pipeline complet pour valider les étapes.
2. Ajouter un `requirements.txt` (fourni) et/ou `environment.yml` pour reproductibilité (GPU vs CPU).
3. Extraire les fonctions réutilisables du notebook en scripts Python (`src/`) pour pouvoir automatiser et tester.
4. Ajouter des tests unitaires légers pour la lecture/parse des datasets et la conversion en JSON.

## Licence & contact

