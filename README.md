# scRAW

Version autonome centrée sur le coeur de l'algorithme scRAW.

Ce dépôt conserve uniquement les composants nécessaires pour exécuter un run complet (préprocessing, entraînement, clustering final, métriques, figures), sans les scripts d'orchestration de sweeps/ablations/recherches hyperparamètres.

## Structure conservée

- `src/scraw_dedicated/cli.py` : point d'entrée principal.
- `src/scraw_dedicated/algorithms/` : coeur modèle/loss/clustering.
- `src/scraw_dedicated/preprocessing.py` : pipeline preprocessing.
- `src/scraw_dedicated/metrics.py` : calcul des métriques.
- `src/scraw_dedicated/visualization.py` : génération des figures.
- `src/scraw_dedicated/presets.py` : presets de configuration.
- `run_scraw_dedicated.py` : wrapper local simple vers le CLI.

## Installation

```bash
cd /Users/fabienbidet/Documents/MASTER\ 2/STAGE/scRAW
python -m pip install -e .
```

## Lancement rapide

Commande minimale:

```bash
scraw-run \
  --data data/baron_human_pancreas.h5ad \
  --output results/demo_run
```

Cette commande:

- utilise automatiquement le preset `default`
- choisit automatiquement le device (`--device auto`)
- lance le preprocessing, l'entraînement, le clustering final, le calcul des métriques et la génération des figures

Si tu ne veux pas installer la commande `scraw-run`, tu peux lancer exactement la même run avec le wrapper local:

```bash
python run_scraw_dedicated.py \
  --data data/baron_human_pancreas.h5ad \
  --output results/demo_run
```

## Commande standard

Option 1 (commande installée):

```bash
scraw-run \
  --preset default \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --device cpu
```

Option 2 (wrapper local):

```bash
python run_scraw_dedicated.py \
  --preset default \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --device cpu
```

Si `--preset` est omis, la CLI utilise désormais le preset `default`.

## Arguments utiles

- `--data` : chemin vers le fichier `.h5ad` d'entrée
- `--output` : dossier de sortie de la run
- `--device auto|cpu|cuda|mps` : choix du device, `auto` par défaut
- `--preset default|baron_best|pancreas_best` : configuration de départ
- `--verbose` : affiche les logs détaillés pendant l'exécution

## Cas fréquents

Dataset sans colonne batch compatible avec DANN:

```bash
scraw-run \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --dann off
```

Dataset multi-batch avec une colonne batch qui ne s'appelle pas `batch`:

```bash
scraw-run \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --batch-key study
```

Run plus lisible dans le terminal:

```bash
scraw-run \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --verbose
```

## Sorties

Le run génère notamment:

- `config/config_used.json`
- `config/algorithm_hyperparams_used.json`
- `results/results.json`
- `results/analysis_results.csv`
- `results/clustering_final/*`
- `results/loss_history/*`
- `figures/*`
