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

## Exécution

Option 1 (commande installée):

```bash
scraw-run \
  --preset baron_best \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --device cpu
```

Option 2 (wrapper local):

```bash
python run_scraw_dedicated.py \
  --preset baron_best \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --device cpu
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
