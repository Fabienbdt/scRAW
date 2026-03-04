# scraw

Projet autonome et allégé pour entraîner **scRAW** sur fichiers `.h5ad`.

Objectif:
- garder uniquement les composants utiles pour les runs Baron/Pancreas validés,
- reproduire le set d'ablations Deep2 (baseline + ablations triplet/reconstruction + DANN),
- conserver le même format d'exports (métriques, labels, loss, figures).

## Structure

```text
scraw_dedicated/
  pyproject.toml
  requirements.txt
  README.md
  REFERENCE_RUNS.md
  run_scraw_dedicated.py
  run_seed_robustness.py
  run_hyperparam_search.py
  scripts/
    run_single.py
    run_seed_robustness.py
    run_deep2_sweep.py
    run_hyperparam_search.py
    nohup_run_hyperparam_search.sh
  src/scraw_dedicated/
    cli.py
    deep2_sweep.py
    seed_robustness.py
    hyperparam_search.py
    presets.py
    preprocessing.py
    metrics.py
    visualization.py
    algorithms/
      base_autoencoder.py       # base minimal AE
      scraw_losses_and_weights.py
      scraw_clustering.py
      scraw_algorithm.py        # orchestration principale scRAW
      scRAW.py                  # wrapper compat
    core/
      algorithm_registry.py
      config.py
```

## Guide des scripts (quoi lancer et pourquoi)

### Scripts racine (simples à utiliser)

- `run_scraw_dedicated.py`
  - Utilité: lancer **un run unique** (entraînement + métriques + figures).
  - Quand l'utiliser: test d'un preset, run de production, debug d'une config.
  - Entrées: dataset `.h5ad`, preset, dossier de sortie.
  - Sorties: dossier complet `config/`, `data/`, `results/`, `figures/`.

- `run_seed_robustness.py`
  - Utilité: lancer **plusieurs seeds** et mesurer la robustesse.
  - Quand l'utiliser: valider qu'un réglage est stable et pas seulement bon sur une seed.
  - Entrées: dataset `.h5ad`, preset, liste de seeds.
  - Sorties: un dossier par seed + `seed_metrics.csv` + `seed_aggregate.json`.

- `run_hyperparam_search.py`
  - Utilité: lancer une **recherche d'hyperparamètres complète** (baseline + single-param + pairwise + batch correction).
  - Quand l'utiliser: explorer systématiquement les hyperparamètres clés et classer automatiquement les meilleurs runs.
  - Entrées: dataset `.h5ad`, preset, dossier de sortie.
  - Sorties: arborescence `runs/`, logs structurés, `summaries/all_runs_metrics.csv`, `ranked_by_score.csv`, `best_by_group.csv`.

### Scripts dans `scripts/` (wrappers explicites)

- `scripts/run_single.py`
  - Wrapper de `run_scraw_dedicated.py`.
  - Utile si tu veux un point d'entrée scripté clair dans `scripts/`.

- `scripts/run_seed_robustness.py`
  - Wrapper de `run_seed_robustness.py`.
  - Même comportement, orienté exécution batch.

- `scripts/run_deep2_sweep.py`
  - Utilité: rejouer automatiquement les **ablations Deep2** (baseline + ablations).
  - Quand l'utiliser: comparaison systématique des variantes.
  - Entrées: dataset Baron, dossier de sortie sweep, éventuellement `--only`.
  - Sorties: un dossier par expérience + `summary.csv` global.

- `scripts/run_hyperparam_search.py`
  - Wrapper de `run_hyperparam_search.py`.
  - Même logique de recherche complète, pratique pour les exécutions scriptées.

- `scripts/nohup_run_hyperparam_search.sh`
  - Lance la recherche complète en arrière-plan via `nohup`.
  - Force le mode métriques uniquement (`--metrics-only`) pour accélérer les sweeps.
  - Organise les résultats automatiquement dans `results/hparam_search/<preset>_<dataset>_metrics_only_<timestamp>/`.

### Modules internes (`src/scraw_dedicated/`)

- `cli.py`
  - Rôle: pipeline complet d'un run (chargement, preprocessing, entraînement, export).
  - C'est le coeur de l'exécution single-run.

- `deep2_sweep.py`
  - Rôle: orchestration de plusieurs runs d'ablations.
  - Appelle le CLI run par run et agrège les résultats.

- `seed_robustness.py`
  - Rôle: orchestration multi-seeds.
  - Calcule statistiques de stabilité (moyenne, écart-type, min, max).

- `presets.py`
  - Rôle: presets validés (`baron_best`, `pancreas_best`) + paramètres par défaut.

- `preprocessing.py`
  - Rôle: filtres QC, normalisation, log1p, HVG, scaling.

- `metrics.py`
  - Rôle: métriques clustering/classification (NMI, ARI, ACC, RareACC, silhouette, etc.).

- `visualization.py`
  - Rôle: génération de toutes les figures (UMAP final, batch, pondéré, évolution, losses).

### Sous-modules algorithmiques (`src/scraw_dedicated/algorithms/`)

- `base_autoencoder.py`
  - Rôle: briques de base du réseau (AE MLP, encodage, gradient reversal).

- `scraw_losses_and_weights.py`
  - Rôle: reconstruction NB/MSE, pondération dynamique, triplet rare.

- `scraw_clustering.py`
  - Rôle: pseudo-labels Leiden/KMeans et clustering final HDBSCAN.

- `scraw_algorithm.py`
  - Rôle: orchestration complète de l'entraînement scRAW (fit/predict).

- `scRAW.py`
  - Rôle: wrapper de compatibilité pour anciens imports.

## Workflow recommandé

1. `run_scraw_dedicated.py` pour valider un run unique.
2. `run_seed_robustness.py` pour vérifier la stabilité inter-seeds.
3. `scripts/run_deep2_sweep.py` pour comparer plusieurs ablations sur un même dataset.
4. `scripts/nohup_run_hyperparam_search.sh` pour une exploration hyperparamètres complète en long run.

## Ce qui a été simplifié

Le coeur `algorithms/scRAW.py` ne garde que les chemins réellement utilisés dans les ablations:
- autoencoder 2 phases (warm-up puis weighted),
- reconstruction `nb` ou `mse`,
- pseudo-labels `leiden` (fallback unique `kmeans`),
- pondération cellule = fréquence cluster + densité latente,
- rare loss `triplet`,
- DANN adversarial optionnel,
- clustering final `hdbscan`,
- snapshots latents pour UMAP evolution.



## Installation

Option recommandée:

```bash
cd /Users/fabienbidet/Documents/MASTER\ 2/STAGE/scraw_dedicated
python -m pip install -e .
```

Option sans installation:
- utiliser directement les wrappers `run_*.py` et `scripts/run_*.py`.

## Exécution

### 1) Run unique

```bash
cd /Users/fabienbidet/Documents/MASTER\ 2/STAGE/scraw_dedicated
python run_scraw_dedicated.py \
  --preset baron_best \
  --data /chemin/dataset.h5ad \
  --output /chemin/output_run \
  --device cpu
```

### 2) Robustesse multi-seeds

```bash
python run_seed_robustness.py \
  --preset baron_best \
  --data /chemin/dataset.h5ad \
  --output-root /chemin/output_seeds \
  --seeds 11,22,33,42,77 \
  --metrics-only
```

### 3) Sweep Deep2 (ablations)

```bash
python scripts/run_deep2_sweep.py \
  --data /chemin/baron_human_pancreas.h5ad \
  --output-root /chemin/output_deep2
```

Limiter à quelques configs:

```bash
python scripts/run_deep2_sweep.py \
  --data /chemin/baron_human_pancreas.h5ad \
  --output-root /chemin/output_deep2 \
  --only baseline_trip10_s35,baseline_true_unsupervised,ablate_triplet,ablate_reco_cluster,ablate_reco_density
```

### 4) Recherche hyperparamètres complète (métriques uniquement)

Exécution directe:

```bash
python run_hyperparam_search.py \
  --preset baron_best \
  --data /chemin/baron_human_pancreas.h5ad \
  --output-root /chemin/output_hparam_search \
  --device cpu
```

Exécution en arrière-plan `nohup` (recommandé pour runs longs):

```bash
bash scripts/nohup_run_hyperparam_search.sh
```

Exemple avec variables explicites:

```bash
PRESET=baron_best \
DATA_PATH=/Users/fabienbidet/Documents/MASTER\ 2/STAGE/SCRBenchmark/data/baron_human_pancreas.h5ad \
DEVICE=cpu \
  SEARCH_GROUPS=baseline,single,pairwise,batch \
MAX_RUNS=0 \
bash scripts/nohup_run_hyperparam_search.sh
```

## Presets

- `baron_best`: preset Baron validé (baseline trip10_s35)
- `pancreas_best`: preset Pancreas validé (inclut DANN)

Les overrides restent disponibles via:
- `--param KEY=VALUE` (algo)
- `--preprocess KEY=VALUE` (préprocessing)

## Sorties générées

Pour chaque run:
- `config/config_used.json`
- `config/algorithm_hyperparams_used.json`
- `data/processed.h5ad`
- `results/analysis_results.csv`
- `results/results.csv`
- `results/results.json`
- `results/labels/labels_scraw_run0.csv`
- `results/loss_history/loss_scraw_run0.json`
- `figures/*.png` (sauf si `--metrics-only`)

## Vérification rapide

```bash
python scripts/run_single.py --help
python scripts/run_seed_robustness.py --help
python scripts/run_deep2_sweep.py --help
python scripts/run_hyperparam_search.py --help
```
