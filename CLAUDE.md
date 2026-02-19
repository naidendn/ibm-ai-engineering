# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python implementations from IBM's **Machine Learning with Python** course (part of the IBM AI Engineering Professional Certificate). Scripts are self-contained exercises covering supervised and unsupervised ML algorithms using scikit-learn.

## Running Scripts

Each script is a standalone executable. Datasets are fetched automatically from IBM Cloud Object Storage — no local data files needed:

```bash
python scripts/<script_name>.py
```

There are no build, test, or lint automation setups in this repo.

## Architecture

```
scripts/     # Standalone Python scripts, one per algorithm/topic
cheatsheets/ # Reference PDFs (regression, supervised, unsupervised learning)
```

### Script Patterns

- Scripts fetch data via URL (IBM Cloud Object Storage: `cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud`)
- Visualization with `matplotlib` and `plotly`; `plt.show()` calls are often commented out during development
- `umap` is imported as `import umap.umap_ as UMAP` (not the standard `import umap`)

### Algorithm Coverage

| Category | Scripts |
|---|---|
| Regression | `simple_linear_regression.py`, `multiple_linear_regression.py`, `regression_trees.py` |
| Classification | `logistic_regression.py`, `knn.py`, `decision_trees.py`, `support_vector_machine.py`, `multi_class_classification.py` |
| Ensemble | `random_forrests_xgboost.py` |
| Clustering | `k_means_clustering.py`, `k_means_clustering_real_data.py`, `dbscan_hdbscan.py` |
| Dimensionality Reduction | `pca.py`, `pca_iris.py`, `tsne_umap.py` |
| Pipeline | `obesity_risk_pipeline.py` |

## Key Dependencies

- `scikit-learn` — core ML algorithms and pipelines
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `plotly`, `seaborn` — visualization
- `xgboost` — gradient boosting
- `umap-learn` — UMAP dimensionality reduction
- `hdbscan` — density-based clustering
