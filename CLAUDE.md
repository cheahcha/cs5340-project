# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS5340 course project: probabilistic graphical models (Bayesian Network, MRF, GMM/EM) applied to the Cleveland Heart Disease dataset (303 patients, 13 clinical features, binary target `num`). The goal is both classification and uncertainty analysis under missing evidence.

## Running notebooks

```bash
jupyter notebook          # launch Jupyter; open any .ipynb
jupyter nbconvert --to notebook --execute <notebook>.ipynb   # run headlessly
```

There is no build system, test suite, or linter configured.

## Architecture

### Shared utility module: `robustness_utils.py`

This is the central library imported by the robustness experiments notebook. It contains everything needed to reproduce the BN/MRF pipeline:

- **`load_discretized_heart_data(csv_path)`** — canonical preprocessing: loads `heart_disease_cleaned.csv` and applies all discretization bins. This is the single source of truth for binning thresholds; do not redefine them elsewhere.
- **`get_model_specs()`** — returns the 4 models as `ModelSpec` objects: `Base BN`, `Best BN`, `Base MRF`, `Best MRF`. Each spec holds a `build_infer` factory that takes a training `DataFrame` and returns a pgmpy `VariableElimination` or `BeliefPropagation` object.
- **`run_full_baseline(df_disc)`** — 5-fold stratified CV on all 4 models with full features; returns a DataFrame of fold-level metrics.
- **`run_missing_scenarios(df_disc, scenarios, experiment_name)`** — same CV loop but with feature subsets removed, computing both absolute and relative-to-full metrics.
- **`predict_row` / `predict_dataset`** — inference wrappers; fall back to 0.5 on any pgmpy exception.
- Metric helpers: `summarize_predictions`, `expected_calibration_error`, `binary_entropy`, `binary_variance`.
- Plot helpers: `plot_single_feature_heatmap`, `plot_group_bars`, `plot_within_family_*`.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `eda.ipynb` | Exploratory analysis, correlation, outliers, mode imputation, saves `heart_disease_cleaned.csv` |
| `gmm_em.ipynb` | EM imputation, GMM from scratch (verified vs sklearn, ARI=1.0), unsupervised clustering (K=2 by BIC), GMM as generative classifier (AUC=0.743, continuous features only) |
| `bayesian_network.ipynb` | Fixed 3-layer DAG (risk factors → disease → symptoms), MLE parameter learning via pgmpy, Variable Elimination inference, 5-fold CV |
| `markov_random_field.ipynb` | Undirected 3-layer graph, Belief Propagation, MI-reweighted potentials, hyperparameter search (126 configs), best: alpha=10.0 gamma=1.5 AUC=0.916 |
| `evaluation.ipynb` | Unified comparison of BN, LR, Naive Bayes, RF, SVM across 9 metrics with 5-fold CV |

### Data flow

```
heart_disease_cleaned.csv
    └─► load_discretized_heart_data()   ← canonical discretization in robustness_utils.py
            └─► ModelSpec.build_infer(train_df)
                    └─► predict_dataset(infer, test_df, observed_cols)
                            └─► summarize_predictions(y_true, probs)
```

### Key constants (`robustness_utils.py`)

- `STATE_NAMES` — discrete state cardinalities for all 14 variables (must match discretization bins)
- `BASE_EDGES` — 13-edge star topology: each risk factor → `num`, `num` → each symptom
- `BEST_BN_EDGES` — 17-edge hill-climbing learned DAG (verified acyclic via assert)
- `ALL_SINGLE_FEATURE_REMOVAL` / `CLINICAL_GROUP_REMOVAL` — experiment scenario definitions
- `RISK_FACTORS` = `["age", "sex", "fbs", "chol", "trestbps"]`
- `SYMPTOMS` = `["cp", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "restecg"]`

### Discretization thresholds

| Feature | Bins / Mapping |
|---------|---------------|
| `age` | 0–45 → 0, 45–60 → 1, 60+ → 2 |
| `trestbps` | < 130 → 0, ≥ 130 → 1 |
| `chol` | ≤ 199 → 0, 200–240 → 1, > 240 → 2 |
| `thalach` | ≤ 119 → 0, 120–150 → 1, > 150 → 2 |
| `oldpeak` | 0–1 → 0, 1–2 → 1, > 2 → 2 |
| `cp` | remapped 1-indexed → 0-indexed (subtract 1) |
| `slope` | remapped 1-indexed → 0-indexed (subtract 1) |
| `thal` | {3→0, 6→1, 7→2} |
| `ca`, `sex`, `fbs`, `restecg`, `exang` | rounded float → int |

## Key dependencies

- `pgmpy` — BN and MRF models, Variable Elimination, Belief Propagation
- `scikit-learn` — baselines, StratifiedKFold, metrics
- `ucimlrepo` — dataset download (used in `eda.ipynb`)
- `networkx` — DAG validation and visualization

## Docs

- `docs/evaluation-plan.md` — planned experiments and prioritised task list
- `docs/notebook-review.md` — per-notebook review with specific improvement items
