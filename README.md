# Heart Disease Risk Modeling with Probabilistic Graphical Models

CS5340 (Uncertainty Modelling in AI) course project. We apply Bayesian Networks, Markov Random Fields, and Gaussian Mixture Models to the [Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) to explore how uncertainty is represented and propagated in directed vs undirected graphical models.

## Dataset

303 patients, 13 clinical features (age, sex, chest pain type, resting BP, cholesterol, fasting glucose, ECG results, max heart rate, exercise-induced angina, ST depression, slope, major vessels, thalassemia), binary target (0 = healthy, 1 = disease). 6 missing values imputed via EM. ~46% positive rate.

## Models & Results

| Model | Accuracy | F1 | AUC | ECE |
|-------|----------|----|-----|-----|
| BN (MLE) | 0.772 | 0.738 | 0.795 | 0.165 |
| BN (BDeu) | 0.812 | 0.786 | 0.896 | 0.143 |
| BN (BDeu + Enriched DAG) | 0.812 | 0.791 | 0.898 | 0.142 |
| BN (Hill-Climbing + BDeu) | 0.845 | 0.820 | **0.916** | — |
| MRF (Tuned: α=10.0, γ=1.5) | 0.809 | 0.756 | **0.916** | 0.170 |
| GMM generative classifier | — | — | 0.743 | — |
| Logistic Regression | 0.835 | 0.813 | 0.905 | 0.109 |
| Naive Bayes | 0.832 | 0.811 | 0.912 | 0.129 |
| Random Forest | 0.825 | 0.805 | 0.898 | 0.112 |
| **SVM (RBF)** | **0.838** | **0.818** | 0.891 | **0.096** |

All results are 5-fold stratified cross-validation means. Full details in [`docs/findings-summary.md`](docs/findings-summary.md).

## Notebooks

Run in this order:

| Notebook | Contents |
|----------|----------|
| `eda.ipynb` | Exploratory analysis, correlation, missing value handling, saves `heart_disease_cleaned.csv` |
| `gmm_em.ipynb` | EM-based imputation, GMM from scratch, unsupervised clustering (K=2, BIC), generative classifier |
| `bayesian_network.ipynb` | Fixed 3-layer DAG, MLE → BDeu → enriched DAG → hill-climbing structure learning |
| `markov_random_field.ipynb` | Undirected graph, Belief Propagation, MI-reweighted potentials, hyperparameter search (126 configs) |
| `evaluation.ipynb` | Unified comparison across 9 metrics, calibration analysis, McNemar significance testing |
| `robustness_experiments.ipynb` | Missing-evidence scenarios: single-feature and clinical group removal |

`robustness_utils.py` is a shared library used by `robustness_experiments.ipynb`. It contains the canonical data preprocessing pipeline, all four BN/MRF model definitions, and helpers for running CV and missing-evidence scenarios.

## Installation

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Findings

- **MRF-Tuned and BN-HC achieve AUC 0.916**, matching or exceeding all discriminative baselines while remaining interpretable as probabilistic graphical models.
- **BN (MLE) collapses** due to zero-count CPT entries; switching to a BDeu prior alone recovers +0.10 AUC.
- **SVM is best calibrated** (ECE=0.096); all PGMs need Platt scaling before probabilities can be used as clinical risk scores.
- **`ca` (major vessels) is the single most critical feature** — removing it causes the largest AUC drop (~0.025) across all models.
- **MRF is marginally more robust to missing evidence** than BN: its undirected structure compensates via alternative paths when a node is withheld.
- **GMM clustering** (K=2) recovers two clinically coherent patient subgroups without using labels: older/high-risk (55% disease) vs younger/lower-risk (26% disease).
- After Bonferroni correction, **no pairwise model difference is statistically significant** on n=303 — the dataset is too small to claim superiority beyond BN-MLE vs the rest.
