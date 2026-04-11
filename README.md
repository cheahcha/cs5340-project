# Heart Disease Risk Modeling with Probabilistic Graphical Models

CS5340 (Uncertainty Modelling in AI) course project. We apply Bayesian Networks, Markov Random Fields, and Gaussian Mixture Models to the [Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) to explore how uncertainty is represented and propagated in directed vs undirected graphical models.

## Dataset

303 patients, 13 clinical features (age, sex, chest pain type, resting BP, cholesterol, fasting glucose, ECG results, max heart rate, exercise-induced angina, ST depression, slope, major vessels, thalassemia), binary target (0 = healthy, 1 = disease present). 6 missing values imputed by mode.

## Models & Results

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| Bayesian Network (base) | 0.772 | 0.738 | 0.795 |
| MRF (tuned: α=10.0, γ=1.5) | 0.809 | 0.756 | 0.916 |
| GMM generative classifier | — | — | 0.743 |
| Logistic Regression | 0.835 | 0.813 | 0.905 |
| Naive Bayes | 0.832 | 0.811 | 0.912 |
| Random Forest | 0.825 | 0.805 | 0.898 |
| SVM (RBF) | 0.838 | 0.818 | 0.891 |

All results are 5-fold stratified cross-validation means.

## Notebooks

Run in this order:

| Notebook | Contents |
|----------|----------|
| `eda.ipynb` | Exploratory analysis, correlation, missing value handling |
| `gmm_em.ipynb` | EM-based imputation, GMM from scratch, unsupervised clustering, generative classifier |
| `bayesian_network.ipynb` | Fixed 3-layer DAG, MLE parameter learning, Variable Elimination inference |
| `markov_random_field.ipynb` | Undirected graph, Belief Propagation, MI-reweighted potentials, hyperparameter search |
| `evaluation.ipynb` | Unified comparison of all models across 9 metrics |

`robustness_utils.py` is a shared library used by the robustness experiments. It contains the canonical data preprocessing pipeline, model definitions for all four BN/MRF variants, and helpers for running missing-evidence scenarios.

## Installation

```bash
pip install pgmpy scikit-learn pandas numpy matplotlib seaborn networkx ucimlrepo jupyter
```

## Key Findings

- **MRF-Tuned achieves AUC 0.916**, matching or exceeding all discriminative baselines, while remaining fully interpretable as a probabilistic graphical model.
- **MI-reweighted potentials** (upweighting edge factors by normalised mutual information) improve AUC by +0.004 over the base MRF with no structural changes.
- **GMM unsupervised clustering** (K=2, BIC-optimal) recovers two clinically meaningful patient phenotypes: a high-risk group (55% disease rate) and a lower-risk group (26%), without using labels.
- **Bayesian Network** shows higher fold-level variance than MRF, with AUC ranging 0.69–0.89 across folds, reflecting sensitivity to the fixed DAG structure under small training sets.
- The top predictors across all models are `ca` (major vessels coloured by fluoroscopy), `thal` (thalassemia type), and `oldpeak` (ST depression).
