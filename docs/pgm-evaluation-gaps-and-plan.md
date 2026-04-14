# CS5340 Project: Evaluation & Experimentation Plan

## Context

The project applies probabilistic graphical models (Bayesian Network, MRF, GMM/EM) to the Cleveland Heart Disease dataset for binary classification. Evaluation is graded on Relevance, Thoroughness, Creativity, Communication, and Peer Review. Current work is solid but has key gaps: no statistical significance testing, incomplete calibration analysis, no failure-mode analysis, missing GMM in the unified evaluation, and some inconsistency between notebooks. The goal of this plan is to round out the evaluation into a cohesive, thorough, and well-communicated story.

---

## Current State Summary

| Model | Accuracy | F1 | AUC | Notes |
|-------|----------|-----|-----|-------|
| Bayesian Network | 0.772 | 0.738 | 0.795 | Poor calibration (ECE=0.165), high log-loss |
| Logistic Regression | 0.835 | 0.813 | 0.905 | Strong baseline |
| Naive Bayes | 0.832 | 0.811 | 0.912 | Best AUC baseline |
| Random Forest | 0.825 | 0.805 | 0.898 | Stable |
| SVM (RBF) | 0.838 | 0.818 | 0.891 | Best Acc/F1 |
| MRF-Tuned | 0.809 | 0.756 | 0.916 | Competitive AUC |
| GMM Classifier | — | — | 0.743 | Uses only 5 continuous features, not yet in unified eval |

---

## Committed Evaluation Scope (from project abstract)

**Metrics** (already partially implemented in `evaluation.ipynb`):
- Classification Accuracy, Log-likelihood, Brier Score, Predictive Entropy, ECE

**Robustness Experiments** (partially missing — top priority to complete):
- Remove key features (e.g., cholesterol `chol`, ECG `restecg`) to simulate incomplete evidence
- Measure posterior variance and entropy changes under missing data
- Compare directed (BN) vs undirected (MRF) behavior under missing data

---

## Plan

### 1. Unify Evaluation in `evaluation.ipynb`

**Goal**: Include ALL models — BN, MRF (baseline + tuned), GMM classifier, and baselines — in one comprehensive comparison.

**Steps**:
- Import/reproduce MRF-tuned and GMM classifier results in `evaluation.ipynb`
- Ensure identical preprocessing (same train/test splits via fixed `random_state`, same 5-fold stratified CV)
- Compute the full metric set for each model: Accuracy, Precision, Recall, F1, AUC, Log-loss, Brier Score, ECE
- Generate unified comparison: bar charts (mean ± std), ROC curves on single plot, calibration curves

**Files to modify**: `evaluation.ipynb`

---

### 2. Calibration Analysis & Post-hoc Calibration

**Goal**: Assess and improve probability calibration (important for medical use). This addresses the "uncertainty modeling" relevance criterion directly.

**Steps**:
- Plot calibration curves (reliability diagrams) for all models
- Report ECE for each model
- Apply Platt scaling (logistic regression on top) to the Bayesian Network (ECE=0.165 → target < 0.05)
- Compare calibrated vs uncalibrated BN
- **Justify**: In clinical settings, well-calibrated probabilities are essential for risk communication

**New section in**: `evaluation.ipynb`

---

### 3. Robustness Experiments — Missing/Incomplete Evidence (FROM ABSTRACT)

**Goal**: Fulfil the core robustness promise from the abstract. This is a major gap in current work. Focus on **BN vs MRF** to tell the directed vs undirected story clearly.

**Steps**:
- Define 3 ablation scenarios:
  - **Scenario A**: Remove `chol` (cholesterol) — weakly correlated feature
  - **Scenario B**: Remove `restecg` (ECG result) — clinical diagnostic feature
  - **Scenario C**: Remove top-2 predictors (`ca`, `thal`) — strongest features
- For each scenario, for **BN and MRF only**:
  - Re-run inference with the removed feature(s) treated as unobserved
  - Measure: Accuracy, AUC, Brier Score, Predictive Entropy (should increase), ECE
- **Directed vs undirected comparison** (BN vs MRF):
  - Under `ca` removal: BN propagates evidence only along DAG paths; MRF distributes via undirected neighbors
  - Plot: per-patient entropy before vs after feature removal for both models
  - Key question: Does MRF maintain better calibration under missing evidence than BN?
- Visualize: bar chart of AUC degradation (full features → ablated) per model

**Files to modify**: `evaluation.ipynb` (new section), pull code from `bayesian_network.ipynb` and `markov_random_field.ipynb`

---

### 4. Statistical Significance Testing

**Goal**: Determine if differences between models are statistically significant (thoroughness criterion).

**Steps**:
- Apply McNemar's test for pairwise comparisons of classification errors between top models (BN vs MRF, MRF vs NB, etc.)
- Report p-values; note which differences are significant at α=0.05
- Add 95% CI on AUC using bootstrap (500 samples)

**New section in**: `evaluation.ipynb`

---

### 5. Failure Mode / Error Analysis

**Goal**: Understand which patients are misclassified and why (creativity + thoroughness).

**Steps**:
- Identify patients misclassified by all or most models ("hard cases")
- Identify patients correctly classified by graphical models (BN/MRF) but not discriminative baselines (or vice versa)
- Analyze patterns: Are hard cases older? Do they have atypical symptom combinations?
- Subgroup analysis: performance stratified by sex (feature `sex`) and age group
- Report: confusion matrix breakdown, false positive vs false negative profiles

**New section in**: `evaluation.ipynb`

---

### 6. Feature Importance & Ablation Study

**Goal**: Understand what drives each model's predictions (interpretability, thoroughness).

**Steps**:
- **MRF**: Plot edge factor strengths (MI weights); rank features by unary factor discriminativeness
- **BN**: Compute mutual information between each node and target; rank features
- **GMM**: Compare cluster centroids; highlight which features differ most between clusters
- **Baselines**: Feature importance from Random Forest; coefficients from Logistic Regression
- Ablation: Remove top-2 features (ca, thal) and measure AUC drop across models — quantifies feature dependency

**Files to modify**: `evaluation.ipynb`, potentially `markov_random_field.ipynb`

---

### 7. GMM Classifier

GMM will be included in the unified evaluation as-is (AUC=0.743) with a clear note that it uses only 5 continuous features (excludes discrete `ca` and `thal`), explaining the lower performance. This is a legitimate and informative finding.

---

### 8. Clinical Threshold Analysis

**Goal**: Select optimal decision thresholds for clinical use (addresses real-world applicability).

**Steps**:
- Compute precision-recall curves alongside ROC for top models
- Identify optimal threshold by F1, by recall ≥ 0.90 (minimize false negatives in medical context), and by Youden index
- Report trade-offs: at recall=0.90, what is precision? Compare models at this operating point
- Brief discussion of cost asymmetry (missing a true positive = missed diagnosis)

**New section in**: `evaluation.ipynb`

---

## Prioritized Order

| Priority | Task | Grading Impact | Effort |
|----------|------|--------|--------|
| **High** | 1. Unify evaluation (all models, all metrics) | Thoroughness | Low |
| **High** | 2. Robustness experiments (BN vs MRF, missing evidence) | Relevance + Thoroughness | Medium |
| **High** | 3. Calibration analysis + Platt scaling for BN | Relevance + Thoroughness | Medium |
| **Medium** | 4. Statistical significance testing (McNemar's, bootstrap CI) | Thoroughness | Low |
| **Medium** | 5. Failure mode / error analysis | Thoroughness + Creativity | Medium |
| **Medium** | 6. Feature importance (MI ranking, RF importance, LR coefficients) | Thoroughness + Communication | Medium |
| **Low** | 7. Clinical threshold analysis (precision-recall curves) | Thoroughness | Low |

---

## Critical Files

- `evaluation.ipynb` — main evaluation notebook (primary target)
- `markov_random_field.ipynb` — MRF implementation (extract best model for unified eval)
- `gmm_em.ipynb` — GMM classifier (improve or port to evaluation)
- `bayesian_network.ipynb` — BN implementation (calibration target)
- `heart_disease_cleaned.csv` — cleaned dataset used across all notebooks

---

## Verification Checklist

1. **Unified eval**: All 7+ models appear in bar charts with error bars; single ROC plot with all curves
2. **Calibration**: Reliability diagrams show Platt-calibrated BN hugging the diagonal better than uncalibrated
3. **Significance tests**: McNemar's p-values table with BH-corrected α
4. **Error analysis**: Section showing "consensus hard cases" and their feature profiles
5. **Feature importance**: Ranked bar chart per model; ablation table showing AUC with/without top-2 features
