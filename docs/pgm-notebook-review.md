# CS5340 Project: Notebook Review & Improvement Guide

This document summarises findings from a thorough review of all project notebooks, with specific, actionable improvements for the final report/submission.

---

## Overall Health Summary

| Notebook | Flow | Visuals | Explanations | Completeness | Priority Issues |
|----------|------|---------|--------------|--------------|-----------------|
| `eda.ipynb` | Good, messy end | Excellent | Excellent | 85% | Remove "Older stuff" section; fill empty cells; add final summary |
| `evaluation.ipynb` | Excellent | Excellent | Good | 80% | Add significance tests; explain fold variance; add recommendation |
| `bayesian_network.ipynb` | Clear | Poor (no graph) | Good | 85% | Add network visualization; show learned CPTs |
| `markov_random_field.ipynb` | Excellent | Excellent | Excellent | 90% | Show edge MI values; verify BP convergence; add final recommendation |
| `gmm_em.ipynb` | Very good | Excellent | Excellent | 75% | Explain GMM classifier handicap; fix out-of-range imputation discussion |
| `README.md` | — | — | — | 5% | Complete rewrite needed |

---

## eda.ipynb

### What Works Well
- Clear narrative: data loading → summary → missing values → correlations → visualizations
- Strong clinical context in markdown (e.g., links ST depression to coronary artery disease)
- Good feature-level insights: identifies `ca`, `thal`, `oldpeak`, `thalach` as top 4 predictors

### Issues to Fix

**1. Remove the "Older stuff" section**
- Cells from `id="49c291e2"` onwards contain redundant exploratory code duplicating earlier analyses
- Delete or move to an appendix

**2. Fill the empty markdown cell (`id="3cc30b87"`)**
- Currently blank; should contain a section summary or transition

**3. Fix duplicate "Section 4" headers**
- There are two `## 4.` headers (Outlier Detection and Summary Statistics)
- Renumber sections consistently

**4. Add a final summary cell**
- The notebook ends abruptly with no conclusion
- Add a summary table stating explicitly:
  - Features retained downstream: all 13
  - Rows retained: 303 (6 missing values imputed by mode)
  - Discretization strategy: to be applied in BN/MRF notebooks
  - Top 4 predictors for downstream focus: `ca`, `thal`, `oldpeak`, `thalach`

**5. Justify mode imputation**
- States "we can use mode imputation" without justification
- Add 1-2 sentences: only 6 missing values across 303 rows (<2%), so simple imputation is appropriate; EM-based imputation is explored in `gmm_em.ipynb`

---

## evaluation.ipynb

### What Works Well
- 9-metric evaluation suite covering classification, calibration, and uncertainty (Accuracy, Precision, Recall, F1, AUC, Log-loss, Brier Score, Entropy, ECE)
- Rigorous 5-fold stratified CV with fixed random seed (42)
- Clear bar charts with error bars and per-fold ROC curves

### Issues to Fix

**1. Add statistical significance testing** *(high priority)*
- Currently reports mean ± std with no p-values
- Add McNemar's test for pairwise model comparisons on the same fold predictions
- Add bootstrap 95% CI on AUC (500 samples)

**2. Explain fold-level variance for BN**
- BN shows AUC range 0.686–0.888 across folds (ΔAUC = 0.202) — much higher than other models
- Add a markdown explanation: likely reflects sensitivity to the particular training/test split given the BN's fixed DAG structure

**3. Interpret the ECE values**
- BN ECE = 0.165 is reported but not contextualised
- Add: "ECE > 0.10 indicates substantial miscalibration; BN's probabilities should not be used directly as clinical risk scores without calibration"

**4. Explain negative log-loss magnitude for BN**
- BN log-loss of −3.30 vs LR's −0.387 looks alarming but is correct
- Add a note clarifying this is per-sample negative log-likelihood, and BN's high value reflects overconfident wrong predictions

**5. Add a final recommendation cell**
- Notebook ends with numbers but no conclusion
- Add: "For clinical use prioritising calibrated uncertainty, MRF-Tuned offers the best AUC (0.916) with competitive accuracy. For raw discriminative performance, SVM achieves best F1 (0.818)."

**6. Move `COLORS` dict to top of notebook**
- Currently defined mid-notebook; causes NameError if notebook is rerun from middle

---

## bayesian_network.ipynb

### What Works Well
- Well-motivated 3-layer DAG (Risk Factors → Disease → Symptoms) with clinical justification
- Explicit discretization thresholds for all features
- Clear inference examples (Q1–Q4) demonstrating prior, forward, backward, and conditional queries

### Issues to Fix

**1. Add a network visualization** *(medium priority)*
- No DAG plot; MRF notebook has a beautiful 3-layer graph but BN has none
- Add `networkx` visualization showing nodes coloured by layer (risk factors / disease / symptoms)

**2. Show the learned Conditional Probability Tables (CPTs)**
- After MLE parameter learning, print or visualise key CPTs (e.g., `P(num | risk factors)`)
- This is the most interpretable part of a BN — currently hidden

**3. State explicitly that structure is hand-coded**
- The DAG is based on domain knowledge, not learned
- Add a markdown note: "Structure is fixed based on clinical knowledge. Structure learning (e.g., Hill Climbing, PC algorithm) is left for future work."

**4. Discuss VE failure fallback**
- Code falls back to probability 0.5 on Variable Elimination failure
- Add: how often does this happen? If >5% of test samples, it significantly affects results

**5. Address high fold-level variance**
- Fold 2: AUC = 0.799, Fold 5: AUC = 0.686
- Investigate and explain — are certain folds missing certain discretisation bins in training data?

---

## markov_random_field.ipynb

### What Works Well
- Best-documented notebook: clear problem → fix → result reasoning for each improvement
- Excellent 3-layer network visualizations
- Systematic hyperparameter search (126 configurations) with ranked results

### Issues to Fix

**1. Show actual MI values for edges**
- MI reweighting is the key innovation, but the MI values themselves are never displayed
- Add a bar chart of edge MI values — which edges are most informative?

**2. Add a final recommendation**
- Section 9 shows 12 top configs but doesn't state clearly which to use
- Add: "Recommended: Base graph + alpha=10.0 + gamma=1.5 (AUC=0.916). Enriched graph adds no benefit despite added complexity."

**3. Clarify the "remaining gap" language**
- "Remaining gap: −0.0042" is confusing — MRF-Tuned (0.916) *beats* Naive Bayes (0.912)
- Reword: "MRF-Tuned exceeds the best baseline (Naive Bayes, AUC=0.912) by +0.004"

**4. Add Belief Propagation convergence check**
- BP is used for inference but no convergence monitoring is shown
- Add iteration count and residual plot to confirm convergence

**5. Show calibration metrics in Section 10 consolidation**
- Final comparison only shows AUC bars; Brier Score and log-loss are not shown
- Add these to the consolidated comparison for completeness

---

## gmm_em.ipynb

### What Works Well
- From-scratch GMM implementation verified against sklearn (ARI = 1.0)
- Rich motivation section explaining why GMM/EM is appropriate
- Excellent cluster interpretation (high-risk vs lower-risk phenotypes)
- Comprehensive findings section (Finding 1–5)

### Issues to Fix

**1. Explicitly address the GMM classifier's limited feature set**
- AUC = 0.743 because only 5 continuous features are used (excludes `ca`, `thal` — the two strongest predictors)
- This is stated but not emphasised: add a clear note in the results cell
- Frame it positively: "Using only continuous features, GMM achieves AUC=0.743. The performance gap vs baselines is attributable to this constraint, not to the GMM model itself."

**2. Address out-of-range EM imputation properly**
- EM produces `ca = -0.25`, `thal ≈ 4.6` (both invalid for discrete features)
- Add a brief section on why this happens (Gaussian assumption on discrete data) and what the fix would be (MICE, ordinal models)
- This demonstrates understanding of the limitation, which is what markers want to see

**3. Validate EM imputation against mode imputation**
- EM imputation is demonstrated but never compared to mode imputation on downstream performance
- Either add a comparison (run BN/LR with EM-imputed vs mode-imputed data) or explicitly note this as future work

**4. Justify K=2 selection more strongly**
- BIC selects K=2, which is stated but not deeply analysed
- Add: "K=2 aligns with the binary disease status, supporting the interpretation that GMM recovers a clinically meaningful latent structure"

**5. Clarify the multiple feature set definitions**
- `CONT_COLS_GMM` (5 features for classifier) vs `CONT_COLS` (used in EM imputation) are confusing
- Add a comment or markdown cell mapping each variable set to its purpose

---

## README.md

The current README is essentially empty. It needs a complete rewrite covering:

```markdown
# CS5340 Heart Disease Probabilistic Modeling

## Overview
[1–2 paragraphs: what the project does, why it matters]

## Dataset
[Cleveland Heart Disease, 303 patients, 13 features, binary target]

## Models
| Notebook | Model | Best AUC |
|----------|-------|----------|
| eda.ipynb | Exploratory Analysis | — |
| gmm_em.ipynb | GMM + EM | 0.743 |
| bayesian_network.ipynb | Bayesian Network | 0.795 |
| markov_random_field.ipynb | MRF (Tuned) | 0.916 |
| evaluation.ipynb | Comparative Evaluation | — |

## Results Summary
[Key findings in 3–5 bullet points]

## Requirements
[pip install pgmpy networkx scikit-learn ucimlrepo ...]

## Usage
[How to run each notebook in order]
```

---

## Cross-Notebook Consistency Issues

These issues span multiple notebooks and should be fixed together:

**1. Feature sets are inconsistent**
- GMM classifier: 5 continuous features only
- BN/MRF/baselines: all 13 features
- This makes performance comparisons unfair — GMM is handicapped
- Fix: note explicitly in `evaluation.ipynb` that GMM uses fewer features

**2. Discretization thresholds are duplicated**
- Same bin thresholds defined in both `bayesian_network.ipynb` and `markov_random_field.ipynb`
- Fix: extract to a shared `preprocessing.py` or `utils.py` (or at minimum note they must match)

**3. Missing value handling differs**
- `heart_disease_cleaned.csv` uses mode imputation
- `gmm_em.ipynb` loads raw data and applies EM imputation, but this is never propagated
- Fix: either use EM-imputed data everywhere or explicitly state mode imputation is the canonical choice

**4. Baselines use default hyperparameters without justification**
- LogReg `C=1.0`, SVM `C=1.0 kernel=rbf`, RF `n_estimators=100` — all defaults
- Fix: add one sentence per baseline justifying or acknowledging the default choice

---

## Quick Wins (< 1 hour each)

These require minimal effort but noticeably improve presentation quality:

- [ ] Add network visualization to `bayesian_network.ipynb`
- [ ] Move `COLORS` dict to top of `evaluation.ipynb`
- [ ] Fill empty markdown cell in `eda.ipynb`
- [ ] Remove "Older stuff" section from `eda.ipynb`
- [ ] Add final recommendation cells to `evaluation.ipynb` and `markov_random_field.ipynb`
- [ ] Show edge MI values as a bar chart in `markov_random_field.ipynb`
- [ ] Reword "remaining gap: -0.0042" in `markov_random_field.ipynb`
- [ ] Write `README.md`
