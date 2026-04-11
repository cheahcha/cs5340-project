# Project Findings Summary

CS5340 course project: probabilistic graphical models applied to the Cleveland Heart Disease dataset.

---

## Dataset

Cleveland Heart Disease dataset — 303 patients, 13 clinical features, binary target (`num`: disease/no disease, 46% positive rate). All continuous features discretized into ordinal bins for use with pgmpy. 6 missing values (4 × `ca`, 2 × `thal`) imputed via EM.

---

## Models Implemented

### 1. Bayesian Network (BN) — `bayesian_network.ipynb`

Fixed 3-layer directed DAG: Risk Factors → `num` → Symptoms. Three progressively improved variants:

| Variant | Key change | AUC |
|---------|-----------|-----|
| BN-MLE | MLE parameters | 0.795 |
| BN-BDeu | BDeu prior (fixes zero-count collapse in sparse CPTs) | 0.896 (+0.101) |
| BN-Enriched | + 4 direct risk-factor→symptom edges | 0.898 (+0.002) |
| BN-HC | Hill-climbing structure learning + BDeu | **0.916** (+0.121 vs MLE) |

MLE was the worst because ~242 training rows couldn't fill the 72 CPT cells for `num`'s five discrete parents, producing zero-probability entries that collapsed discriminability. BDeu alone closes ~80% of the gap.

### 2. Markov Random Field (MRF) — `markov_random_field.ipynb`

Same 3-layer structure but undirected. Pairwise + unary potentials with Laplace smoothing. Three variants:

| Variant | Key change | AUC |
|---------|-----------|-----|
| MRF-Base | α=1.0, γ=0 | 0.912 |
| MRF-MI | MI-reweighted pairwise potentials (γ=1.5) | 0.913 |
| MRF-Tuned | Grid search over α/γ (best: α=10.0, γ=1.5) | **0.916** |

Enriching the graph with 4 extra edges produced no benefit beyond MI reweighting, since the tuned α already acts as strong smoothing that dominates the signal.

### 3. GMM / EM — `gmm_em.ipynb`

Two applications:

- **EM imputation**: converged in 4 iterations (low missingness, ~2%). Correctly personalises imputed values per patient (e.g., same missing `ca` → 0.24 vs 0.85 depending on other features), unlike mode imputation which assigns 0 to all. Limitation: multivariate Gaussian assumption poorly models inherently discrete features like `thal`.
- **GMM clustering** (K=2 by BIC): discovered two clinically coherent subgroups — older patients with higher ST-depression (55% disease rate) vs younger with higher max HR (26% disease rate). ARI vs true labels ≈ 0.20.
- **GMM generative classifier**: AUC=0.743 (continuous features only), well below the discriminative models due to the generative objective and restriction to 5 continuous features.

---

## Unified Evaluation — `evaluation.ipynb`

5-fold stratified CV across all 8 models, 9 metrics (Accuracy, Precision, Recall, F1, AUC, Log-Likelihood, Brier Score, Predictive Entropy, ECE):

| Model | Accuracy | F1 | AUC | ECE |
|-------|----------|----|-----|-----|
| BN (MLE) | 0.772 | 0.738 | 0.795 | 0.165 |
| BN (BDeu) | 0.812 | 0.786 | 0.896 | 0.143 |
| BN (BDeu+Enriched) | 0.812 | 0.791 | 0.898 | 0.142 |
| Logistic Regression | 0.835 | 0.813 | 0.905 | 0.109 |
| Naive Bayes | 0.832 | 0.811 | **0.912** | 0.129 |
| Random Forest | 0.825 | 0.805 | 0.898 | 0.112 |
| **SVM (RBF)** | **0.838** | **0.818** | 0.891 | **0.096** |
| MRF (Tuned) | 0.809 | 0.756 | **0.912** | 0.170 |

### Key findings

- **MRF ties Naive Bayes for best AUC (0.912)** — competitive with discriminative models despite having no discriminative training objective.
- **SVM is best for accuracy/F1** but MRF/BN offer interpretability via explicit graph structure.
- **BN (MLE) is significantly worse** than all other models; BDeu prior alone closes ~80% of the gap.
- **No pair of models is statistically distinguishable** after Bonferroni correction (28 pairs, n=303 too small for significance at α=0.0018).
- **Calibration**: SVM is best calibrated (ECE=0.096); MRF and BN-MLE are worst (ECE≈0.17). Platt scaling substantially reduces ECE for both graphical models.

---

## Robustness Under Missing Evidence — `robustness_experiments.ipynb`

Tested how models degrade when features are withheld at inference time, simulating real-world scenarios where not all clinical tests are available.

### Single-feature removal

`ca` (coronary artery count) is the most critical feature for all models — removing it causes the largest mean AUC drop (~0.025). Consistent with its known clinical importance (strongest correlation with disease, r=0.52).

### Clinical group removal

| Scenario | Features removed | Base BN ΔAUC | Base MRF ΔAUC |
|----------|-----------------|-------------|--------------|
| Drop imaging/perfusion | ca, thal | −0.13 | −0.14 |
| Drop exercise ECG | restecg, thalach, exang, oldpeak, slope | −0.06 | −0.05 |
| Drop risk factors | age, sex, fbs, chol, trestbps | −0.03 | −0.02 |
| Drop all diagnostics | all except demographics | **−0.26** | **−0.22** |

### BN vs MRF under missing evidence

On average, Base BN degrades slightly more than Base MRF (mean AUC drop −0.022 vs −0.019). The undirected MRF can partially compensate for a missing node via alternative paths through shared neighbours, while the BN's directed structure severs information flow along specific paths. The Best BN (hill-climbing) is more resilient than the Base BN because its learned multi-hop structure creates redundant inference paths.

---

## Overall Conclusions

1. **For AUC/ranking**: MRF-Tuned and BN-HC are the best PGMs, matching Naive Bayes at AUC≈0.916.
2. **For accuracy/F1**: Discriminative models (SVM, LR) have a ~0.03 edge — expected given their direct optimisation objective.
3. **For calibration**: All PGMs need post-hoc calibration (Platt scaling) before probabilities can be used as clinical risk scores.
4. **For robustness**: MRF is marginally more robust to missing features than BN due to its undirected propagation, but the difference is small in practice.
5. **For interpretability + performance**: BN (BDeu+Enriched) and MRF-Tuned offer the best trade-off — competitive AUC with explicit, explainable graph structure suitable for clinical reasoning.
