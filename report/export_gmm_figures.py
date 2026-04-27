"""
Export all GMM/EM figures for the LaTeX report.
Run from the project root: python export_gmm_figures.py
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_score, recall_score, log_loss, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

OUTDIR = "figures"

# ── Load data from local cleaned CSV (mode-imputed, no network needed) ────────
df_raw = pd.read_csv("heart_disease_cleaned.csv")
df_raw['num'] = (df_raw['num'] > 0).astype(int)
y = df_raw['num'].values
# Note: cleaned CSV has no missing values (already mode-imputed).
# For the EM imputation figure, we artificially re-introduce NaN at the
# same positions the raw UCI data had missing values (ca: rows 87,166,192,266;
# thal: rows 87,266) so the figure is reproducible without network access.
MISSING_CA = [87, 166, 192, 266]
MISSING_THAL = [87, 266]

# ── EM Imputation ─────────────────────────────────────────────────────────────
CONT_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']

def em_impute(X_raw, max_iter=200, tol=1e-5):
    X = X_raw.astype(float).copy()
    N, D = X.shape
    missing = np.isnan(X)
    col_means = np.nanmean(X, axis=0)
    for j in range(D):
        X[missing[:, j], j] = col_means[j]
    log_likelihoods = []
    for it in range(max_iter):
        mu = X.mean(axis=0)
        Sigma = np.cov(X.T) + 1e-6 * np.eye(D)
        ll = np.sum(stats.multivariate_normal.logpdf(X, mean=mu, cov=Sigma))
        log_likelihoods.append(ll)
        X_new = X.copy()
        for i in range(N):
            miss_idx = np.where(missing[i])[0]
            if len(miss_idx) == 0:
                continue
            obs_idx = np.where(~missing[i])[0]
            Sigma_mo = Sigma[np.ix_(miss_idx, obs_idx)]
            Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
            try:
                mu_cond = mu[miss_idx] + Sigma_mo @ np.linalg.solve(Sigma_oo, X[i, obs_idx] - mu[obs_idx])
                X_new[i, miss_idx] = mu_cond
            except np.linalg.LinAlgError:
                X_new[i, miss_idx] = mu[miss_idx]
        if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
        X = X_new
    return X, log_likelihoods

# Re-introduce missing values for EM demonstration
df_with_nan = df_raw.copy()
for idx in MISSING_CA:
    df_with_nan.loc[idx, 'ca'] = np.nan
for idx in MISSING_THAL:
    df_with_nan.loc[idx, 'thal'] = np.nan

X_cont_raw = df_with_nan[CONT_COLS].values
X_em, ll_history = em_impute(X_cont_raw)
df_em = df_with_nan.copy()
df_em[CONT_COLS] = X_em

# Mode imputation baseline (just uses the cleaned CSV values directly)
df_mode = df_raw.copy()

# ── Figure 1: EM Imputation Convergence + Imputed Values ─────────────────────
missing_rows_ca = MISSING_CA
missing_rows_thal = MISSING_THAL
mode_ca = df_mode.loc[missing_rows_ca, 'ca'].values
em_ca = df_em.loc[missing_rows_ca, 'ca'].values
mode_thal = df_mode.loc[missing_rows_thal, 'thal'].values
em_thal = df_em.loc[missing_rows_thal, 'thal'].values

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
axes[0].plot(ll_history, color='steelblue', linewidth=2, marker='o', markersize=6)
axes[0].set_xlabel('Iteration', fontsize=11)
axes[0].set_ylabel('Log-Likelihood', fontsize=11)
axes[0].set_title('(a) EM Log-Likelihood Convergence', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

x_pos = np.arange(len(missing_rows_ca))
axes[1].bar(x_pos - 0.2, mode_ca, 0.4, label='Mode', color='#FF7C00', alpha=0.85)
axes[1].bar(x_pos + 0.2, em_ca, 0.4, label='EM', color='steelblue', alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f'Row {r}' for r in missing_rows_ca], fontsize=9)
axes[1].set_title('(b) Imputed values — ca (4 missing)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Imputed value')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

x_pos2 = np.arange(len(missing_rows_thal))
axes[2].bar(x_pos2 - 0.2, mode_thal, 0.4, label='Mode', color='#FF7C00', alpha=0.85)
axes[2].bar(x_pos2 + 0.2, em_thal, 0.4, label='EM', color='steelblue', alpha=0.85)
axes[2].set_xticks(x_pos2)
axes[2].set_xticklabels([f'Row {r}' for r in missing_rows_thal], fontsize=9)
axes[2].set_title('(c) Imputed values — thal (2 missing)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Imputed value')
axes[2].legend(fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/gmm_em_imputation.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: gmm_em_imputation.png")

# ── Prepare continuous features for GMM ───────────────────────────────────────
CONT_COLS_GMM = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_em[CONT_COLS_GMM].values)

# ── Figure 2: BIC/AIC Model Selection ────────────────────────────────────────
K_range = range(1, 9)
bic_scores, aic_scores, ll_scores = [], [], []
for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42, max_iter=300, n_init=5).fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))
    ll_scores.append(gmm.score(X_scaled) * len(X_scaled))

best_k_bic = list(K_range)[int(np.argmin(bic_scores))]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(K_range, bic_scores, 'o-', color='steelblue', linewidth=2, markersize=7, label='BIC')
axes[0].plot(K_range, aic_scores, 's-', color='darkorange', linewidth=2, markersize=7, label='AIC')
axes[0].axvline(best_k_bic, color='steelblue', linestyle='--', alpha=0.6, label=f'Best BIC: K={best_k_bic}')
axes[0].set_xlabel('Number of Components K', fontsize=11)
axes[0].set_ylabel('Score (lower is better)', fontsize=11)
axes[0].set_title('(a) BIC & AIC vs K', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(K_range))

axes[1].plot(K_range, ll_scores, 'D-', color='seagreen', linewidth=2, markersize=7)
axes[1].set_xlabel('Number of Components K', fontsize=11)
axes[1].set_ylabel('Log-Likelihood', fontsize=11)
axes[1].set_title('(b) Log-Likelihood vs K', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(K_range))

plt.tight_layout()
plt.savefig(f'{OUTDIR}/gmm_bic_aic.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: gmm_bic_aic.png")

# ── Figure 3: PCA Cluster Visualization ──────────────────────────────────────
K_best = best_k_bic
gmm_best = GaussianMixture(n_components=K_best, covariance_type='full',
                           random_state=42, max_iter=300, n_init=5).fit(X_scaled)
cluster_labels = gmm_best.predict(X_scaled)
soft_probs = gmm_best.predict_proba(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_

CLUSTER_COLORS = plt.cm.Set1(np.linspace(0, 0.8, K_best))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for k in range(K_best):
    mask = cluster_labels == k
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[CLUSTER_COLORS[k]], s=40, alpha=0.75,
                    label=f'Cluster {k} (n={mask.sum()})', edgecolors='none')
centres_pca = pca.transform(gmm_best.means_)
axes[0].scatter(centres_pca[:, 0], centres_pca[:, 1],
                marker='X', s=200, c='black', zorder=5, label='Centres')
axes[0].set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=11)
axes[0].set_title(f'(a) GMM Clusters (K={K_best})', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for label, col, name in [(0, '#4C72B0', 'Healthy'), (1, '#C44E52', 'Disease')]:
    mask = y == label
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=col, s=40, alpha=0.65, label=name, edgecolors='none')
axes[1].set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)', fontsize=11)
axes[1].set_title('(b) True Disease Label', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/gmm_pca_clusters.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: gmm_pca_clusters.png")

# ── Figure 4: Cluster Mean Profiles ──────────────────────────────────────────
df_cluster = df_em[CONT_COLS_GMM].copy()
df_cluster['cluster'] = cluster_labels
means_per_cluster = df_cluster.groupby('cluster')[CONT_COLS_GMM].mean()

fig, axes = plt.subplots(1, len(CONT_COLS_GMM), figsize=(16, 4))
for ax, col in zip(axes, CONT_COLS_GMM):
    vals = [means_per_cluster.loc[k, col] for k in range(K_best)]
    bars = ax.bar(range(K_best), vals, color=CLUSTER_COLORS[:K_best], alpha=0.85, edgecolor='white')
    ax.set_xticks(range(K_best))
    ax.set_xticklabels([f'C{k}' for k in range(K_best)], fontsize=10)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Cluster Mean Profiles (Original Scale)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/gmm_cluster_profiles.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: gmm_cluster_profiles.png")

# ── Cross-validation for GMM classifier ───────────────────────────────────────
def gmm_classifier_predict_proba(gmm0, gmm1, prior0, prior1, X):
    log_p0 = gmm0.score_samples(X) + np.log(prior0)
    log_p1 = gmm1.score_samples(X) + np.log(prior1)
    log_norm = logsumexp(np.stack([log_p0, log_p1], axis=1), axis=1)
    return np.exp(log_p1 - log_norm)

N_SPLITS = 5
K_COMP = best_k_bic
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
X_full = df_em[CONT_COLS_GMM].values
y_full = y.copy()

MODEL_NAMES = ['GMM', 'LogisticRegression', 'GaussianNB', 'RandomForest', 'SVM']
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss', 'brier']
cv_results = {m: {met: [] for met in METRICS} for m in MODEL_NAMES}
pool_true = {m: [] for m in MODEL_NAMES}
pool_proba = {m: [] for m in MODEL_NAMES}

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_full, y_full)):
    X_tr, X_te = X_full[tr_idx], X_full[te_idx]
    y_tr, y_te = y_full[tr_idx], y_full[te_idx]
    scaler_f = StandardScaler()
    X_tr_s = scaler_f.fit_transform(X_tr)
    X_te_s = scaler_f.transform(X_te)

    prior0 = (y_tr == 0).mean()
    prior1 = (y_tr == 1).mean()
    gmm0 = GaussianMixture(n_components=K_COMP, covariance_type='full',
                           random_state=42, max_iter=300, n_init=5).fit(X_tr_s[y_tr == 0])
    gmm1 = GaussianMixture(n_components=K_COMP, covariance_type='full',
                           random_state=42, max_iter=300, n_init=5).fit(X_tr_s[y_tr == 1])
    p_gmm = gmm_classifier_predict_proba(gmm0, gmm1, prior0, prior1, X_te_s)
    yp_gmm = (p_gmm >= 0.5).astype(int)

    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_s, y_tr)
    gnb = GaussianNB().fit(X_tr_s, y_tr)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_s, y_tr)
    svm = SVC(kernel='rbf', probability=True, random_state=42).fit(X_tr_s, y_tr)

    fold_data = [
        ('GMM', yp_gmm, p_gmm),
        ('LogisticRegression', lr.predict(X_te_s), lr.predict_proba(X_te_s)[:, 1]),
        ('GaussianNB', gnb.predict(X_te_s), gnb.predict_proba(X_te_s)[:, 1]),
        ('RandomForest', rf.predict(X_te_s), rf.predict_proba(X_te_s)[:, 1]),
        ('SVM', svm.predict(X_te_s), svm.predict_proba(X_te_s)[:, 1]),
    ]
    for name, yp, pp in fold_data:
        cv_results[name]['accuracy'].append(accuracy_score(y_te, yp))
        cv_results[name]['precision'].append(precision_score(y_te, yp, zero_division=0))
        cv_results[name]['recall'].append(recall_score(y_te, yp, zero_division=0))
        cv_results[name]['f1'].append(f1_score(y_te, yp, zero_division=0))
        cv_results[name]['auc'].append(roc_auc_score(y_te, pp))
        cv_results[name]['log_loss'].append(log_loss(y_te, pp))
        cv_results[name]['brier'].append(brier_score_loss(y_te, pp))
        pool_true[name].extend(y_te.tolist())
        pool_proba[name].extend(pp.tolist())

print("Cross-validation complete.")

# ── Figure 5: ROC Curves ─────────────────────────────────────────────────────
COLORS = {
    'GMM': '#9B59B6',
    'LogisticRegression': '#4C72B0',
    'GaussianNB': '#55A868',
    'RandomForest': '#C44E52',
    'SVM': '#8172B2',
}
SHORT = {'GMM': 'GMM', 'LogisticRegression': 'LR', 'GaussianNB': 'GNB',
         'RandomForest': 'RF', 'SVM': 'SVM'}

fig, ax = plt.subplots(figsize=(7, 5.5))
for name in MODEL_NAMES:
    yt = np.array(pool_true[name])
    yp = np.array(pool_proba[name])
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
            label=f'{SHORT[name]} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — GMM Classifier vs Baselines\n(5-fold CV, continuous features only)',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/gmm_roc_curves.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: gmm_roc_curves.png")

# ── Figure 6: Performance Bar Charts ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(MODEL_NAMES))
short_labels = [SHORT[m] for m in MODEL_NAMES]

for ax, metric, label in [
    (axes[0], 'accuracy', 'Accuracy'),
    (axes[1], 'f1', 'F1 Score'),
    (axes[2], 'auc', 'AUC-ROC'),
]:
    means = [np.mean(cv_results[m][metric]) for m in MODEL_NAMES]
    stds = [np.std(cv_results[m][metric]) for m in MODEL_NAMES]
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS[m] for m in MODEL_NAMES], alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('5-Fold CV — Continuous Features Only', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/gmm_performance_bars.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: gmm_performance_bars.png")

print("\nAll GMM figures exported to figures/")
