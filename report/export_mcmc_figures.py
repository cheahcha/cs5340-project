"""
Export MCMC/latent inference figures for the LaTeX report.
Run from the project root: python export_mcmc_figures.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import roc_auc_score

from robustness_utils import (
    load_discretized_heart_data, STATE_NAMES, BASE_EDGES,
    binary_entropy, ALL_FEATURES
)

np.random.seed(42)
OUTDIR = "figures"
df_disc = load_discretized_heart_data('heart_disease_cleaned.csv')

RISK_FACTORS = ['age', 'sex', 'fbs', 'chol', 'trestbps']
SYMPTOMS = ['cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'restecg']

# ── Build BN ──────────────────────────────────────────────────────────────────
def build_bn_model(train_df):
    model = DiscreteBayesianNetwork(BASE_EDGES)
    model.fit(train_df, estimator=BayesianEstimator, state_names=STATE_NAMES,
              prior_type='BDeu', equivalent_sample_size=10.0)
    return model, VariableElimination(model)

split_idx = int(0.8 * len(df_disc))
train_df = df_disc.iloc[:split_idx].reset_index(drop=True)
test_df = df_disc.iloc[split_idx:].reset_index(drop=True)
y_test = test_df['num'].values
bn_model, ve = build_bn_model(train_df)

# ── Gibbs Sampler (same as notebook) ──────────────────────────────────────────
class GibbsWithEvidence:
    def __init__(self, model):
        self.model = model
        self.cpds = {cpd.variable: cpd for cpd in model.get_cpds()}

    def _full_conditional(self, var, state):
        n_states = len(STATE_NAMES[var])
        log_probs = np.zeros(n_states)
        for val in range(n_states):
            state_copy = dict(state); state_copy[var] = val
            cpd = self.cpds[var]
            parents = list(cpd.get_evidence())
            if parents:
                try:
                    idx = tuple(cpd.state_names[p].index(state_copy[p]) for p in parents)
                    log_probs[val] += np.log(max(cpd.values[val][idx], 1e-10))
                except: pass
            else:
                try: log_probs[val] += np.log(max(cpd.values[val], 1e-10))
                except: pass
            for child in self.model.get_children(var):
                child_cpd = self.cpds[child]
                child_parents = list(child_cpd.get_evidence())
                try:
                    idx = tuple(child_cpd.state_names[p].index(state_copy[p]) for p in child_parents)
                    child_val = state_copy[child]
                    if isinstance(child_val, (int, np.integer)):
                        log_probs[val] += np.log(max(child_cpd.values[child_val][idx], 1e-10))
                except: pass
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        return probs / probs.sum()

    def sample_with_evidence(self, evidence, latent_vars, n_samples=2000,
                             burn_in=500, thin=2, seed=42):
        rng = np.random.default_rng(seed)
        state = dict(evidence)
        for v in latent_vars:
            state[v] = rng.integers(0, len(STATE_NAMES[v]))
        samples = []
        for step in range(n_samples + burn_in):
            for v in latent_vars:
                probs = self._full_conditional(v, state)
                state[v] = int(rng.choice(len(STATE_NAMES[v]), p=probs))
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples.append({v: state[v] for v in latent_vars})
        return pd.DataFrame(samples)

gibbs = GibbsWithEvidence(bn_model)

# ── Figure 1: Convergence Diagnostics ─────────────────────────────────────────
print("Running convergence diagnostics...")
test_patient = df_disc.iloc[42]
obs = {col: int(test_patient[col]) for col in ALL_FEATURES}
n_chains, n_samples, burn_in = 5, 1000, 200
chains = []
for seed in range(n_chains):
    s = gibbs.sample_with_evidence(evidence=obs, latent_vars=['num'],
                                   n_samples=n_samples, burn_in=burn_in, thin=1, seed=seed*100)
    chains.append(s['num'].values)
chains = np.array(chains)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, ax in enumerate(axes.flatten()[:5]):
    ax.plot(chains[i], alpha=0.7, linewidth=0.8, color=f'C{i}')
    ax.axhline(chains[i].mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean={chains[i].mean():.3f}')
    ax.set_title(f'Chain {i+1}', fontsize=10)
    ax.set_xlabel('Sample'); ax.set_ylabel('num'); ax.legend(fontsize=8)
ax = axes.flatten()[5]
running_mean = np.cumsum(chains[0]) / np.arange(1, len(chains[0]) + 1)
ax.plot(running_mean, color='navy', linewidth=1.5)
ax.axhline(chains[0].mean(), color='red', linestyle='--', label='Final mean')
ax.set_title('Running Mean — Chain 1', fontsize=10)
ax.set_xlabel('Sample'); ax.set_ylabel('P(num=1)'); ax.legend(fontsize=8)
plt.suptitle('Gibbs Sampler Convergence Diagnostics', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/mcmc_convergence.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: mcmc_convergence.png")

# ── Run 3 scenarios ───────────────────────────────────────────────────────────
S2_OBSERVED = [f for f in ALL_FEATURES if f not in ['ca', 'thal']]
S3_OBSERVED = ['age', 'sex', 'fbs', 'chol', 'trestbps']

def run_scenario(observed_cols, latent_vars, label):
    print(f"Running {label}...")
    ve_probs, mcmc_probs = [], []
    for i, (_, row) in enumerate(test_df.iterrows()):
        if i % 10 == 0: print(f'  {i+1}/{len(test_df)}...', end='\r')
        evidence = {col: int(row[col]) for col in observed_cols}
        # VE
        try:
            q = ve.query(['num'], evidence=evidence, show_progress=False)
            ve_probs.append(float(q.values[1]))
        except:
            ve_probs.append(0.5)
        # MCMC
        samples = gibbs.sample_with_evidence(
            evidence=evidence, latent_vars=latent_vars,
            n_samples=500, burn_in=100, thin=2, seed=i)
        mcmc_probs.append(float((samples['num'] == 1).mean()))
    print(f"\n  {label} done.")
    return np.array(ve_probs), np.array(mcmc_probs)

ve_s1, mc_s1 = run_scenario(ALL_FEATURES, ['num'], 'S1')
ve_s2, mc_s2 = run_scenario(S2_OBSERVED, ['num', 'ca', 'thal'], 'S2')
ve_s3, mc_s3 = run_scenario(S3_OBSERVED, ['num'] + [f for f in ALL_FEATURES if f not in S3_OBSERVED], 'S3')

# ── Figure 2: MCMC vs VE scatter (S1) ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(ve_s1, mc_s1, alpha=0.6, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect agreement')
corr = np.corrcoef(ve_s1, mc_s1)[0, 1]
ax.set_xlabel('VE P(num=1)', fontsize=11); ax.set_ylabel('MCMC P(num=1)', fontsize=11)
ax.set_title(f'S1: MCMC vs VE (r={corr:.3f})', fontsize=11, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(ve_s1, bins=20, alpha=0.6, color='steelblue', label='VE')
ax.hist(mc_s1, bins=20, alpha=0.6, color='firebrick', label='MCMC')
ax.set_xlabel('P(num=1)', fontsize=11); ax.set_ylabel('Count', fontsize=11)
ax.set_title('S1: Posterior Distribution Comparison', fontsize=11, fontweight='bold')
ax.legend()
plt.suptitle('Scenario S1: num latent, all 13 features observed', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/mcmc_s1_scatter.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: mcmc_s1_scatter.png")

# ── Figure 3: AUC comparison + entropy across scenarios ──────────────────────
auc_ve = [roc_auc_score(y_test, p) for p in [ve_s1, ve_s2, ve_s3]]
auc_mc = [roc_auc_score(y_test, p) for p in [mc_s1, mc_s2, mc_s3]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
scenarios = ['S1\n(num latent)', 'S2\n(num+ca+thal)', 'S3\n(risk factors\nonly)']
x = np.arange(3); w = 0.35
bars_ve = ax.bar(x - w/2, auc_ve, w, label='VE (exact)', color='steelblue', alpha=0.85)
bars_mc = ax.bar(x + w/2, auc_mc, w, label='MCMC (approx.)', color='firebrick', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=9)
ax.set_ylim(0, 1.08); ax.set_ylabel('AUC', fontsize=11); ax.legend(fontsize=10)
ax.set_title('MCMC vs VE: AUC Across Scenarios', fontsize=11, fontweight='bold')
ax.bar_label(bars_ve, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(bars_mc, fmt='%.3f', padding=3, fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ent_s1, ent_s2, ent_s3 = binary_entropy(mc_s1), binary_entropy(mc_s2), binary_entropy(mc_s3)
ax.hist(ent_s1, bins=15, alpha=0.6, color='steelblue', label=f'S1 (mean={ent_s1.mean():.3f})')
ax.hist(ent_s2, bins=15, alpha=0.6, color='firebrick', label=f'S2 (mean={ent_s2.mean():.3f})')
ax.hist(ent_s3, bins=15, alpha=0.6, color='green', label=f'S3 (mean={ent_s3.mean():.3f})')
ax.set_xlabel('Predictive Entropy H[P(num=1)]', fontsize=11); ax.set_ylabel('Count', fontsize=11)
ax.set_title('Posterior Uncertainty by Scenario', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/mcmc_scenario_comparison.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: mcmc_scenario_comparison.png")

# Print AUC table for reference
print("\n── AUC Summary ──")
for s, ve_a, mc_a in zip(['S1','S2','S3'], auc_ve, auc_mc):
    print(f"  {s}: VE={ve_a:.4f}  MCMC={mc_a:.4f}  |Δ|={abs(ve_a-mc_a):.4f}")

print("\nAll MCMC figures exported.")
