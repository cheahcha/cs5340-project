from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
try:
    from pgmpy.models import DiscreteMarkovNetwork
except ImportError:
    from pgmpy.models import MarkovNetwork as DiscreteMarkovNetwork
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import brier_score_loss, log_loss, mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RISK_FACTORS = ["age", "sex", "fbs", "chol", "trestbps"]
SYMPTOMS = ["cp", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "restecg"]
ALL_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
MATCHED_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
MODEL_ORDER = ["Base BN", "Best BN", "Base MRF", "Best MRF"]
FAMILY_ORDER = ["BN", "MRF"]
MATCHED_MODEL_ORDER = ["Matched BN", "Matched MRF", "GMM"]
MODEL_DISPLAY_LABELS = {
    "Base BN": "Base BN",
    "Best BN": "BN-HC",
    "Base MRF": "Base MRF",
    "Best MRF": "MRF-Tuned",
    "Matched BN": "Matched BN",
    "Matched MRF": "Matched MRF",
    "GMM": "GMM",
}

STATE_NAMES = {
    "age": [0, 1, 2],
    "sex": [0, 1],
    "fbs": [0, 1],
    "chol": [0, 1, 2],
    "trestbps": [0, 1],
    "num": [0, 1],
    "cp": [0, 1, 2, 3],
    "thalach": [0, 1, 2],
    "exang": [0, 1],
    "oldpeak": [0, 1, 2],
    "slope": [0, 1, 2],
    "ca": [0, 1, 2, 3],
    "thal": [0, 1, 2],
    "restecg": [0, 1, 2],
}

BASE_EDGES = [(rf, "num") for rf in RISK_FACTORS] + [("num", sym) for sym in SYMPTOMS]
MATCHED_EDGES = [
    ("age", "num"),
    ("trestbps", "num"),
    ("chol", "num"),
    ("num", "thalach"),
    ("num", "oldpeak"),
]

BEST_BN_EDGES = [
    ("age", "fbs"),
    ("ca", "age"),
    ("cp", "exang"),
    ("num", "ca"),
    ("num", "cp"),
    ("num", "exang"),
    ("num", "sex"),
    ("num", "thal"),
    ("oldpeak", "num"),
    ("oldpeak", "slope"),
    ("restecg", "chol"),
    ("restecg", "oldpeak"),
    ("slope", "thalach"),
    ("thal", "sex"),
    ("thalach", "num"),
    ("thalach", "thal"),
    ("trestbps", "slope"),
]

assert nx.is_directed_acyclic_graph(nx.DiGraph(BEST_BN_EDGES))

ALL_SINGLE_FEATURE_REMOVAL = {f"drop_{feature}": [feature] for feature in ALL_FEATURES}
MATCHED_SINGLE_FEATURE_REMOVAL = {f"drop_{feature}": [feature] for feature in MATCHED_FEATURES}

CLINICAL_GROUP_REMOVAL = {
    "drop_risk_factors": ["age", "sex", "fbs", "chol", "trestbps"],
    "drop_exercise_ecg": ["restecg", "thalach", "exang", "oldpeak", "slope"],
    "drop_imaging_perf": ["ca", "thal"],
    "drop_all_diagnostics": ["cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
}
CLINICAL_GROUP_LABELS = {
    "drop_risk_factors": "Risk Factors Removed",
    "drop_exercise_ecg": "Exercise/ECG Removed",
    "drop_imaging_perf": "Imaging/Perfusion Removed",
    "drop_all_diagnostics": "Diagnostics Removed",
}
MATCHED_GROUP_REMOVAL = {
    "drop_risk": ["age", "trestbps", "chol"],
    "drop_exercise": ["thalach", "oldpeak"],
}

BASELINE_METRICS = ["auc", "brier", "log_loss", "ece", "mean_entropy", "mean_variance"]
SCENARIO_METRICS = [
    "auc",
    "brier",
    "log_loss",
    "ece",
    "mean_entropy",
    "mean_variance",
    "auc_drop",
    "brier_increase",
    "logloss_increase",
    "ece_increase",
    "delta_entropy",
    "delta_variance",
    "delta_prob_abs",
]
IMPROVEMENT_METRICS = [
    "auc_gain",
    "brier_gain",
    "logloss_gain",
    "ece_gain",
    "auc_drop_gain",
    "brier_robust_gain",
    "logloss_robust_gain",
    "ece_robust_gain",
    "delta_prob_reduction",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    level: str
    description: str
    build_infer: Callable[[pd.DataFrame], object]


class DiscreteInferenceWrapper:
    def __init__(self, infer: object, transform_value: Callable[[str, float], int]):
        self.infer = infer
        self.transform_value = transform_value

    def predict_proba_subset(self, row: pd.Series, observed_cols: list[str]) -> float:
        evidence = {column: int(self.transform_value(column, row[column])) for column in observed_cols}
        try:
            query = _safe_query(self.infer, evidence)
            return float(query.values[1])
        except Exception:
            return 0.5


class ExactMarginalGMMClassifier:
    def __init__(
        self,
        gmm0: GaussianMixture,
        gmm1: GaussianMixture,
        prior0: float,
        prior1: float,
        scaler: StandardScaler,
        feature_order: list[str],
    ):
        self.gmm0 = gmm0
        self.gmm1 = gmm1
        self.prior0 = float(prior0)
        self.prior1 = float(prior1)
        self.scaler = scaler
        self.feature_order = list(feature_order)
        self.feature_index = {feature: idx for idx, feature in enumerate(self.feature_order)}

    @staticmethod
    def _log_gaussian_subset(x_obs: np.ndarray, mean_obs: np.ndarray, cov_obs: np.ndarray) -> float:
        cov = np.asarray(cov_obs, dtype=float) + 1e-6 * np.eye(len(mean_obs))
        diff = np.asarray(x_obs, dtype=float) - np.asarray(mean_obs, dtype=float)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 1e-4 * np.eye(len(mean_obs))
            sign, logdet = np.linalg.slogdet(cov)
        mahal = diff @ np.linalg.solve(cov, diff)
        return float(-0.5 * (len(mean_obs) * np.log(2 * np.pi) + logdet + mahal))

    def _class_log_likelihood(self, gmm: GaussianMixture, x_obs: np.ndarray, obs_idx: list[int]) -> float:
        terms = []
        for k in range(gmm.n_components):
            mean_obs = gmm.means_[k, obs_idx]
            cov_obs = gmm.covariances_[k][np.ix_(obs_idx, obs_idx)]
            terms.append(
                np.log(gmm.weights_[k] + 1e-300)
                + self._log_gaussian_subset(x_obs, mean_obs, cov_obs)
            )
        return float(logsumexp(terms))

    def predict_proba_subset(self, row: pd.Series, observed_cols: list[str]) -> float:
        if not observed_cols:
            return self.prior1

        obs_idx = [self.feature_index[column] for column in observed_cols]
        x_obs_raw = np.array([float(row[column]) for column in observed_cols], dtype=float)
        x_obs = (x_obs_raw - self.scaler.mean_[obs_idx]) / self.scaler.scale_[obs_idx]

        log_p0 = np.log(self.prior0 + 1e-300) + self._class_log_likelihood(self.gmm0, x_obs, obs_idx)
        log_p1 = np.log(self.prior1 + 1e-300) + self._class_log_likelihood(self.gmm1, x_obs, obs_idx)
        log_norm = float(logsumexp([log_p0, log_p1]))
        return float(np.exp(log_p1 - log_norm))


def load_discretized_heart_data(csv_path: str = "heart_disease_cleaned.csv") -> pd.DataFrame:
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file)
    df_disc = df.copy()

    df_disc["num"] = (df_disc["num"] > 0).astype(int)
    df_disc["age"] = pd.cut(df_disc["age"], bins=[0, 45, 60, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["trestbps"] = (df_disc["trestbps"] >= 130).astype(int)
    df_disc["chol"] = pd.cut(df_disc["chol"], bins=[0, 199, 240, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["thalach"] = pd.cut(df_disc["thalach"], bins=[0, 119, 150, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["oldpeak"] = pd.cut(df_disc["oldpeak"], bins=[-0.001, 1.0, 2.0, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["cp"] = (df_disc["cp"] - 1).astype(int)
    df_disc["slope"] = (df_disc["slope"] - 1).astype(int)
    df_disc["thal"] = df_disc["thal"].astype(float).round().astype(int).map({3: 0, 6: 1, 7: 2})

    for col in ["ca", "sex", "fbs", "restecg", "exang"]:
        df_disc[col] = df_disc[col].astype(float).round().astype(int)

    assert df_disc.isnull().sum().sum() == 0
    return df_disc


def load_matched_heart_data(csv_path: str = "heart_disease_cleaned.csv") -> pd.DataFrame:
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file)
    df["num"] = (df["num"] > 0).astype(int)
    return df[MATCHED_FEATURES + ["num"]].copy()


def discretize_matched_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    df_disc = df.copy()
    df_disc["num"] = (df_disc["num"] > 0).astype(int)
    df_disc["age"] = pd.cut(df_disc["age"], bins=[0, 45, 60, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["trestbps"] = (df_disc["trestbps"] >= 130).astype(int)
    df_disc["chol"] = pd.cut(df_disc["chol"], bins=[0, 199, 240, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["thalach"] = pd.cut(df_disc["thalach"], bins=[0, 119, 150, float("inf")], labels=[0, 1, 2]).astype(int)
    df_disc["oldpeak"] = pd.cut(df_disc["oldpeak"], bins=[-0.001, 1.0, 2.0, float("inf")], labels=[0, 1, 2]).astype(int)
    return df_disc[MATCHED_FEATURES + ["num"]].copy()


def discretize_matched_value(feature: str, value: float) -> int:
    value = float(value)
    if feature == "age":
        return 0 if value <= 45 else 1 if value <= 60 else 2
    if feature == "trestbps":
        return int(value >= 130)
    if feature == "chol":
        return 0 if value <= 199 else 1 if value <= 240 else 2
    if feature == "thalach":
        return 0 if value <= 119 else 1 if value <= 150 else 2
    if feature == "oldpeak":
        return 0 if value <= 1.0 else 1 if value <= 2.0 else 2
    raise KeyError(f"Unknown matched feature: {feature}")


MATCHED_STATE_NAMES = {key: value for key, value in STATE_NAMES.items() if key in MATCHED_FEATURES + ["num"]}


def model_definition_table(model_specs: dict[str, ModelSpec] | None = None) -> pd.DataFrame:
    rows = []
    for spec in (model_specs or get_model_specs()).values():
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "level": spec.level,
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def build_bn_infer(
    train_df: pd.DataFrame,
    edge_list: list[tuple[str, str]],
    equivalent_sample_size: float = 10.0,
    state_names: dict[str, list[int]] = STATE_NAMES,
) -> VariableElimination:
    model = DiscreteBayesianNetwork(edge_list)
    model.fit(
        train_df,
        estimator=BayesianEstimator,
        state_names=state_names,
        prior_type="BDeu",
        equivalent_sample_size=equivalent_sample_size,
    )
    return VariableElimination(model)


def unary_potential(data: pd.DataFrame, var: str, states: list[int], alpha: float = 1.0) -> np.ndarray:
    counts = data[var].value_counts().reindex(states, fill_value=0).astype(float).values
    return (counts + alpha) / (counts.sum() + alpha * len(states))


def pairwise_potential(
    data: pd.DataFrame,
    u: str,
    v: str,
    states_u: list[int],
    states_v: list[int],
    alpha: float = 1.0,
) -> np.ndarray:
    table = pd.crosstab(data[u], data[v], dropna=False)
    table = table.reindex(index=states_u, columns=states_v, fill_value=0).astype(float)
    joint = table.values + alpha
    return joint / joint.sum()


def compute_edge_mi(data: pd.DataFrame, edge_list: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    mi_vals: dict[tuple[str, str], float] = {}
    for u, v in edge_list:
        mi_vals[tuple(sorted((u, v)))] = float(mutual_info_score(data[u], data[v]))

    max_mi = max(mi_vals.values()) if mi_vals else 1.0
    if max_mi <= 1e-12:
        return {key: 0.0 for key in mi_vals}
    return {key: value / max_mi for key, value in mi_vals.items()}


def build_mrf_infer(
    train_df: pd.DataFrame,
    edge_list: list[tuple[str, str]],
    alpha: float = 1.0,
    gamma: float = 0.0,
    feature_columns: list[str] | None = None,
    state_names: dict[str, list[int]] = STATE_NAMES,
) -> BeliefPropagation:
    feature_columns = feature_columns or ALL_FEATURES
    edge_weights = compute_edge_mi(train_df, edge_list) if gamma > 0 else None

    mrf = DiscreteMarkovNetwork()
    mrf.add_nodes_from(feature_columns + ["num"])
    mrf.add_edges_from(edge_list)

    for node in feature_columns + ["num"]:
        phi = unary_potential(train_df, node, state_names[node], alpha=alpha)
        mrf.add_factors(
            DiscreteFactor(
                variables=[node],
                cardinality=[len(state_names[node])],
                values=phi,
                state_names={node: state_names[node]},
            )
        )

    for u, v in edge_list:
        phi_uv = pairwise_potential(train_df, u, v, state_names[u], state_names[v], alpha=alpha)
        if edge_weights is not None and gamma > 0:
            weight = edge_weights.get(tuple(sorted((u, v))), 0.0)
            phi_uv = np.power(phi_uv, 1.0 + gamma * weight)
            phi_uv = phi_uv / phi_uv.sum()

        mrf.add_factors(
            DiscreteFactor(
                variables=[u, v],
                cardinality=[len(state_names[u]), len(state_names[v])],
                values=phi_uv,
                state_names={u: state_names[u], v: state_names[v]},
            )
        )

    assert mrf.check_model()
    return BeliefPropagation(mrf)


def get_model_specs() -> dict[str, ModelSpec]:
    return {
        "Base BN": ModelSpec(
            name="Base BN",
            family="BN",
            level="Base",
            description="Original 13-edge DAG with BDeu parameter learning",
            build_infer=lambda train_df: build_bn_infer(train_df, BASE_EDGES, equivalent_sample_size=10.0),
        ),
        "Best BN": ModelSpec(
            name="Best BN",
            family="BN",
            level="Best",
            description="Hill-climbing learned DAG with BDeu parameter learning",
            build_infer=lambda train_df: build_bn_infer(train_df, BEST_BN_EDGES, equivalent_sample_size=10.0),
        ),
        "Base MRF": ModelSpec(
            name="Base MRF",
            family="MRF",
            level="Base",
            description="Original 13-edge undirected graph with alpha=1.0 and gamma=0.0",
            build_infer=lambda train_df: build_mrf_infer(train_df, BASE_EDGES, alpha=1.0, gamma=0.0),
        ),
        "Best MRF": ModelSpec(
            name="Best MRF",
            family="MRF",
            level="Best",
            description="Tuned 13-edge undirected graph with alpha=10.0 and gamma=1.5",
            build_infer=lambda train_df: build_mrf_infer(train_df, BASE_EDGES, alpha=10.0, gamma=1.5),
        ),
    }


def build_matched_bn_infer(train_df: pd.DataFrame, equivalent_sample_size: float = 10.0) -> DiscreteInferenceWrapper:
    train_disc = discretize_matched_heart_data(train_df)
    infer = build_bn_infer(
        train_disc,
        MATCHED_EDGES,
        equivalent_sample_size=equivalent_sample_size,
        state_names=MATCHED_STATE_NAMES,
    )
    return DiscreteInferenceWrapper(infer, discretize_matched_value)


def build_matched_mrf_infer(train_df: pd.DataFrame, alpha: float = 1.0, gamma: float = 0.0) -> DiscreteInferenceWrapper:
    train_disc = discretize_matched_heart_data(train_df)
    infer = build_mrf_infer(
        train_disc,
        MATCHED_EDGES,
        alpha=alpha,
        gamma=gamma,
        feature_columns=MATCHED_FEATURES,
        state_names=MATCHED_STATE_NAMES,
    )
    return DiscreteInferenceWrapper(infer, discretize_matched_value)


def build_gmm_exact_infer(train_df: pd.DataFrame, n_components: int = 2) -> ExactMarginalGMMClassifier:
    X_train = train_df[MATCHED_FEATURES].values
    y_train = train_df["num"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    prior0 = float((y_train == 0).mean())
    prior1 = float((y_train == 1).mean())
    gmm0 = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        max_iter=300,
        n_init=5,
        reg_covar=1e-6,
    ).fit(X_train_scaled[y_train == 0])
    gmm1 = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        max_iter=300,
        n_init=5,
        reg_covar=1e-6,
    ).fit(X_train_scaled[y_train == 1])

    return ExactMarginalGMMClassifier(gmm0, gmm1, prior0, prior1, scaler, MATCHED_FEATURES)


def get_matched_model_specs(gmm_components: int = 2) -> dict[str, ModelSpec]:
    return {
        "Matched BN": ModelSpec(
            name="Matched BN",
            family="BN",
            level="Matched",
            description="5-feature BN on age/trestbps/chol -> num -> thalach/oldpeak",
            build_infer=lambda train_df: build_matched_bn_infer(train_df, equivalent_sample_size=10.0),
        ),
        "Matched MRF": ModelSpec(
            name="Matched MRF",
            family="MRF",
            level="Matched",
            description="5-feature MRF on the same skeleton as the matched BN",
            build_infer=lambda train_df: build_matched_mrf_infer(train_df, alpha=1.0, gamma=0.0),
        ),
        "GMM": ModelSpec(
            name="GMM",
            family="GMM",
            level="Matched",
            description="Class-conditional 2-component GMM with exact marginalization at test time",
            build_infer=lambda train_df: build_gmm_exact_infer(train_df, n_components=gmm_components),
        ),
    }


def _safe_query(infer: object, evidence: dict[str, int]):
    try:
        return infer.query(variables=["num"], evidence=evidence, show_progress=False)
    except TypeError:
        return infer.query(variables=["num"], evidence=evidence)


def predict_row(infer: object, row: pd.Series, observed_cols: list[str]) -> float:
    if hasattr(infer, "predict_proba_subset"):
        try:
            return float(infer.predict_proba_subset(row, observed_cols))
        except Exception:
            return 0.5
    evidence = {column: int(row[column]) for column in observed_cols}
    try:
        query = _safe_query(infer, evidence)
        return float(query.values[1])
    except Exception:
        return 0.5


def predict_dataset(infer: object, test_df: pd.DataFrame, observed_cols: list[str]) -> np.ndarray:
    return np.array([predict_row(infer, row, observed_cols) for _, row in test_df.iterrows()])


def clip_probs(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities), 1e-10, 1 - 1e-10)


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = clip_probs(probabilities)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def binary_variance(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities)
    return p * (1 - p)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if mask.sum() == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return float(ece / len(y_true))


def summarize_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    p = clip_probs(probabilities)
    y = np.asarray(y_true)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "mean_entropy": float(binary_entropy(p).mean()),
        "mean_variance": float(binary_variance(p).mean()),
    }


def add_relative_metrics(
    summary: dict[str, float],
    full_summary: dict[str, float],
    full_probabilities: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    out = dict(summary)
    out["auc_drop"] = summary["auc"] - full_summary["auc"]
    out["brier_increase"] = summary["brier"] - full_summary["brier"]
    out["logloss_increase"] = summary["log_loss"] - full_summary["log_loss"]
    out["ece_increase"] = summary["ece"] - full_summary["ece"]
    out["delta_entropy"] = summary["mean_entropy"] - full_summary["mean_entropy"]
    out["delta_variance"] = summary["mean_variance"] - full_summary["mean_variance"]
    out["delta_prob_abs"] = float(np.mean(np.abs(clip_probs(probabilities) - clip_probs(full_probabilities))))
    return out


def run_full_baseline(
    df_disc: pd.DataFrame,
    model_specs: dict[str, ModelSpec] | None = None,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    model_specs = model_specs or get_model_specs()
    feature_columns = feature_columns or ALL_FEATURES
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df_disc[feature_columns], df_disc["num"]), start=1):
        train_df = df_disc.iloc[train_idx].reset_index(drop=True)
        test_df = df_disc.iloc[test_idx].reset_index(drop=True)
        y_true = test_df["num"].values

        for spec in model_specs.values():
            infer = spec.build_infer(train_df)
            probabilities = predict_dataset(infer, test_df, feature_columns)
            summary = summarize_predictions(y_true, probabilities)
            rows.append(
                {
                    "fold": fold,
                    "model": spec.name,
                    "family": spec.family,
                    "level": spec.level,
                    **summary,
                }
            )

    return pd.DataFrame(rows)


def run_missing_scenarios(
    df_disc: pd.DataFrame,
    scenarios: dict[str, list[str]],
    experiment_name: str,
    model_specs: dict[str, ModelSpec] | None = None,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    model_specs = model_specs or get_model_specs()
    feature_columns = feature_columns or ALL_FEATURES
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df_disc[feature_columns], df_disc["num"]), start=1):
        train_df = df_disc.iloc[train_idx].reset_index(drop=True)
        test_df = df_disc.iloc[test_idx].reset_index(drop=True)
        y_true = test_df["num"].values

        infer_map = {}
        full_probabilities = {}
        full_summaries = {}
        for spec in model_specs.values():
            infer = spec.build_infer(train_df)
            infer_map[spec.name] = infer
            probs = predict_dataset(infer, test_df, feature_columns)
            infer_summary = summarize_predictions(y_true, probs)
            full_probabilities[spec.name] = probs
            full_summaries[spec.name] = infer_summary

        for scenario_name, removed_cols in scenarios.items():
            observed_cols = [col for col in feature_columns if col not in removed_cols]

            for spec in model_specs.values():
                probs = predict_dataset(infer_map[spec.name], test_df, observed_cols)
                summary = summarize_predictions(y_true, probs)
                summary = add_relative_metrics(
                    summary=summary,
                    full_summary=full_summaries[spec.name],
                    full_probabilities=full_probabilities[spec.name],
                    probabilities=probs,
                )

                rows.append(
                    {
                        "experiment": experiment_name,
                        "fold": fold,
                        "scenario": scenario_name,
                        "removed": ",".join(removed_cols),
                        "n_removed": len(removed_cols),
                        "model": spec.name,
                        "family": spec.family,
                        "level": spec.level,
                        **summary,
                    }
                )

    return pd.DataFrame(rows)


def run_random_missingness(
    df_disc: pd.DataFrame,
    feature_columns: list[str],
    model_specs: dict[str, ModelSpec],
    missing_counts: list[int] | None = None,
    repeats: int = 20,
    experiment_name: str = "random_missingness",
    random_state: int = 42,
) -> pd.DataFrame:
    missing_counts = missing_counts or list(range(len(feature_columns)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(random_state)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df_disc[feature_columns], df_disc["num"]), start=1):
        train_df = df_disc.iloc[train_idx].reset_index(drop=True)
        test_df = df_disc.iloc[test_idx].reset_index(drop=True)
        y_true = test_df["num"].values

        infer_map = {}
        full_probabilities = {}
        full_summaries = {}
        for spec in model_specs.values():
            infer = spec.build_infer(train_df)
            infer_map[spec.name] = infer
            probs = predict_dataset(infer, test_df, feature_columns)
            full_probabilities[spec.name] = probs
            full_summaries[spec.name] = summarize_predictions(y_true, probs)

        for n_removed in missing_counts:
            if n_removed < 0 or n_removed > len(feature_columns):
                raise ValueError(f"Invalid n_removed={n_removed} for {len(feature_columns)} features")

            for repeat in range(repeats):
                sampled_removed = []
                for _ in range(len(test_df)):
                    if n_removed == 0:
                        sampled_removed.append([])
                    else:
                        sampled_removed.append(sorted(rng.choice(feature_columns, size=n_removed, replace=False).tolist()))

                for spec in model_specs.values():
                    probs = []
                    for (_, row), removed_cols in zip(test_df.iterrows(), sampled_removed):
                        observed_cols = [col for col in feature_columns if col not in removed_cols]
                        probs.append(predict_row(infer_map[spec.name], row, observed_cols))
                    probs_arr = np.array(probs)
                    summary = summarize_predictions(y_true, probs_arr)
                    summary = add_relative_metrics(
                        summary=summary,
                        full_summary=full_summaries[spec.name],
                        full_probabilities=full_probabilities[spec.name],
                        probabilities=probs_arr,
                    )
                    rows.append(
                        {
                            "experiment": experiment_name,
                            "fold": fold,
                            "repeat": repeat,
                            "scenario": f"random_{n_removed}",
                            "removed": "<random>",
                            "n_removed": n_removed,
                            "model": spec.name,
                            "family": spec.family,
                            "level": spec.level,
                            **summary,
                        }
                    )

    return pd.DataFrame(rows)


def summarize_baseline_results(
    baseline_df: pd.DataFrame,
    model_order: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    metrics = metrics or BASELINE_METRICS
    summary = baseline_df.groupby("model")[metrics].mean()
    if model_order is not None:
        summary = summary.reindex(model_order)
    return summary.round(4)


def summarize_scenario_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.groupby(["scenario", "model"])[SCENARIO_METRICS].mean().round(4)


def summarize_random_missingness_results(
    results_df: pd.DataFrame,
    metrics: list[str] | None = None,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    metrics = metrics or SCENARIO_METRICS
    summary = results_df.groupby(["n_removed", "model"])[metrics].mean().round(4)
    if model_order is None:
        return summary
    ordered_pieces = []
    for n_removed in sorted(results_df["n_removed"].unique()):
        slice_df = summary.loc[n_removed].reindex(model_order)
        slice_df["n_removed"] = n_removed
        ordered_pieces.append(slice_df.reset_index().set_index(["n_removed", "model"]))
    return pd.concat(ordered_pieces)


def compute_within_family_improvement(results_df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["experiment", "scenario", "removed", "n_removed", "fold"]
    rows = []

    for family in FAMILY_ORDER:
        base_df = results_df[(results_df["family"] == family) & (results_df["level"] == "Base")]
        best_df = results_df[(results_df["family"] == family) & (results_df["level"] == "Best")]

        merged = base_df.merge(best_df, on=merge_keys, suffixes=("_base", "_best"))

        for _, row in merged.iterrows():
            rows.append(
                {
                    "experiment": row["experiment"],
                    "fold": row["fold"],
                    "scenario": row["scenario"],
                    "removed": row["removed"],
                    "n_removed": row["n_removed"],
                    "family": family,
                    "pair": f"Base-to-Best {family}",
                    "auc_gain": row["auc_best"] - row["auc_base"],
                    "brier_gain": row["brier_base"] - row["brier_best"],
                    "logloss_gain": row["log_loss_base"] - row["log_loss_best"],
                    "ece_gain": row["ece_base"] - row["ece_best"],
                    "auc_drop_gain": row["auc_drop_best"] - row["auc_drop_base"],
                    "brier_robust_gain": row["brier_increase_base"] - row["brier_increase_best"],
                    "logloss_robust_gain": row["logloss_increase_base"] - row["logloss_increase_best"],
                    "ece_robust_gain": row["ece_increase_base"] - row["ece_increase_best"],
                    "delta_prob_reduction": row["delta_prob_abs_base"] - row["delta_prob_abs_best"],
                }
            )

    return pd.DataFrame(rows)


def summarize_improvements(improvement_df: pd.DataFrame) -> pd.DataFrame:
    return improvement_df.groupby(["experiment", "scenario", "family"])[IMPROVEMENT_METRICS].mean().round(4)


def plot_single_feature_heatmap(
    results_df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str = "RdYlGn",
    center: float = 0.0,
    model_order: list[str] | None = None,
    feature_order: list[str] | None = None,
) -> None:
    plot_df = results_df.copy()
    plot_df["feature"] = plot_df["removed"]
    matrix = (
        plot_df.groupby(["feature", "model"])[value_col]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="model", values=value_col)
        .reindex(
            index=feature_order or ALL_FEATURES,
            columns=model_order or MODEL_ORDER,
        )
    )
    matrix = matrix.rename(columns=MODEL_DISPLAY_LABELS)

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap=cmap, center=center, linewidths=0.5)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Removed feature")
    plt.tight_layout()
    plt.show()


def plot_group_bars(
    results_df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    scenario_order: list[str] | None = None,
    model_order: list[str] | None = None,
) -> None:
    plot_df = results_df.groupby(["scenario", "model"])[value_col].mean().reset_index()
    scenario_order = scenario_order or list(CLINICAL_GROUP_REMOVAL.keys())
    scenario_labels = [CLINICAL_GROUP_LABELS.get(scenario, scenario) for scenario in scenario_order]

    plot_df["scenario"] = pd.Categorical(
        plot_df["scenario"],
        scenario_order,
        ordered=True,
    )
    plot_df["model"] = pd.Categorical(plot_df["model"], model_order or MODEL_ORDER, ordered=True)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=plot_df.sort_values(["scenario", "model"]), x="scenario", y=value_col, hue="model")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scenario")
    ax.set_xticklabels(scenario_labels, rotation=20, ha="center")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            label = text.get_text()
            text.set_text(MODEL_DISPLAY_LABELS.get(label, label))
    plt.tight_layout()
    plt.show()


def plot_random_missingness_curves(
    results_df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    model_order: list[str] | None = None,
    colors: dict[str, str] | None = None,
) -> None:
    model_order = model_order or MATCHED_MODEL_ORDER
    default_colors = {
        "Base BN": "#4C72B0",
        "Best BN": "#0D6EFD",
        "Base MRF": "#C44E52",
        "Best MRF": "#E63946",
        "Matched BN": "#4C72B0",
        "Matched MRF": "#C44E52",
        "GMM": "#7B61FF",
    }
    colors = colors or default_colors

    summary = (
        results_df.groupby(["n_removed", "model"])[value_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["model"] = pd.Categorical(summary["model"], model_order, ordered=True)
    summary = summary.sort_values(["model", "n_removed"])

    plt.figure(figsize=(8.5, 5))
    for model in model_order:
        sub = summary[summary["model"] == model]
        if sub.empty:
            continue
        x = sub["n_removed"].to_numpy()
        y = sub["mean"].to_numpy()
        err = np.nan_to_num(sub["std"].to_numpy(), nan=0.0)
        plt.plot(x, y, marker="o", linewidth=2, label=MODEL_DISPLAY_LABELS.get(model, model), color=colors.get(model))
        plt.fill_between(x, y - err, y + err, alpha=0.15, color=colors.get(model))

    plt.title(title)
    plt.xlabel("Number of randomly removed features")
    plt.ylabel(ylabel)
    x_values = sorted(results_df["n_removed"].unique())
    if x_values and all(float(x).is_integer() for x in x_values):
        plt.xticks(range(int(min(x_values)), int(max(x_values)) + 1))
    else:
        plt.xticks(x_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_within_family_single_heatmap(
    improvement_df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str = "RdYlGn",
    center: float = 0.0,
) -> None:
    plot_df = improvement_df[improvement_df["experiment"] == "single_feature"].copy()
    plot_df["feature"] = plot_df["removed"]
    matrix = (
        plot_df.groupby(["feature", "family"])[value_col]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="family", values=value_col)
        .reindex(index=ALL_FEATURES, columns=FAMILY_ORDER)
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap=cmap, center=center, linewidths=0.5)
    plt.title(title)
    plt.xlabel("Family")
    plt.ylabel("Removed feature")
    plt.tight_layout()
    plt.show()


def plot_within_family_group_bars(
    improvement_df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = (
        improvement_df[improvement_df["experiment"] == "clinical_group"]
        .groupby(["scenario", "family"])[value_col]
        .mean()
        .reset_index()
    )
    plot_df["scenario"] = pd.Categorical(plot_df["scenario"], list(CLINICAL_GROUP_REMOVAL.keys()), ordered=True)
    plot_df["family"] = pd.Categorical(plot_df["family"], FAMILY_ORDER, ordered=True)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_df.sort_values(["scenario", "family"]), x="scenario", y=value_col, hue="family")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Scenario")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
