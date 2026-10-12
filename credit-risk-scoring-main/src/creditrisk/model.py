"""Champion/challenger training: logistic regression (scorecard) vs gradient boosting.

Models are trained without class re-weighting so predicted probabilities stay calibrated
to the portfolio default rate; the asymmetric cost matrix sets the decision threshold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data import TARGET
from .features import feature_columns, preprocessor


@dataclass
class ModelMetrics:
    name: str
    roc_auc: float
    gini: float
    ks: float
    brier: float
    cv_auc_mean: float
    cv_auc_std: float
    threshold: float
    precision: float
    recall: float
    expected_cost: float
    confusion: list[list[int]]
    roc: dict = field(default_factory=dict)
    ks_curve: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)


@dataclass
class TrainingReport:
    champion: str
    n_train: int
    n_test: int
    base_default_rate: float
    models: dict[str, dict]

    def to_dict(self) -> dict:
        return asdict(self)


def candidates(seed: int = config.RANDOM_SEED) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor()),
            ("clf", LogisticRegression(C=0.5, max_iter=5000, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", preprocessor()),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                   max_iter=300, l2_regularization=1.0,
                                                   random_state=seed)),
        ]),
    }


def ks_statistic(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    fpr, tpr, thr = roc_curve(y, p)
    ks = float(np.max(tpr - fpr))
    return ks, {"threshold": thr[1:].tolist(), "tpr": tpr[1:].tolist(), "fpr": fpr[1:].tolist()}


def cost_optimal_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold minimising Statlog's asymmetric cost, searched on a fine grid."""
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        cost = (fn * config.COST_FALSE_NEGATIVE + fp * config.COST_FALSE_POSITIVE) / len(y)
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return round(best_t, 3), round(best_cost, 4)


def evaluate(name: str, pipe: Pipeline, X_tr, y_tr, X_te, y_te, seed: int) -> ModelMetrics:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    y = np.asarray(y_te)
    auc = float(roc_auc_score(y, p))
    ks, ks_curve = ks_statistic(y, p)
    thr, cost = cost_optimal_threshold(y, p)
    pred = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    return ModelMetrics(
        name=name, roc_auc=round(auc, 4), gini=round(2 * auc - 1, 4), ks=round(ks, 4),
        brier=round(float(brier_score_loss(y, p)), 4),
        cv_auc_mean=round(float(cv_auc.mean()), 4), cv_auc_std=round(float(cv_auc.std()), 4),
        threshold=thr, precision=round(float(precision_score(y, pred)), 4),
        recall=round(float(recall_score(y, pred)), 4), expected_cost=cost,
        confusion=confusion_matrix(y, pred).tolist(),
        roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()}, ks_curve=ks_curve,
        calibration={"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()},
    )


def train(df: pd.DataFrame, seed: int = config.RANDOM_SEED,
          test_size: float = config.TEST_SIZE) -> tuple[Pipeline, TrainingReport, pd.Index]:
    num, cat = feature_columns()
    X, y = df[num + cat], df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed,
                                              stratify=y)
    results: dict[str, ModelMetrics] = {}
    pipes = candidates(seed)
    for name, pipe in pipes.items():
        results[name] = evaluate(name, pipe, X_tr, y_tr, X_te, y_te, seed)
    # Champion is chosen on cross-validated AUC (not the test split) to avoid selection leakage.
    champion = max(results, key=lambda n: results[n].cv_auc_mean)
    report = TrainingReport(
        champion=champion, n_train=len(X_tr), n_test=len(X_te),
        base_default_rate=round(float(y.mean()), 4),
        models={n: asdict(m) for n, m in results.items()},
    )
    return pipes[champion], report, X_te.index


def coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Standardised coefficients for the logistic model (empty for the boosting model)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient"])
    names = pipe.named_steps["prep"].get_feature_names_out()
    df = pd.DataFrame({"feature": names, "coefficient": clf.coef_[0]})
    df["feature"] = df["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index) \
             .reset_index(drop=True)


def save(pipe: Pipeline, report: TrainingReport, model_path: Path = config.MODEL_PATH,
         metrics_path: Path = config.METRICS_JSON) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(report.to_dict(), indent=2))


def load(model_path: Path = config.MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)
