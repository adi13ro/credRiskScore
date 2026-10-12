"""Feature engineering and Weight-of-Evidence / Information Value analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, TARGET

ENGINEERED = ["monthly_payment", "amount_per_year_age"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["monthly_payment"] = out["credit_amount"] / out["duration_months"]
    out["amount_per_year_age"] = out["credit_amount"] / out["age"]
    return out


def feature_columns() -> tuple[list[str], list[str]]:
    return NUMERIC + ENGINEERED, CATEGORICAL


def preprocessor() -> ColumnTransformer:
    num, cat = feature_columns()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def woe_table(df: pd.DataFrame, feature: str, bins: int = 5) -> pd.DataFrame:
    """WoE and IV contribution per category (numerics are quantile-binned)."""
    col = df[feature]
    if feature in NUMERIC + ENGINEERED:
        col = pd.qcut(col, q=bins, duplicates="drop").astype(str)
    g = df.groupby(col, observed=True)[TARGET].agg(["count", "sum"])
    g.columns = ["n", "bad"]
    g["good"] = g["n"] - g["bad"]
    total_good, total_bad = g["good"].sum(), g["bad"].sum()
    eps = 0.5
    dist_good = (g["good"] + eps) / (total_good + eps * len(g))
    dist_bad = (g["bad"] + eps) / (total_bad + eps * len(g))
    g["default_rate"] = g["bad"] / g["n"]
    g["woe"] = np.log(dist_good / dist_bad)
    g["iv"] = (dist_good - dist_bad) * g["woe"]
    return g.reset_index().rename(columns={feature: "value"}).assign(feature=feature)


def information_value(df: pd.DataFrame) -> pd.DataFrame:
    num, cat = feature_columns()
    rows = []
    for f in num + cat:
        iv = woe_table(df, f)["iv"].sum()
        rows.append({"feature": f, "iv": round(float(iv), 4), "strength": _iv_label(iv)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def _iv_label(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"
