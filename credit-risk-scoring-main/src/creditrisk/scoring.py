"""Probability of default -> scorecard points, risk bands and rule-based risk indicators."""

from __future__ import annotations

import bisect
import math

import numpy as np
import pandas as pd

from . import config

# Band lower bounds (inclusive) in ascending order; bisect finds the band in O(log n).
BAND_EDGES = [0, 550, 600, 650, 700]
BAND_NAMES = ["E", "D", "C", "B", "A"]
BAND_ACTION = {"A": "auto-approve", "B": "approve", "C": "manual review",
               "D": "decline (appeal to senior underwriter)", "E": "decline"}

_FACTOR = config.SCORE_PDO / math.log(2)
_OFFSET = config.SCORE_BASE - _FACTOR * math.log(config.SCORE_BASE_ODDS)


def pd_to_score(p_default: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.atleast_1d(np.asarray(p_default, dtype=float)), 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    return np.round(_OFFSET + _FACTOR * np.log(odds_good)).astype(int)


def score_to_band(score: int) -> str:
    return BAND_NAMES[max(bisect.bisect_right(BAND_EDGES, score) - 1, 0)]


def band_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("band", observed=True).agg(
        applications=("application_id", "count"),
        defaults=("default", "sum"),
        mean_pd=("p_default", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reindex(BAND_NAMES[::-1]).fillna(0)
    g["default_rate"] = (g["defaults"] / g["applications"].replace(0, np.nan)).fillna(0)
    g["share"] = g["applications"] / g["applications"].sum()
    g["action"] = [BAND_ACTION[b] for b in g.index]
    return g.reset_index()


# ---------------- risk indicators ----------------
# Classic underwriting red flags. The pipeline checks the model agrees with each one:
# applicants carrying the flag should have both a higher observed default rate and a
# higher mean predicted PD than the portfolio.
RISK_INDICATORS: dict[str, callable] = {
    "overdrawn checking account": lambda d: d["checking_status"] == "< 0 DM",
    "no savings": lambda d: d["savings_status"] == "< 100 DM",
    "past payment delays": lambda d: d["credit_history"] == "past payment delays",
    "installment >= 4% of income": lambda d: d["installment_rate_pct"] >= 4,
    "loan longer than 36 months": lambda d: d["duration_months"] > 36,
    "large loan (top decile)": lambda d: d["credit_amount"] >= d["credit_amount"].quantile(0.9),
    "unemployed": lambda d: d["employment_since"] == "unemployed",
    "age under 25": lambda d: d["age"] < 25,
    "other installment plans": lambda d: d["other_installment_plans"] != "none",
    "no property": lambda d: d["property"] == "none",
}


def indicator_report(df: pd.DataFrame) -> pd.DataFrame:
    base_rate = df["default"].mean()
    base_pd = df["p_default"].mean()
    rows = []
    for name, rule in RISK_INDICATORS.items():
        mask = rule(df)
        sub = df[mask]
        if sub.empty:
            continue
        obs = sub["default"].mean()
        pred = sub["p_default"].mean()
        rows.append({
            "indicator": name,
            "applications": int(mask.sum()),
            "share": round(float(mask.mean()), 3),
            "observed_default_rate": round(float(obs), 3),
            "mean_predicted_pd": round(float(pred), 3),
            "observed_lift": round(float(obs / base_rate), 2),
            "predicted_lift": round(float(pred / base_pd), 2),
            "model_agrees": bool((obs > base_rate) == (pred > base_pd)),
        })
    return pd.DataFrame(rows).sort_values("observed_lift", ascending=False).reset_index(drop=True)


def flags_for(row: pd.Series, reference: pd.DataFrame) -> list[str]:
    """Risk indicators raised by one applicant (quantile rules use the reference portfolio)."""
    frame = pd.DataFrame([row])
    out = []
    for name, rule in RISK_INDICATORS.items():
        if "decile" in name:
            hit = row["credit_amount"] >= reference["credit_amount"].quantile(0.9)
        else:
            hit = bool(rule(frame).iloc[0])
        if hit:
            out.append(name)
    return out
