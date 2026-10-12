"""Statlog German Credit data (UCI, 1,000 loan applications, 30 % defaults).

Codes like A11 are decoded to readable categories. Falls back to a synthetic frame with
the same schema when the download fails so the pipeline still runs offline.
"""

from __future__ import annotations

import io
import logging
import zipfile

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate_pct", "personal_status_sex",
    "other_debtors", "residence_since_years", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
    "target",
]

NUMERIC = ["duration_months", "credit_amount", "installment_rate_pct", "residence_since_years",
           "age", "existing_credits", "num_dependents"]
CATEGORICAL = ["checking_status", "credit_history", "purpose", "savings_status",
               "employment_since", "other_debtors", "property", "other_installment_plans",
               "housing", "job", "telephone"]
# Excluded from the model on fairness grounds; kept in the data for monitoring.
PROTECTED = ["personal_status_sex", "foreign_worker"]
TARGET = "default"

CODES = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM",
                        "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank",
                       "A32": "existing paid duly", "A33": "past payment delays",
                       "A34": "critical / other credits"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture", "A43": "radio/tv",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "other"},
    "savings_status": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                       "A64": ">= 1000 DM", "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
                         "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male divorced", "A92": "female div/married",
                            "A93": "male single", "A94": "male married", "A95": "female single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings / insurance", "A123": "car / other",
                 "A124": "none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unskilled non-resident", "A172": "unskilled resident", "A173": "skilled",
            "A174": "management / self-employed"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def download_uci(url: str = config.UCI_URL, timeout: int = 90) -> pd.DataFrame:
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf, zf.open("german.data") as fh:
        df = pd.read_csv(fh, sep=r"\s+", header=None, names=COLUMNS)
    return _decode(df)


def _decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in CODES.items():
        out[col] = out[col].map(mapping).fillna(out[col])
    out[TARGET] = (out["target"] == 2).astype(int)   # 2 = bad in Statlog
    out = out.drop(columns="target")
    out.insert(0, "application_id", range(1, len(out) + 1))
    return out


def synthetic(n: int = 1000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Schema-compatible fallback with plausible correlations, not a substitute for real data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "duration_months": rng.integers(4, 72, n),
        "credit_amount": np.round(rng.lognormal(7.8, 0.7, n)).astype(int),
        "installment_rate_pct": rng.integers(1, 5, n),
        "residence_since_years": rng.integers(1, 5, n),
        "age": rng.integers(19, 75, n),
        "existing_credits": rng.integers(1, 4, n),
        "num_dependents": rng.integers(1, 3, n),
    })
    for col, mapping in CODES.items():
        df[col] = rng.choice(list(mapping.values()), n)
    logit = (-1.4 + 0.02 * df["duration_months"] + 0.3 * df["installment_rate_pct"]
             - 0.02 * df["age"] + (df["checking_status"] == "< 0 DM") * 0.9
             + (df["credit_history"] == "critical / other credits") * -0.6
             + (df["savings_status"] == "< 100 DM") * 0.4)
    df["target"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), 2, 1)
    return _decode(df)


def load_raw(force_download: bool = False, allow_synthetic: bool = True) -> pd.DataFrame:
    path = config.RAW_CSV
    if path.exists() and not force_download:
        return pd.read_csv(path)
    try:
        df = download_uci()
        df["origin"] = "uci"
    except Exception as exc:
        if not allow_synthetic:
            raise
        log.warning("Download failed (%s); using synthetic data", exc)
        df = synthetic()
        df["origin"] = "synthetic"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d applications to %s", len(df), path)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Schema and range checks. Raises on anything a model must not silently accept."""
    missing = set(NUMERIC + CATEGORICAL + PROTECTED + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[NUMERIC + CATEGORICAL].isna().any().any():
        raise ValueError("Null values in model features")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be 0/1")
    if (df["age"] < 18).any() or (df["credit_amount"] <= 0).any() or \
            (df["duration_months"] <= 0).any():
        raise ValueError("Out-of-range values in age, credit_amount or duration_months")
    return df
