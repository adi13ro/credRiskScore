import numpy as np
import pandas as pd

from creditrisk import config, scoring


def test_score_scale_anchor_and_pdo():
    base_odds = config.SCORE_BASE_ODDS
    # PD at the base odds lands exactly on the base score
    assert scoring.pd_to_score(1 / (1 + base_odds))[0] == config.SCORE_BASE
    # doubling the odds adds one PDO
    assert scoring.pd_to_score(1 / (1 + 2 * base_odds))[0] == config.SCORE_BASE + config.SCORE_PDO
    s = scoring.pd_to_score(np.array([0.01, 0.1, 0.5, 0.9]))
    assert list(s) == sorted(s, reverse=True)


def test_band_lookup_edges():
    assert scoring.score_to_band(700) == "A"
    assert scoring.score_to_band(699) == "B"
    assert scoring.score_to_band(650) == "B"
    assert scoring.score_to_band(600) == "C"
    assert scoring.score_to_band(550) == "D"
    assert scoring.score_to_band(549) == "E"
    assert scoring.score_to_band(-50) == "E"


def test_band_table_and_indicators():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "application_id": range(n),
        "checking_status": rng.choice(["< 0 DM", "no checking account"], n),
        "savings_status": rng.choice(["< 100 DM", ">= 1000 DM"], n),
        "credit_history": rng.choice(["past payment delays", "existing paid duly"], n),
        "installment_rate_pct": rng.integers(1, 5, n),
        "duration_months": rng.integers(6, 60, n),
        "credit_amount": rng.integers(500, 15000, n),
        "employment_since": rng.choice(["unemployed", "1-4 years"], n),
        "age": rng.integers(19, 70, n),
        "other_installment_plans": rng.choice(["none", "bank"], n),
        "property": rng.choice(["none", "real estate"], n),
    })
    # make overdrawn accounts genuinely riskier so the indicator test has signal
    risk = 0.15 + 0.4 * (df["checking_status"] == "< 0 DM")
    df["default"] = (rng.random(n) < risk).astype(int)
    df["p_default"] = np.clip(risk + rng.normal(0, 0.05, n), 0.01, 0.99)
    df["score"] = scoring.pd_to_score(df["p_default"])
    df["band"] = df["score"].map(scoring.score_to_band)

    bands = scoring.band_table(df)
    assert list(bands["band"]) == ["A", "B", "C", "D", "E"]
    assert bands["applications"].sum() == n
    assert abs(bands["share"].sum() - 1) < 1e-9

    ind = scoring.indicator_report(df)
    row = ind.set_index("indicator").loc["overdrawn checking account"]
    assert row["observed_lift"] > 1 and row["predicted_lift"] > 1 and row["model_agrees"]

    flags = scoring.flags_for(df.iloc[0], df)
    assert isinstance(flags, list)
