import pytest

from creditrisk import data, features


def test_synthetic_has_schema_and_validates():
    df = data.synthetic(n=300)
    data.validate(df)
    assert len(df) == 300
    assert set(data.NUMERIC + data.CATEGORICAL + data.PROTECTED + [data.TARGET]) <= set(df.columns)
    assert df[data.TARGET].isin([0, 1]).all()
    assert 0.1 < df[data.TARGET].mean() < 0.6


def test_validate_rejects_bad_rows():
    df = data.synthetic(n=50)
    bad = df.copy()
    bad.loc[0, "age"] = 12
    with pytest.raises(ValueError):
        data.validate(bad)
    bad = df.drop(columns=["purpose"])
    with pytest.raises(ValueError):
        data.validate(bad)


def test_engineer_and_iv():
    df = features.engineer(data.synthetic(n=400))
    assert (df["monthly_payment"] > 0).all()
    iv = features.information_value(df)
    assert set(iv["feature"]) == set(features.feature_columns()[0] + features.feature_columns()[1])
    assert (iv["iv"] >= 0).all()
    # synthetic generator makes checking_status informative
    assert iv.set_index("feature").loc["checking_status", "iv"] > 0.02
    woe = features.woe_table(df, "duration_months")
    assert {"value", "n", "bad", "woe", "iv"} <= set(woe.columns)
