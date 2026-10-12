from creditrisk import data, features, model


def test_train_champion_challenger(tmp_path):
    df = features.engineer(data.synthetic(n=500))
    pipe, report, test_idx = model.train(df, test_size=0.3)
    assert report.champion in report.models
    assert set(report.models) == {"logistic_regression", "gradient_boosting"}
    for m in report.models.values():
        assert 0.5 < m["roc_auc"] <= 1.0
        assert 0 <= m["ks"] <= 1
        assert 0.05 <= m["threshold"] <= 0.95
    assert len(test_idx) == 150

    model.save(pipe, report, tmp_path / "m.joblib", tmp_path / "m.json")
    loaded = model.load(tmp_path / "m.joblib")
    num, cat = features.feature_columns()
    p = loaded.predict_proba(df[num + cat].head(3))[:, 1]
    assert ((0 <= p) & (p <= 1)).all()

    coefs = model.coefficients(model.candidates()["logistic_regression"].fit(
        df[num + cat], df[data.TARGET]))
    assert not coefs.empty and "checking_status" in " ".join(coefs["feature"])
