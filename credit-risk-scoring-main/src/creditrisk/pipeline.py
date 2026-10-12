"""End-to-end: acquire -> validate -> engineer -> train champion/challenger -> score -> report."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config, data, features, model, scoring

log = logging.getLogger("creditrisk.pipeline")


def run(force_download: bool = False, allow_synthetic: bool = True) -> dict:
    df = data.validate(data.load_raw(force_download, allow_synthetic))
    df = features.engineer(df)
    log.info("Loaded %d applications, default rate %.1f%%", len(df), 100 * df[data.TARGET].mean())

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    iv = features.information_value(df)
    iv.to_csv(config.IV_CSV, index=False)
    log.info("Top predictors by IV: %s", ", ".join(iv["feature"].head(5)))

    pipe, report, test_idx = model.train(df)
    champ = report.models[report.champion]
    log.info("Champion %s: test AUC %.3f | KS %.3f | CV AUC %.3f +/- %.3f | threshold %.2f",
             report.champion, champ["roc_auc"], champ["ks"], champ["cv_auc_mean"],
             champ["cv_auc_std"], champ["threshold"])
    model.save(pipe, report)

    num, cat = features.feature_columns()
    scored = df.copy()
    scored["p_default"] = pipe.predict_proba(df[num + cat])[:, 1].round(4)
    scored["score"] = scoring.pd_to_score(scored["p_default"])
    scored["band"] = scored["score"].map(scoring.score_to_band)
    scored["decision"] = scored["band"].map(scoring.BAND_ACTION)
    scored["split"] = "train"
    scored.loc[test_idx, "split"] = "test"
    scored.to_csv(config.SCORED_CSV, index=False)

    scoring.band_table(scored).to_csv(config.BANDS_CSV, index=False)
    indicators = scoring.indicator_report(scored)
    indicators.to_csv(config.INDICATORS_CSV, index=False)
    disagreements = indicators.loc[~indicators["model_agrees"], "indicator"].tolist()
    if disagreements:
        log.warning("Model disagrees with risk indicators: %s", disagreements)
    else:
        log.info("Model direction agrees with all %d risk indicators", len(indicators))
    return {"rows": len(scored), "report": report.to_dict(), "disagreements": disagreements}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Fail instead of falling back to synthetic data when offline")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    result = run(args.force_download, not args.no_synthetic)
    summary = {k: {m: v for m, v in r.items() if m in
                   ("roc_auc", "gini", "ks", "brier", "cv_auc_mean", "threshold", "expected_cost")}
               for k, r in result["report"]["models"].items()}
    print(json.dumps({"champion": result["report"]["champion"], "models": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
