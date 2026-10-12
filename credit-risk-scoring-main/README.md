# Credit Risk Scoring Model

Predicts the probability that a consumer loan applicant defaults, converts it to a
scorecard-style points score with A–E risk bands, checks the model against classic
underwriting risk indicators, and presents everything in an interactive Streamlit dashboard
for business stakeholders.

```
Statlog German Credit ──▶ validate ──▶ engineer ──▶ WoE / IV ──▶ champion vs challenger
 (1,000 applications)                                             (logistic vs boosting)
                                                                          │
         Streamlit dashboard ◀── bands, risk indicators, scored portfolio ◀┘
```

## Results (hold-out test split, 250 applications, 30% base default rate)

| Model | ROC-AUC | Gini | KS | Brier | 5-fold CV AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.800 | 0.601 | 0.518 | 0.157 | 0.761 ± 0.021 |
| Gradient boosting | 0.784 | 0.568 | 0.467 | 0.164 | 0.779 ± 0.026 |

Champion: **gradient boosting**, chosen on cross-validated AUC rather than the test
split to avoid selection leakage. Published benchmarks on this dataset sit around 0.75–0.80 AUC.

The decision threshold (0.11) minimises the dataset's asymmetric
cost matrix: approving a loan that defaults costs five times as much as declining one that would
have been repaid. Strongest predictors by information value: checking status, credit history, duration months, savings status, purpose.

The model's direction agrees with all ten rule-based risk indicators (overdrawn checking
account, no savings, past payment delays, high installment rate, long loan, large loan,
unemployed, under 25, other installment plans, no property): applicants carrying each flag have
both a higher observed default rate and a higher mean predicted PD than the portfolio.

## Stack

Python 3.12, pandas, scikit-learn, Streamlit, Plotly, pytest, ruff, uv. Docker optional.

## Run it

```bash
make install     # uv sync --extra dev
make pipeline    # download -> validate -> IV -> train -> score -> reports (about 15 s)
make dashboard   # http://localhost:8501
make test        # 8 tests incl. a headless render of the dashboard
```

Docker (trains at build time):

```bash
docker build -t credit-risk . && docker run -p 8501:8501 credit-risk
```

Offline, the pipeline falls back to a schema-compatible synthetic dataset so everything still
runs; pass `--no-synthetic` to fail instead.

## Dashboard

| Tab | What it shows |
|---|---|
| Risk distribution | Score histogram by outcome, applications and default rate per band, band table with recommended action, PD by loan purpose |
| Model performance | Champion vs challenger table, ROC, KS and calibration curves, confusion matrix at the cost-optimal threshold |
| Drivers | Information value per feature, weight-of-evidence per category, logistic coefficients |
| Risk indicators | Observed vs predicted default lift for each underwriting red flag, agreement check |
| Score an applicant | Form for a new application: PD, score, band, action, risk flags raised |

Sidebar filters: all rows vs hold-out only, loan purpose, credit amount.

## Scorecard

PD is mapped to points with the standard log-odds scale: 600 points at 2:1 good/bad odds and
20 points to double the odds (`src/creditrisk/scoring.py`). Bands are looked up with `bisect`
on fixed edges:

| Band | Score | Action |
|---|---|---|
| A | ≥ 700 | auto-approve |
| B | 650–699 | approve |
| C | 600–649 | manual review |
| D | 550–599 | decline (appeal to senior underwriter) |
| E | < 550 | decline |

## Fairness note

`personal_status_sex` and `foreign_worker` are present in the source data but excluded from the
model features. They stay in the scored output so the portfolio can be monitored for disparate
outcomes.

## Layout

```
src/creditrisk/
  data.py        UCI download, code decoding, synthetic fallback, schema validation
  features.py    engineered ratios, preprocessing, WoE / IV
  model.py       champion/challenger training, KS, cost-optimal threshold, calibration
  scoring.py     PD -> score -> band, risk indicator rules and agreement report
  pipeline.py    CLI entry point
app/dashboard.py Streamlit app
tests/           unit tests + dashboard smoke test
```

## Dataset

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.
1,000 applications, 20 attributes, 700 good / 300 bad.
