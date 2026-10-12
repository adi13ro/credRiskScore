from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

UCI_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
RAW_CSV = DATA_RAW / "german_credit_raw.csv"
SCORED_CSV = DATA_PROCESSED / "applications_scored.csv"
IV_CSV = DATA_PROCESSED / "information_value.csv"
INDICATORS_CSV = DATA_PROCESSED / "risk_indicators.csv"
BANDS_CSV = DATA_PROCESSED / "score_bands.csv"
METRICS_JSON = MODELS / "metrics.json"
MODEL_PATH = MODELS / "credit_model.joblib"

RANDOM_SEED = 42
TEST_SIZE = 0.25

# Scorecard scaling: 600 points at 2:1 good/bad odds (roughly the portfolio average for
# this 30 %-default dataset), 20 points to double the odds. A prime portfolio would use
# a higher base odds such as 50:1.
SCORE_BASE = 600
SCORE_BASE_ODDS = 2
SCORE_PDO = 20

# Statlog cost matrix: a bad loan approved costs 5x a good loan declined.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
