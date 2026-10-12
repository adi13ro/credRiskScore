import pytest
from streamlit.testing.v1 import AppTest

from creditrisk import config

pytestmark = pytest.mark.skipif(
    not config.SCORED_CSV.exists() or not config.MODEL_PATH.exists(),
    reason="run `creditrisk-pipeline` first to produce dashboard artefacts",
)


def test_dashboard_renders():
    at = AppTest.from_file(str(config.ROOT / "app" / "dashboard.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Credit Risk Scoring"
    assert len(at.metric) >= 6
    assert len(at.tabs) == 5
