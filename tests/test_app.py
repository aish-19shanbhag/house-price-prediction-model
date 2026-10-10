"""Headless test of the Streamlit demo with streamlit.testing.v1.AppTest."""

from pathlib import Path

import pytest

from src.config import MODEL_FILE, ZIP_PROFILE_FILE

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
pytestmark = pytest.mark.skipif(not (MODEL_FILE.exists() and ZIP_PROFILE_FILE.exists()),
                                reason="run `python -m src.train` first")


def _price(at):
    return float(at.metric[0].value.replace("$", "").replace(",", ""))


def test_app_renders_an_estimate():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].label == "Estimated sale price"
    assert 100_000 < _price(at) < 3_000_000


def test_bigger_better_house_costs_more():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    base = _price(at)
    at.number_input[0].set_value(3200)          # living area
    at.slider[2].set_value(10)                  # construction grade
    at.checkbox[0].check()                      # waterfront
    at.run()
    assert not at.exception
    assert _price(at) > base * 1.5
