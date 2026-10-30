import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, MINIMUM_HISTORY, build_features, latest_features


def history(rows=MINIMUM_HISTORY):
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.linspace(1500, 1600, rows)
    return pd.DataFrame({
        "Date": dates, "Open": close - 1, "High": close + 2,
        "Low": close - 3, "Close": close,
    })


def test_feature_order_and_complete_latest_row():
    result = latest_features(history())
    assert list(result.columns) == FEATURE_COLUMNS
    assert result.shape == (1, len(FEATURE_COLUMNS))


def test_features_match_core_formula():
    data = history()
    result = build_features(data)
    expected = np.log(data["Close"].iloc[-1] / data["Close"].iloc[-2]) * 100
    assert result["current_return_pct"].iloc[-1] == pytest.approx(expected)


@pytest.mark.parametrize("bad_data", [
    history(MINIMUM_HISTORY - 1),
    history().assign(Close=lambda frame: frame["Close"].where(frame.index != 10)),
])
def test_invalid_or_insufficient_history(bad_data):
    with pytest.raises(ValueError):
        latest_features(bad_data)


def test_duplicate_dates_are_rejected():
    data = history()
    data.loc[1, "Date"] = data.loc[0, "Date"]
    with pytest.raises(ValueError, match="duplicate"):
        build_features(data)


def test_invalid_dates_are_rejected():
    data = history()
    data["Date"] = data["Date"].astype(object)
    data.loc[0, "Date"] = "not-a-date"
    with pytest.raises(ValueError, match="invalid"):
        build_features(data)
