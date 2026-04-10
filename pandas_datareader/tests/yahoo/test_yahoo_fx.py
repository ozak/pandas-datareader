import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DataFrame, MultiIndex
import pytest

from pandas_datareader._testing import skip_on_exception
from pandas_datareader._utils import RemoteDataError
from pandas_datareader.yahoo.fx import YahooFXReader

START = datetime(2023, 1, 2)
END = datetime(2023, 1, 31)

# ---------------------------------------------------------------------------
# Helpers for mock-based tests
# ---------------------------------------------------------------------------

# Three representative Unix timestamps (Jan 2-4 2023, 16:00 UTC)
_TS = [1672675200, 1672761600, 1672848000]

_QUOTE = {
    "open": [1.0693, 1.0602, 1.0537],
    "high": [1.0720, 1.0650, 1.0570],
    "low": [1.0650, 1.0570, 1.0490],
    "close": [1.0710, 1.0638, 1.0551],
    "volume": [0, 0, 0],
}

_YAHOO_JSON = {
    "chart": {
        "result": [
            {
                "timestamp": _TS,
                "indicators": {"quote": [_QUOTE]},
            }
        ]
    }
}


def _make_mock_response(payload=None):
    """Return a mock requests.Response with .text set to *payload* JSON."""
    if payload is None:
        payload = _YAHOO_JSON
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp

_XFAIL_REASON = (
    "Yahoo Finance may block requests without valid cookies/crumb; "
    "tests require live network access."
)


@pytest.mark.xfail(reason=_XFAIL_REASON)
class TestYahooFX:
    @skip_on_exception(RemoteDataError)
    def test_single_symbol_returns_dataframe(self):
        df = YahooFXReader("EURUSD", start=START, end=END).read()
        assert isinstance(df, DataFrame)

    @skip_on_exception(RemoteDataError)
    def test_single_symbol_has_expected_columns(self):
        df = YahooFXReader("EURUSD", start=START, end=END).read()
        for col in ("Open", "High", "Low", "Close"):
            assert col in df.columns

    @skip_on_exception(RemoteDataError)
    def test_volume_column_is_dropped(self):
        df = YahooFXReader("EURUSD", start=START, end=END).read()
        assert "Volume" not in df.columns

    @skip_on_exception(RemoteDataError)
    def test_single_symbol_index_is_date(self):
        df = YahooFXReader("EURUSD", start=START, end=END).read()
        assert df.index.name == "Date"

    @skip_on_exception(RemoteDataError)
    def test_single_symbol_nonempty(self):
        df = YahooFXReader("EURUSD", start=START, end=END).read()
        assert len(df) > 0

    @skip_on_exception(RemoteDataError)
    def test_single_symbol_values_are_numeric(self):
        df = YahooFXReader("EURUSD", start=START, end=END).read()
        assert pd.api.types.is_float_dtype(df["Close"])

    @skip_on_exception(RemoteDataError)
    def test_multiple_symbols_returns_dataframe(self):
        df = YahooFXReader(["EURUSD", "GBPUSD"], start=START, end=END).read()
        assert isinstance(df, DataFrame)

    @skip_on_exception(RemoteDataError)
    def test_multiple_symbols_has_paircode_date_multiindex(self):
        df = YahooFXReader(["EURUSD", "GBPUSD"], start=START, end=END).read()
        assert isinstance(df.index, MultiIndex)
        assert df.index.names == ["PairCode", "Date"]

    @skip_on_exception(RemoteDataError)
    def test_multiple_symbols_contains_all_pairs(self):
        symbols = ["EURUSD", "GBPUSD"]
        df = YahooFXReader(symbols, start=START, end=END).read()
        assert set(df.index.get_level_values("PairCode")) == set(symbols)

    @skip_on_exception(RemoteDataError)
    def test_multiple_symbols_volume_is_dropped(self):
        df = YahooFXReader(["EURUSD", "GBPUSD"], start=START, end=END).read()
        assert "Volume" not in df.columns

    @skip_on_exception(RemoteDataError)
    def test_dataframe_input(self):
        symbols_df = DataFrame(index=["EURUSD", "GBPUSD"])
        df = YahooFXReader(symbols_df, start=START, end=END).read()
        assert isinstance(df, DataFrame)
        assert isinstance(df.index, MultiIndex)

    def test_invalid_symbol_raises(self):
        with pytest.raises(RemoteDataError):
            YahooFXReader("INVALID_PAIR_XYZ", start=START, end=END).read()

    def test_all_symbols_invalid_raises_remote_data_error(self):
        with pytest.raises(RemoteDataError):
            YahooFXReader(
                ["INVALID_PAIR_XYZ", "ANOTHER_BAD_PAIR"], start=START, end=END
            ).read()


class TestYahooFXInit:
    """Tests for constructor validation that do not require network access."""

    def test_invalid_interval_raises_value_error(self):
        with pytest.raises(ValueError):
            YahooFXReader("EURUSD", start=START, end=END, interval="NOT_VALID")

    def test_daily_interval_accepted(self):
        r = YahooFXReader("EURUSD", start=START, end=END, interval="d")
        assert r.interval == "1d"

    def test_weekly_interval_accepted(self):
        r = YahooFXReader("EURUSD", start=START, end=END, interval="wk")
        assert r.interval == "1wk"

    def test_monthly_interval_accepted(self):
        r = YahooFXReader("EURUSD", start=START, end=END, interval="mo")
        assert r.interval == "1mo"


# ---------------------------------------------------------------------------
# Mock-based tests — no network required, exercise every code branch
# ---------------------------------------------------------------------------


class TestYahooFXMocked:
    """Full-coverage tests using mocked HTTP responses."""

    # ------------------------------------------------------------------
    # Single-symbol path
    # ------------------------------------------------------------------

    def test_single_symbol_returns_dataframe(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert isinstance(df, DataFrame)

    def test_single_symbol_ohlc_columns_present(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        for col in ("Open", "High", "Low", "Close"):
            assert col in df.columns, f"Missing column: {col}"

    def test_single_symbol_volume_dropped(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert "Volume" not in df.columns

    def test_single_symbol_index_name_is_date(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert df.index.name == "Date"

    def test_single_symbol_nonempty(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert len(df) == len(_TS)

    def test_single_symbol_close_values_correct(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert list(df["Close"]) == _QUOTE["close"]

    def test_single_symbol_close_is_float(self):
        reader = YahooFXReader("EURUSD", start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert pd.api.types.is_float_dtype(df["Close"])

    def test_get_params_appends_equals_x(self):
        """_get_params must append '=X' to the symbol for the FX API."""
        reader = YahooFXReader("EURUSD", start=START, end=END)
        params = reader._get_params("EURUSD")
        assert params["symbol"] == "EURUSD=X"

    def test_read_one_data_url_contains_symbol_equals_x(self):
        """_read_one_data must call the chart URL with symbol=X appended."""
        reader = YahooFXReader("EURUSD", start=START, end=END)
        captured_urls = []

        def fake_get_response(url, params=None):
            captured_urls.append(url)
            return _make_mock_response()

        with patch.object(reader, "_get_response", side_effect=fake_get_response):
            reader._read_one_data("EURUSD")

        assert captured_urls[0].endswith("EURUSD=X")

    # ------------------------------------------------------------------
    # Multiple-symbol path
    # ------------------------------------------------------------------

    def test_multiple_symbols_returns_dataframe(self):
        reader = YahooFXReader(["EURUSD", "GBPUSD"], start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert isinstance(df, DataFrame)

    def test_multiple_symbols_has_paircode_date_multiindex(self):
        reader = YahooFXReader(["EURUSD", "GBPUSD"], start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert isinstance(df.index, MultiIndex)
        assert df.index.names == ["PairCode", "Date"]

    def test_multiple_symbols_contains_all_pairs(self):
        symbols = ["EURUSD", "GBPUSD"]
        reader = YahooFXReader(symbols, start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert set(df.index.get_level_values("PairCode")) == set(symbols)

    def test_multiple_symbols_volume_dropped(self):
        reader = YahooFXReader(["EURUSD", "GBPUSD"], start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert "Volume" not in df.columns

    # ------------------------------------------------------------------
    # DataFrame input path
    # ------------------------------------------------------------------

    def test_dataframe_input_returns_multiindex(self):
        symbols_df = DataFrame(index=["EURUSD", "GBPUSD"])
        reader = YahooFXReader(symbols_df, start=START, end=END)
        with patch.object(reader, "_get_response", return_value=_make_mock_response()):
            df = reader.read()
        assert isinstance(df.index, MultiIndex)
        assert df.index.names == ["PairCode", "Date"]

    # ------------------------------------------------------------------
    # Partial-failure path in _dl_mult_symbols
    # ------------------------------------------------------------------

    def test_partial_failure_warns_and_returns_good_pairs(self):
        """When one symbol fails with OSError, a SymbolWarning is issued and
        the remaining symbol's data is returned."""
        symbols = ["EURUSD", "BAD"]
        reader = YahooFXReader(symbols, start=START, end=END)

        call_count = {"n": 0}

        def fake_get_response(url, params=None):
            call_count["n"] += 1
            if "BAD" in url:
                raise OSError("simulated network failure")
            return _make_mock_response()

        from pandas_datareader._utils import SymbolWarning

        with patch.object(reader, "_get_response", side_effect=fake_get_response):
            with pytest.warns(SymbolWarning):
                df = reader.read()

        assert isinstance(df.index, MultiIndex)
        pairs = set(df.index.get_level_values("PairCode"))
        assert pairs == {"EURUSD"}
        assert "BAD" not in pairs

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_single_invalid_symbol_raises_remote_data_error(self):
        reader = YahooFXReader("INVALID", start=START, end=END)
        # Simulate what the base reader does when it cannot connect
        with patch.object(
            reader, "_get_response", side_effect=RemoteDataError("Unable to read URL")
        ):
            with pytest.raises(RemoteDataError):
                reader.read()

    def test_all_symbols_invalid_raises_remote_data_error(self):
        reader = YahooFXReader(["BAD1", "BAD2"], start=START, end=END)
        with patch.object(
            reader, "_get_response", side_effect=OSError("simulated failure")
        ):
            with pytest.raises(RemoteDataError):
                reader.read()
