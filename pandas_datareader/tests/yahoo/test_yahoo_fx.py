from datetime import datetime

import pandas as pd
from pandas import DataFrame, MultiIndex
import pytest

from pandas_datareader._testing import skip_on_exception
from pandas_datareader._utils import RemoteDataError
from pandas_datareader.yahoo.fx import YahooFXReader

START = datetime(2023, 1, 2)
END = datetime(2023, 1, 31)

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
        with pytest.raises(Exception):  # noqa: B017
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
