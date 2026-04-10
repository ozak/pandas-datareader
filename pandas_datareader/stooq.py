from pandas_datareader._utils import RemoteDataError
from pandas_datareader.base import _DailyBaseReader

_STOOQ_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
)


class StooqDailyReader(_DailyBaseReader):
    """
    Returns DataFrame/dict of Dataframes of historical stock prices from
    symbols, over date range, start to end.

    Parameters
    ----------
    symbols : string, array-like object (list, tuple, Series), or DataFrame
        Single stock symbol (ticker), array-like object of symbols or
        DataFrame with index containing stock symbols.
    start : string, int, date, datetime, Timestamp
        Starting date. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980'). Defaults to
        20 years before current date.
    end : string, int, date, datetime, Timestamp
        Ending date
    retry_count : int, default 3
        Number of times to retry query request.
    pause : int, default 0.1
        Time, in seconds, to pause between consecutive queries of chunks. If
        single value given for symbol, represents the pause between retries.
    chunksize : int, default 25
        Number of symbols to download consecutively before initiating pause.
    session : Session, default None
        requests.sessions.Session instance to be used

    Notes
    -----
    See `Stooq <https://stooq.com>`__
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers = {"User-Agent": _STOOQ_USER_AGENT}

    @staticmethod
    def _sanitize_response(response):
        content = response.content
        # Stooq may return an HTML page (CAPTCHA / rate-limit) instead of CSV.
        # Detect it early so users get a clear RemoteDataError instead of a
        # confusing pandas.errors.ParserError.
        stripped = content.lstrip().lstrip(b"\xef\xbb\xbf")  # strip BOM
        if stripped[:9].startswith(b"<!"):
            raise RemoteDataError(
                "Stooq returned an HTML page instead of CSV data. "
                "The service may be rate-limiting or blocking automated requests."
            )
        return content

    @property
    def url(self):
        """API URL"""
        return "https://stooq.com/q/d/l/"

    def _get_params(self, symbol, country="US"):
        symbol_parts = symbol.split(".")
        if not symbol.startswith("^"):
            if len(symbol_parts) == 1:
                symbol = ".".join([symbol, country])
            elif symbol_parts[1].lower() == "pl":
                symbol = symbol_parts[0]
            else:
                if symbol_parts[1].lower() not in [
                    "de",
                    "hk",
                    "hu",
                    "jp",
                    "uk",
                    "us",
                    "f",
                    "b",
                ]:
                    symbol = ".".join([symbol, "US"])

        params = {
            "s": symbol,
            "i": self.freq or "d",
            "d1": self.start.strftime("%Y%m%d"),
            "d2": self.end.strftime("%Y%m%d"),
        }

        return params
