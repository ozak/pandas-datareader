import os

import pandas as pd

from pandas_datareader._utils import RemoteDataError
from pandas_datareader.base import _BaseReader


class EcondbReader(_BaseReader):
    """
    Returns DataFrame of historical stock prices from symbol, over date
    range, start to end.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    symbols : string
        Can be in two different formats:
        1. 'ticker=<code>' for fetching a single series,
        where <code> is CPIUS for, e.g. the series at
        https://www.econdb.com/series/CPIUS/
        2. 'dataset=<dataset>&<params>' for fetching full
        or filtered subset of a dataset, like the one at
        https://www.econdb.com/dataset/ABS_GDP. After choosing the desired filters,
        the correctly formatted query string can be easily generated
        from that dataset's page by using the Export function, and choosing Pandas Python3.
    start : string, int, date, datetime, Timestamp
        Starting date. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
    end : string, int, date, datetime, Timestamp
        Ending date
    retry_count : int, default 3
        Number of times to retry query request.
    pause : int, default 0.1
        Time, in seconds, to pause between consecutive queries of chunks. If
        single value given for symbol, represents the pause between retries.
    session : Session, default None
        requests.sessions.Session instance to be used
    api_key : str, optional
        Econdb API token.  If not provided, the environment variable
        ``ECONDB_API_KEY`` is used instead.  A free token can be
        obtained at https://www.econdb.com/.
    """

    _URL = "https://www.econdb.com/api/series/"
    _format = None
    _show = "labels"

    def __init__(
        self,
        symbols,
        start=None,
        end=None,
        retry_count=3,
        pause=0.1,
        session=None,
        freq=None,
        api_key=None,
    ):
        super().__init__(
            symbols=symbols,
            start=start,
            end=end,
            retry_count=retry_count,
            pause=pause,
            session=session,
            freq=freq,
        )
        if api_key is None:
            api_key = os.getenv("ECONDB_API_KEY")
        self.api_key = api_key

        params = dict(s.split("=") for s in self.symbols.split("&"))
        if "from" in params and not start:
            self.start = pd.to_datetime(params["from"], format="%Y-%m-%d")
        if "to" in params and not end:
            self.end = pd.to_datetime(params["to"], format="%Y-%m-%d")

    @property
    def url(self):
        """API URL"""
        if not isinstance(self.symbols, str):
            raise ValueError("data name must be string")

        url = "{}?{}&format=json&page_size=500&expand=both".format(
            self._URL, self.symbols
        )
        if self.api_key:
            url += f"&token={self.api_key}"
        return url

    def read(self):
        """read one data from specified URL"""
        resp = self.session.get(self.url)
        if resp.status_code != 200:
            raise RemoteDataError(
                f"Econdb API request failed (HTTP {resp.status_code}). "
                "Check that the series identifier is valid and that an API key "
                "is provided if required (pass api_key= or set ECONDB_API_KEY)."
            )

        payload = resp.json()

        # The "results" key holds the list of series entries.
        results = payload.get("results")
        if results is None:
            raise RemoteDataError(
                "Econdb API response did not contain 'results'. "
                f"Response keys: {list(payload.keys())}"
            )

        df = pd.DataFrame({"dates": []}).set_index("dates")

        if self._show == "labels":

            def show_func(x):
                return x[x.find(":") + 1 :]

        elif self._show == "codes":

            def show_func(x):
                return x[: x.find(":")]

        unique_keys = {k for s in results for k in s.get("additional_metadata", {})}
        for entry in results:
            # The "data" field may be a list of dicts or a dict depending on
            # the API version; normalise to a list of {"dates", "values"} dicts.
            raw_data = entry.get("data", [])
            if not raw_data:
                continue
            series = pd.DataFrame(raw_data)[["dates", "values"]].set_index("dates")
            head = entry.get("additional_metadata", {})
            for k in unique_keys:
                if k not in head:
                    head[k] = "-1:None"
            if head:
                series.columns = pd.MultiIndex.from_tuples(
                    [[show_func(x) for x in head.values()]],
                    names=[show_func(x) for x in head.keys()],
                )
            else:
                series.rename(columns={"values": entry.get("ticker", "values")}, inplace=True)

            if not df.empty:
                df = df.merge(series, how="outer", left_index=True, right_index=True)
            else:
                df = series

        if df.shape[0] > 0:
            df.index = pd.to_datetime(df.index, errors="ignore")
        df.index.name = "TIME_PERIOD"
        df = df.truncate(self.start, self.end)
        return df
