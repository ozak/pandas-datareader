import pandas as pd

from pandas_datareader._utils import RemoteDataError
from pandas_datareader.base import _BaseReader


class BankOfCanadaReader(_BaseReader):
    """Get data for the given name from Bank of Canada.

    Notes
    -----
    See `Bank of Canada <https://www.bankofcanada.ca/rates/>`__

    Uses the Valet JSON API: https://www.bankofcanada.ca/valet/
    """

    _URL = "https://www.bankofcanada.ca/valet/observations"
    _format = "json"

    @property
    def url(self):
        """API URL"""
        if not isinstance(self.symbols, str):
            raise ValueError("data name must be string")

        return f"{self._URL}/{self.symbols}/json"

    @property
    def params(self):
        """Parameters to use in API calls"""
        return {
            "start_date": self.start.strftime("%Y-%m-%d"),
            "end_date": self.end.strftime("%Y-%m-%d"),
        }

    def _read_lines(self, out):
        """Parse the Valet JSON response into a DataFrame."""
        observations = out.get("observations", [])
        if not observations:
            raise RemoteDataError(
                f"No data returned for series {self.symbols!r}. "
                "Check that the series name is valid."
            )

        records = []
        for obs in observations:
            date = obs["d"]
            # Each observation dict has the series name as a key whose value
            # is a dict {"v": "<value>"}.  Missing values use an empty string.
            series_data = obs.get(self.symbols, {})
            raw_value = series_data.get("v", "")
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                value = float("nan")
            records.append({"DATE": date, self.symbols: value})

        df = pd.DataFrame(records).set_index("DATE")
        df.index = pd.to_datetime(df.index)
        df.index.name = "DATE"
        df = df.sort_index()
        return df
