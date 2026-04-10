import pandas as pd

from pandas_datareader.base import _BaseReader
from pandas_datareader.io import read_jsdmx


class OECDReader(_BaseReader):
    """Get data for the given name from OECD."""

    _format = "json"

    _URL = "https://sdmx.oecd.org/public/rest/data"

    @property
    def url(self):
        """API URL"""
        if not isinstance(self.symbols, str):
            raise ValueError("data name must be string")

        # OECD SDMX 2.1 REST API:
        # https://sdmx.oecd.org/public/rest/data/{agencyID},{dataflowID},{version}/{key}
        return f"{self._URL}/OECD,{self.symbols},+/all"

    @property
    def params(self):
        """Parameters to use in API calls"""
        return {
            "format": "jsondata",
            "startPeriod": str(self.start.year),
            "endPeriod": str(self.end.year),
        }

    def _read_lines(self, out):
        """read one data from specified URL"""
        df = read_jsdmx(out)
        try:
            idx_name = df.index.name  # hack for pandas 0.16.2
            df.index = pd.to_datetime(df.index, errors="ignore")
            for col in df:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            df = df.sort_index()
            df = df.truncate(self.start, self.end)
            df.index.name = idx_name
        except ValueError:
            pass
        return df
