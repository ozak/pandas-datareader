import pandas as pd

from pandas_datareader.base import _BaseReader
from pandas_datareader.io import read_jsdmx
from pandas_datareader.io.jsdmx import read_jsdmx2


class OECDReader(_BaseReader):
    """Get data for the given name from OECD.

    Notes
    -----
    Uses the new OECD Data API (SDMX REST 2.1):
    https://sdmx.oecd.org/public/rest/

    The ``symbols`` parameter should be a dataset reference in the format::

        {agencyID},{dataflowID},{version}/{key}/

    For example: ``OECD.SDD.TPS,DSD_TUD@DF_TUD,1.0/all``

    See https://data-explorer.oecd.org for dataset identifiers.
    """

    _format = "json"

    @property
    def url(self):
        """API URL"""
        url = "https://sdmx.oecd.org/public/rest/data"

        if not isinstance(self.symbols, str):
            raise ValueError("data name must be string")

        return f"{url}/{self.symbols}"

    @property
    def params(self):
        """Parameters to use in API calls"""
        return {"format": "jsondata"}

    def _read_lines(self, out):
        """read one data from specified URL"""
        # Try SDMX-JSON 2.0 format first (new OECD API), fall back to 1.0
        try:
            df = read_jsdmx2(out)
        except (KeyError, TypeError, ValueError):
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
