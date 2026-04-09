from pandas import concat, read_csv

from pandas_datareader._utils import RemoteDataError
from pandas_datareader.base import _BaseReader
from pandas_datareader.compat import is_list_like


class FredReader(_BaseReader):
    """
    Get data for the given name from the St. Louis FED (FRED).
    """

    @property
    def url(self):
        """API URL"""
        return "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def read(self):
        """Read data

        Returns
        -------
        data : DataFrame
            If multiple names are passed for "series" then the index of the
            DataFrame is the outer join of the indices of each series.
        """
        try:
            return self._read()
        finally:
            self.close()

    def _read(self):
        if not is_list_like(self.symbols):
            names = [self.symbols]
        else:
            names = self.symbols

        urls = [f"{self.url}?id={n}" for n in names]

        def fetch_data(url, name):
            """Utility to fetch data"""
            resp = self._read_url_as_StringIO(url)
            data = read_csv(
                resp,
                index_col=0,
                parse_dates=True,
                header=None,
                skiprows=1,
                names=["DATE", name],
                na_values=".",
            )
            # FRED returns an error page (not a proper CSV) for invalid series.
            # Detect this by checking whether the first index value contains
            # "Error" — the error page has a line like "Error: ..." as its header.
            if len(data) > 0 and "Error" in str(data.index[0]):
                raise RemoteDataError(
                    "Failed to get the data. Check that "
                    "{!r} is a valid FRED series.".format(name)
                )
            return data.truncate(self.start, self.end)

        data = [fetch_data(url, n) for url, n in zip(urls, names, strict=True)]
        df = concat(
            data,
            axis=1,
            join="outer",
        )
        return df
