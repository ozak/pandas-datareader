import json
import warnings
from urllib.parse import urlencode

import pandas as pd

from pandas_datareader.base import _BaseReader

# The IEX v1 free API (api.iextrading.com/1.0/) was shut down in June 2019.
# These readers now point at IEX Cloud (cloud.iexapis.com/stable/) and require
# an API key.  Register for a free tier key at https://iexcloud.io/.
#
# Data provided for free by IEX
# Data is furnished in compliance with the guidelines promulgated in the IEX
# API terms of service and manual
# See https://iextrading.com/api-exhibit-a/ for additional information
# and conditions of use

_IEX_V1_SHUTDOWN_MSG = (
    "The IEX v1 free API (api.iextrading.com/1.0/) was shut down. "
    "These readers now use the IEX Cloud API (cloud.iexapis.com/stable/), "
    "which requires a free API key from https://iexcloud.io/. "
    "Pass the key via the api_key parameter or set the IEX_API_KEY "
    "environment variable."
)


class IEX(_BaseReader):
    """
    Serves as the base class for all IEX API services.

    .. deprecated::
        The IEX v1 free API was shut down. Use the IEX Cloud API with a
        free key from https://iexcloud.io/.
    """

    _format = "json"

    def __init__(
        self,
        symbols=None,
        start=None,
        end=None,
        retry_count=3,
        pause=0.1,
        session=None,
        api_key=None,
    ):
        import os

        if api_key is None:
            api_key = os.getenv("IEX_API_KEY")
        self.api_key = api_key

        # Support for sandbox environment (testing purposes)
        if os.getenv("IEX_SANDBOX") == "enable":
            self.sandbox = True
        else:
            self.sandbox = False

        super().__init__(
            symbols=symbols,
            start=start,
            end=end,
            retry_count=retry_count,
            pause=pause,
            session=session,
        )

    @property
    def service(self):
        """Service endpoint"""
        # This property will be overridden by the subclass
        raise NotImplementedError("IEX API service not specified.")

    @property
    def _base_url(self):
        if self.sandbox:
            return "https://sandbox.iexapis.com/stable"
        return "https://cloud.iexapis.com/stable"

    @property
    def url(self):
        """API URL"""
        params = self._get_params(self.symbols)
        if self.api_key:
            params["token"] = self.api_key
        qstring = urlencode(params)
        return f"{self._base_url}/{self.service}?{qstring}"

    def read(self):
        """Read data"""
        if not self.api_key:
            raise ValueError(_IEX_V1_SHUTDOWN_MSG)
        df = super().read()
        if isinstance(df, pd.DataFrame):
            df = df.squeeze()
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        return df

    def _get_params(self, symbols):
        p = {}
        if isinstance(symbols, list):
            p["symbols"] = ",".join(symbols)
        elif isinstance(symbols, str):
            p["symbols"] = symbols
        return p

    def _output_error(self, out):
        """If IEX returns a non-200 status code, we need to notify the user of
        the error returned.

        Parameters
        ----------
        out: bytes
            The raw output from an HTTP request
        """
        try:
            content = json.loads(out.text)
        except Exception as exc:
            raise TypeError("Failed to interpret response as JSON.") from exc

        for key, string in content.items():
            e = f"IEX Output error encountered: {string}"
            if key == "error":
                raise Exception(e)

    def _read_lines(self, out):
        """IEX's output does not need anything complex, so we're overriding to
        use Pandas' default interpreter

        Parameters
        ----------
        out: bytes
            The raw output from an HTTP request

        Returns
        -------
        DataFrame
        """

        # IEX will return a blank line for invalid tickers:
        if isinstance(out, list):
            out = [x for x in out if x is not None]
        return pd.DataFrame(out) if len(out) > 0 else pd.DataFrame()
