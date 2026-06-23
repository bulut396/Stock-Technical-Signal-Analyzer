"""Price-history fetching via yfinance (free, no API key required)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse yfinance's flat or MultiIndex columns into a single level.

    yfinance may return either flat columns (``"Close"``) or MultiIndex
    columns (``("Close", ticker)``) depending on version and arguments. This
    normalizes both shapes to a single-level OHLCV frame.

    Parameters
    ----------
    data : pandas.DataFrame
        The frame returned by :func:`yfinance.download`.
    ticker : str
        The requested ticker, used only to select the right MultiIndex slice
        if present.

    Returns
    -------
    pandas.DataFrame
        A frame with single-level columns ``Open``, ``High``, ``Low``,
        ``Close``, ``Volume``.
    """
    if isinstance(data.columns, pd.MultiIndex):
        # Columns look like ("Close", "AAPL"); select this ticker's slice.
        try:
            data = data.xs(ticker.upper(), axis=1, level=1)
        except KeyError:
            # Some yfinance versions order the levels the other way around.
            data = data.xs(ticker.upper(), axis=1, level=0)
    return data


def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV (open/high/low/close/volume) price history for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. ``"AAPL"``.
    period : str, optional
        A yfinance-compatible period string (``"3mo"``, ``"6mo"``, ``"1y"``,
        ``"2y"``, ...). Defaults to ``"6mo"``.
    interval : str, optional
        A yfinance-compatible bar interval (e.g. ``"1d"``). Defaults to
        ``"1d"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``, indexed by
        date, oldest to newest, with adjusted prices (``auto_adjust=True``).

    Raises
    ------
    ValueError
        If ``ticker`` is empty, if no price data is returned for the symbol,
        or if the request to yfinance fails (network or library error).
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        raise ValueError("Please enter a ticker symbol.")

    try:
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise ValueError(
            "Could not fetch price data. Check your internet connection and "
            "try again."
        ) from exc

    if data is None or data.empty:
        raise ValueError(
            f"No price data found for ticker '{symbol}'. Check the symbol "
            "and try again."
        )

    data = _flatten_columns(data, symbol)

    missing = [col for col in _OHLCV_COLUMNS if col not in data.columns]
    if missing or data.empty:
        raise ValueError(
            f"No price data found for ticker '{symbol}'. Check the symbol "
            "and try again."
        )

    result = data[_OHLCV_COLUMNS].dropna()
    result.columns.name = None  # drop the leftover "Price" MultiIndex level name
    return result
