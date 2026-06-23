"""Technical indicator calculations, implemented directly with pandas/numpy.

No pre-built technical-analysis library is used here on purpose: every formula
below is implemented from its mathematical definition so the math is fully
transparent and auditable.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def _require_min_length(series: pd.Series, min_length: int, label: str) -> None:
    """Raise ``ValueError`` if ``series`` is shorter than ``min_length``."""
    if len(series) < min_length:
        raise ValueError(
            f"At least {min_length} periods of data are required to compute "
            f"{label}, but only {len(series)} were provided."
        )


def sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average.

    The unweighted mean of the last ``window`` closing prices at each point:
    ``SMA[t] = mean(prices[t-window+1 : t+1])``. It smooths out day-to-day
    noise to show the underlying price trend; a rising SMA suggests an
    uptrend, a falling one a downtrend.

    Parameters
    ----------
    prices : pandas.Series
        A series of prices (typically ``Close``), indexed by date.
    window : int, optional
        Number of periods to average. Defaults to ``20``.

    Returns
    -------
    pandas.Series
        The moving average, with ``NaN`` for the first ``window - 1`` points
        where there isn't enough history yet.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``window`` observations.
    """
    _require_min_length(prices, window, f"the {window}-period SMA")
    return prices.rolling(window=window).mean()


def ema(prices: pd.Series, window: int = 20) -> pd.Series:
    """Exponential Moving Average.

    Like the SMA, but weights recent prices more heavily using a smoothing
    factor ``alpha = 2 / (window + 1)``, so it reacts faster to new price
    moves than a simple average of the same length.

    Parameters
    ----------
    prices : pandas.Series
        A series of prices, indexed by date.
    window : int, optional
        The EMA span; larger values react more slowly. Defaults to ``20``.

    Returns
    -------
    pandas.Series
        The exponential moving average, same length as ``prices``.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``window`` observations.
    """
    _require_min_length(prices, window, f"the {window}-period EMA")
    return prices.ewm(span=window, adjust=False).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index.

    Measures the speed and magnitude of recent price changes on a 0-100
    scale. Readings above 70 are conventionally read as "overbought" (the
    price has risen a lot, a pullback may follow); readings below 30 as
    "oversold" (the price has fallen a lot, a bounce may follow).

    Computed using Wilder's smoothing (an exponential moving average with
    ``alpha = 1/window``) for the average gain/loss, which is the standard
    convention for RSI — *not* a simple rolling mean.

    Parameters
    ----------
    prices : pandas.Series
        A series of prices, indexed by date.
    window : int, optional
        The lookback period. Defaults to ``14``.

    Returns
    -------
    pandas.Series
        RSI values in ``[0, 100]``. The first point is ``NaN`` (no price
        change is defined there yet).

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``window + 1`` observations (``window``
        price changes require ``window + 1`` prices).
    """
    _require_min_length(prices, window + 1, f"the {window}-period RSI")

    changes = prices.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)  # absolute value of negative changes

    # Wilder's smoothing: an EWM with alpha = 1/window, not a plain rolling mean.
    avg_gain = gains.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means no losses to smooth -> RSI is defined as 100, not
    # the inf/NaN that falls out of dividing by zero.
    rsi_values = rsi_values.where(avg_loss != 0.0, 100.0)
    return rsi_values


def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, pd.Series]:
    """Moving Average Convergence Divergence.

    A trend/momentum indicator built from the difference between a fast and a
    slow EMA. When the MACD line crosses above its signal line, momentum is
    turning up; crossing below, momentum is turning down. The histogram (the
    gap between the two lines) shows how strong that momentum is.

    Parameters
    ----------
    prices : pandas.Series
        A series of prices, indexed by date.
    fast : int, optional
        Fast EMA span. Defaults to ``12``.
    slow : int, optional
        Slow EMA span. Defaults to ``26``.
    signal : int, optional
        EMA span applied to the MACD line to get the signal line. Defaults to
        ``9``.

    Returns
    -------
    dict
        ``{"macd_line": pandas.Series, "signal_line": pandas.Series,
        "histogram": pandas.Series}``.

    Raises
    ------
    ValueError
        If ``prices`` does not have enough observations for the requested
        ``fast``/``slow``/``signal`` spans (raised by the underlying
        :func:`ema` calls).
    """
    macd_line = ema(prices, fast) - ema(prices, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }


def bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Dict[str, pd.Series]:
    """Bollinger Bands.

    Volatility bands plotted around a moving average: the middle band is a
    simple moving average, and the upper/lower bands sit ``num_std`` standard
    deviations above/below it. The bands widen when the price is volatile and
    narrow when it's calm; a price pushing through a band is often read as a
    move to an extreme relative to its recent range.

    Parameters
    ----------
    prices : pandas.Series
        A series of prices, indexed by date.
    window : int, optional
        Moving-average and standard-deviation lookback. Defaults to ``20``.
    num_std : float, optional
        Number of standard deviations for the bands. Defaults to ``2.0``.

    Returns
    -------
    dict
        ``{"upper": pandas.Series, "middle": pandas.Series,
        "lower": pandas.Series}``.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``window`` observations.
    """
    middle_band = sma(prices, window)
    rolling_std = prices.rolling(window=window).std()
    upper_band = middle_band + num_std * rolling_std
    lower_band = middle_band - num_std * rolling_std
    return {"upper": upper_band, "middle": middle_band, "lower": lower_band}


def volume_trend(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume trend ratio: current volume relative to its recent average.

    ``volume_trend[t] = volume[t] / mean(volume[t-window+1 : t+1])``. Values
    well above 1 mean unusually heavy trading — a real, conviction-backed
    move. Values near or below 1 mean trading is light — a price move on low
    volume is weaker evidence of a genuine trend.

    Parameters
    ----------
    volume : pandas.Series
        A series of traded volume, indexed by date.
    window : int, optional
        Lookback period for the rolling average. Defaults to ``20``.

    Returns
    -------
    pandas.Series
        The ratio of volume to its rolling average.

    Raises
    ------
    ValueError
        If ``volume`` has fewer than ``window`` observations.
    """
    _require_min_length(volume, window, f"the {window}-period volume trend")
    rolling_avg = volume.rolling(window=window).mean()
    return volume / rolling_avg
