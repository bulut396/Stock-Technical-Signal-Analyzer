"""Rule-based BUY/SELL/HOLD signal engine.

Combines RSI, MACD, and Bollinger Bands into a transparent, explainable
signal via simple named voting rules — no black-box scoring, no ML model.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from indicators import bollinger_bands, macd, rsi, volume_trend

_MIN_BARS = 30


def _rsi_vote(latest_rsi: float) -> str:
    """RSI rule: oversold votes BUY, overbought votes SELL, else no vote."""
    if latest_rsi < 30:
        return "BUY"
    if latest_rsi > 70:
        return "SELL"
    return "NONE"


def _macd_vote(histogram: pd.Series) -> str:
    """MACD rule: a histogram sign flip on the latest bar is a crossover."""
    previous, latest = histogram.iloc[-2], histogram.iloc[-1]
    if previous <= 0 and latest > 0:
        return "BUY"
    if previous >= 0 and latest < 0:
        return "SELL"
    return "NONE"


def _bollinger_vote(latest_close: float, latest_lower: float, latest_upper: float) -> str:
    """Bollinger rule: price at the lower/upper extreme of its recent range."""
    if latest_close <= latest_lower:
        return "BUY"
    if latest_close >= latest_upper:
        return "SELL"
    return "NONE"


def _volume_note(latest_volume_trend: float) -> str:
    """Plain-language caveat/confirmation based on relative trading volume."""
    if latest_volume_trend < 1.0:
        return (
            "Trading volume on the latest bar is below its recent average, "
            "so this move has relatively low conviction behind it."
        )
    if latest_volume_trend >= 1.5:
        return (
            "Trading volume on the latest bar is well above its recent "
            "average, so this move looks volume-confirmed."
        )
    return "Trading volume on the latest bar is in line with its recent average."


def generate_signal(ohlcv_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a transparent, rule-based BUY/SELL/HOLD signal.

    Computes RSI, MACD, Bollinger Bands, and a volume-trend ratio on the
    supplied price history (via the indicator functions in
    :mod:`indicators`), then applies three simple, named voting rules:

    - **RSI rule**: RSI below 30 votes BUY ("oversold"); above 70 votes SELL
      ("overbought"); otherwise no vote.
    - **MACD rule**: the MACD histogram flipping from negative to positive on
      the latest bar (a bullish crossover) votes BUY; flipping from positive
      to negative votes SELL; otherwise no vote.
    - **Bollinger rule**: the latest close at or below the lower band votes
      BUY ("price at a lower extreme"); at or above the upper band votes
      SELL; otherwise no vote.

    Trading volume does not cast a vote; it only qualifies the result with a
    plain-language note about conviction.

    Decision: 2 or more BUY votes with 0 SELL votes gives ``"BUY"``; 2 or more
    SELL votes with 0 BUY votes gives ``"SELL"``; any other combination
    (mixed or weak signals) gives ``"HOLD"``.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        Price history with at least ``Close`` and ``Volume`` columns (e.g.
        the output of :func:`price_fetcher.fetch_ohlcv`).

    Returns
    -------
    dict
        ``{"signal": "BUY"|"SELL"|"HOLD", "votes": {"rsi": ..., "macd": ...,
        "bollinger": ...}, "volume_note": str, "rsi_value": float,
        "latest_close": float}``.

    Raises
    ------
    ValueError
        If ``ohlcv_df`` has fewer than 30 bars of history — not enough to
        compute all indicators reliably, so no signal is generated rather
        than risk a misleading one.
    """
    if len(ohlcv_df) < _MIN_BARS:
        raise ValueError(
            f"At least {_MIN_BARS} bars of price history are needed to "
            "generate a reliable signal. Please choose a longer time period."
        )

    close = ohlcv_df["Close"]

    rsi_series = rsi(close)
    macd_result = macd(close)
    bands = bollinger_bands(close)
    vol_trend = volume_trend(ohlcv_df["Volume"])

    latest_close = float(close.iloc[-1])
    latest_rsi = float(rsi_series.iloc[-1])

    votes = {
        "rsi": _rsi_vote(latest_rsi),
        "macd": _macd_vote(macd_result["histogram"]),
        "bollinger": _bollinger_vote(
            latest_close,
            float(bands["lower"].iloc[-1]),
            float(bands["upper"].iloc[-1]),
        ),
    }

    buy_votes = sum(1 for v in votes.values() if v == "BUY")
    sell_votes = sum(1 for v in votes.values() if v == "SELL")

    if buy_votes >= 2 and sell_votes == 0:
        signal = "BUY"
    elif sell_votes >= 2 and buy_votes == 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "votes": votes,
        "volume_note": _volume_note(float(vol_trend.iloc[-1])),
        "rsi_value": latest_rsi,
        "latest_close": latest_close,
    }
