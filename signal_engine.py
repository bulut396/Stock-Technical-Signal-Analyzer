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
    """RSI rule: oversold votes BUY, overbought votes SELL, else no vote.

    Uses a 45/55 band — loosened further from an earlier 40/60 calibration
    (itself already looser than the classic textbook 30/70 levels). This is a
    deliberate sensitivity preference, not a claim that 45/55 is "the
    correct" RSI threshold: it's chosen so the rule fires more often (less
    often landing on no vote at all), at the cost of being an even less
    extreme reading than 40/60 or the classic 30/70 definition.
    """
    if latest_rsi < 45:
        return "BUY"
    if latest_rsi > 55:
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


_BAND_WIDTH_EPSILON = 1e-9


def _bollinger_vote(latest_close: float, latest_lower: float, latest_upper: float) -> str:
    """Bollinger rule: votes based on where price sits within its recent band.

    Rather than requiring the close to touch or exceed a band edge (the
    classic Bollinger signal), this computes ``position = (close - lower) /
    (upper - lower)`` — a 0-to-1 reading of where price sits within its own
    recent volatility range — and votes BUY if that position is in the
    bottom 30% of the range, SELL if it's in the top 30%. This zone was
    loosened from an earlier 20%-zone calibration (same spirit as the RSI
    45/55 change): it fires more often than both the 20%-zone version and the
    textbook "price touches the band" rule, at the cost of being a looser,
    less extreme confirmation than either.

    If the band has (near) zero width — e.g. a perfectly flat price series —
    "position within the band" is undefined, so this votes ``"NONE"`` rather
    than risk a division by zero or a meaningless ratio.
    """
    band_width = latest_upper - latest_lower
    if band_width <= _BAND_WIDTH_EPSILON:
        return "NONE"

    position = (latest_close - latest_lower) / band_width
    if position <= 0.3:
        return "BUY"
    if position >= 0.7:
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

    - **RSI rule**: RSI below 45 votes BUY ("oversold"); above 55 votes SELL
      ("overbought"); otherwise no vote. This 45/55 band is a deliberate
      sensitivity calibration, looser than an earlier 40/60 calibration and
      the classic textbook 30/70 levels — see :func:`_rsi_vote` for why.
    - **MACD rule**: the MACD histogram flipping from negative to positive on
      the latest bar (a bullish crossover) votes BUY; flipping from positive
      to negative votes SELL; otherwise no vote.
    - **Bollinger rule**: votes BUY if the close sits in the bottom 30% of
      its position between the lower and upper band; SELL if it sits in the
      top 30%; otherwise no vote. This 30%-zone version is a deliberate
      sensitivity calibration, looser than an earlier 20%-zone calibration
      and the classic "price touches the band" standard — see
      :func:`_bollinger_vote` for why.

    Trading volume does not cast a vote; it only qualifies the result with a
    plain-language note about conviction.

    Decision: whichever side has more votes wins, as long as it has at least
    one — more BUY votes than SELL votes gives ``"BUY"``; more SELL votes
    than BUY votes gives ``"SELL"``. A tie (including 0 votes either way)
    gives ``"HOLD"`` — so a single rule firing with nothing opposing it is
    now enough to produce a signal, but a genuine conflict between rules
    (e.g. RSI says BUY while Bollinger says SELL) still correctly resolves to
    HOLD rather than arbitrarily picking a side.

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

    if buy_votes > sell_votes and buy_votes >= 1:
        signal = "BUY"
    elif sell_votes > buy_votes and sell_votes >= 1:
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
