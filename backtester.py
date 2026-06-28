"""Historical backtesting of the rule-based signal engine.

Walks day-by-day through price history, regenerating the signal at each day
using only data available up to that day, simulates a simple all-in/all-out
trading strategy driven by those signals, and evaluates the result against a
buy-and-hold benchmark using performance metrics from the sibling
``risk_toolkit`` package (https://github.com/bulut396/Python-Risk-Management-Library).

No risk/performance math is reimplemented here — every metric in
:func:`evaluate_backtest` is computed by calling ``risk_toolkit`` directly.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from risk_toolkit import (
    annualized_volatility,
    calmar_ratio,
    historical_var,
    max_drawdown,
    prices_to_returns,
    sharpe_ratio,
    sortino_ratio,
)

from signal_engine import generate_signal

_MIN_LOOKBACK_DEFAULT = 30


def generate_signal_history(
    ohlcv_df: pd.DataFrame, min_lookback: int = _MIN_LOOKBACK_DEFAULT
) -> pd.DataFrame:
    """Replay the signal engine across history, one day at a time.

    For each day from ``min_lookback`` onward, the signal is generated using
    only the rows up to and including that day — never later rows. This is
    the look-ahead-bias guard: ``ohlcv_df.iloc[: i + 1]`` for day index ``i``
    is a frame whose *last* row is day ``i`` and which contains nothing past
    it, so every indicator computed inside :func:`signal_engine.generate_signal`
    (rolling means, RSI, MACD, Bollinger Bands) only ever sees prices that
    would have actually been known on day ``i``. Each iteration re-slices and
    re-passes the *growing* prefix of the frame rather than ever touching the
    full frame at once, so there is no path by which a future bar can leak
    into a past day's signal.

    This recomputes every indicator from scratch on every iteration — O(n)
    signal calls, each itself O(window) — which is intentionally simple and
    obviously correct rather than fast. It is fine for the few hundred bars a
    typical backtest window covers (a year of daily bars is ~252 rows); it
    will be noticeably slow well past ~1000 days, since cost grows with the
    number of days being backtested.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        Full OHLCV price history, oldest to newest (e.g. the output of
        :func:`price_fetcher.fetch_ohlcv`).
    min_lookback : int, optional
        The first day a signal is generated is the day at index
        ``min_lookback`` (0-indexed) — i.e. the first window passed to
        :func:`signal_engine.generate_signal` has ``min_lookback + 1`` rows.
        Must be at least :data:`signal_engine._MIN_BARS` (30); defaults to 30.

    Returns
    -------
    pandas.DataFrame
        Indexed by date (one row per backtested day), with columns
        ``signal`` (``"BUY"``/``"SELL"``/``"HOLD"``) and ``close`` (that
        day's closing price).

    Raises
    ------
    ValueError
        If ``ohlcv_df`` has fewer than ``min_lookback + 1`` rows.
    """
    if len(ohlcv_df) < min_lookback + 1:
        raise ValueError(
            f"At least {min_lookback + 1} bars of price history are needed to "
            "backtest with this lookback. Please choose a longer time period."
        )

    records = []
    dates = []
    for i in range(min_lookback, len(ohlcv_df)):
        # Look-ahead guard: this prefix ends at day i and contains no later
        # rows, so generate_signal cannot see beyond what was knowable then.
        window = ohlcv_df.iloc[: i + 1]
        result = generate_signal(window)
        records.append({"signal": result["signal"], "close": result["latest_close"]})
        dates.append(ohlcv_df.index[i])

    return pd.DataFrame(records, index=pd.Index(dates, name=ohlcv_df.index.name))


def simulate_strategy(
    signal_history_df: pd.DataFrame, initial_capital: float = 10000.0
) -> pd.DataFrame:
    """Simulate a simple all-in/all-out trading strategy from a signal history.

    Rules (deliberately simple, no black box):

    - **BUY** and not currently holding: invest 100% of available cash in the
      stock at that day's close (no partial sizing, no leverage).
    - **SELL** and currently holding: liquidate the entire position to cash
      at that day's close.
    - **HOLD**, or a BUY/SELL that doesn't change the current position
      (e.g. a BUY while already holding): do nothing.

    No transaction costs or slippage are modeled — see the README for why
    this is an intentional v1 simplification, not an oversight.

    Parameters
    ----------
    signal_history_df : pandas.DataFrame
        Output of :func:`generate_signal_history` — columns ``signal`` and
        ``close``, indexed by date.
    initial_capital : float, optional
        Starting hypothetical cash balance. Defaults to ``10000.0``.

    Returns
    -------
    pandas.DataFrame
        Same date index as ``signal_history_df``, with columns ``signal``,
        ``close``, ``position`` (``"CASH"`` or ``"HOLDING"``), and
        ``portfolio_value`` (cash, or shares times that day's close,
        whichever is held).
    """
    cash = float(initial_capital)
    shares = 0.0
    position = "CASH"
    records = []

    for date, row in signal_history_df.iterrows():
        signal = row["signal"]
        close = float(row["close"])

        if signal == "BUY" and position == "CASH":
            shares = cash / close
            cash = 0.0
            position = "HOLDING"
        elif signal == "SELL" and position == "HOLDING":
            cash = shares * close
            shares = 0.0
            position = "CASH"
        # HOLD, or a BUY/SELL that doesn't change position: no action.

        portfolio_value = cash + shares * close
        records.append(
            {
                "signal": signal,
                "close": close,
                "position": position,
                "portfolio_value": portfolio_value,
            }
        )

    return pd.DataFrame(records, index=signal_history_df.index)


def compute_buy_and_hold(
    ohlcv_df: pd.DataFrame, initial_capital: float = 10000.0
) -> pd.Series:
    """Benchmark: hold the stock for the whole window, untouched.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        Price history with a ``Close`` column, indexed by date. To compare
        fairly against a strategy result, pass the same date range as that
        strategy's ``signal_history_df`` (i.e. invest on the first day a
        signal could have been generated, not earlier warm-up days the
        strategy never actually traded on).
    initial_capital : float, optional
        Starting hypothetical cash balance, fully invested on day one.
        Defaults to ``10000.0``.

    Returns
    -------
    pandas.Series
        ``initial_capital`` worth of shares (bought at the first day's
        close), revalued at each subsequent day's close. Same date index as
        ``ohlcv_df``.
    """
    close = ohlcv_df["Close"].astype("float64")
    shares = float(initial_capital) / float(close.iloc[0])
    values = close * shares
    values.name = "buy_and_hold_value"
    return values


def _trade_stats(strategy_results: pd.DataFrame) -> Tuple[int, float | None]:
    """Count round-trip trades and the fraction that were profitable.

    A trade opens on the day the strategy transitions CASH -> HOLDING (a BUY
    that was actually acted on) and closes on the day it transitions back
    HOLDING -> CASH (a SELL that was actually acted on), or — if the
    backtest window ends while still holding — closes at the last available
    close. A trade is a win if its exit price is strictly higher than its
    entry price. This counts actionable position changes rather than raw BUY
    signal counts, since a repeated BUY signal while already holding doesn't
    open a new trade.

    Returns
    -------
    tuple
        ``(num_trades, win_rate)``. ``win_rate`` is ``None`` if no trades
        occurred (avoids a 0/0 division).
    """
    position = strategy_results["position"]
    close = strategy_results["close"]

    trades = []
    entry_price = None
    prev_position = "CASH"
    for date in strategy_results.index:
        current_position = position.loc[date]
        if current_position == "HOLDING" and prev_position == "CASH":
            entry_price = float(close.loc[date])
        elif current_position == "CASH" and prev_position == "HOLDING":
            trades.append((entry_price, float(close.loc[date])))
            entry_price = None
        prev_position = current_position

    if entry_price is not None:
        # Still holding when the window ends: close out at the last price.
        trades.append((entry_price, float(close.iloc[-1])))

    num_trades = len(trades)
    if num_trades == 0:
        return 0, None

    wins = sum(1 for entry, exit_ in trades if exit_ > entry)
    return num_trades, wins / num_trades


def _safe_metric(fn, *args, **kwargs) -> Any:
    """Call a risk_toolkit metric, substituting a note if it's undefined.

    risk_toolkit raises ``ValueError`` for metrics that are mathematically
    undefined on a given series (e.g. Sharpe/Sortino with zero-variance
    returns, Calmar with zero max drawdown). Catching per-metric — rather
    than around the whole batch — means one undefined metric (say, Calmar on
    a buy-and-hold series that never drew down) doesn't discard the other,
    perfectly well-defined metrics alongside it.
    """
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        return f"undefined: {exc}"


def _compute_side_metrics(
    portfolio_values: pd.Series,
    risk_free_rate: float,
    periods_per_year: int,
    var_confidence: float,
) -> Dict[str, Any]:
    """Compute total return plus every risk_toolkit metric for one value series.

    Each metric is computed independently (see :func:`_safe_metric`), so an
    undefined metric shows up as a ``"undefined: ..."`` string in its own
    slot rather than discarding the rest of the dict.
    """
    returns = prices_to_returns(portfolio_values)
    return {
        "total_return": float(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0),
        "annualized_volatility": _safe_metric(annualized_volatility, returns, periods_per_year),
        "sharpe_ratio": _safe_metric(
            sharpe_ratio, returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "sortino_ratio": _safe_metric(
            sortino_ratio, returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "max_drawdown": _safe_metric(lambda r: max_drawdown(r)["max_drawdown"], returns),
        "calmar_ratio": _safe_metric(calmar_ratio, returns, periods_per_year=periods_per_year),
        "historical_var_95": _safe_metric(historical_var, returns, var_confidence),
    }


def evaluate_backtest(
    strategy_results: pd.DataFrame,
    buy_hold_values: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    var_confidence: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate strategy and buy-and-hold portfolio value series side by side.

    Every metric below is computed by calling ``risk_toolkit`` directly on
    the daily returns derived from each portfolio value series — none of
    this math (volatility, Sharpe, Sortino, max drawdown, Calmar, VaR) is
    reimplemented in this project.

    Parameters
    ----------
    strategy_results : pandas.DataFrame
        Output of :func:`simulate_strategy` (needs its ``portfolio_value``
        and ``position``/``close`` columns — the latter two are used only to
        derive the trade-level win rate, not for the risk_toolkit metrics).
    buy_hold_values : pandas.Series
        Output of :func:`compute_buy_and_hold`.
    risk_free_rate : float, optional
        Annual risk-free rate passed through to Sharpe/Sortino. Defaults to
        ``0.0``.
    periods_per_year : int, optional
        Trading days per year used for annualizing. Defaults to ``252``.
    var_confidence : float, optional
        Confidence level for historical VaR. Defaults to ``0.95``.

    Returns
    -------
    dict
        ``{"strategy": {...}, "buy_and_hold": {...}}``. Each side normally
        contains ``total_return``, ``annualized_volatility``,
        ``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown``,
        ``calmar_ratio``, and ``historical_var_95``. ``"strategy"`` also
        contains ``num_trades`` and ``win_rate`` (the fraction of round-trip
        trades that were profitable — see :func:`_trade_stats`). If any
        individual metric is mathematically undefined for a series (e.g.
        zero-variance returns make Sharpe/Sortino/Calmar undefined), that
        metric's value is the string ``"undefined: <reason>"`` rather than
        the rest of the dict being discarded — see :func:`_safe_metric`. If
        the strategy never traded at all (every day was SELL/HOLD), the
        entire ``"strategy"`` value is replaced with
        ``{"note": "No trades were made in this period — all signals were SELL/HOLD."}``
        instead, since in that case *every* metric is undefined (the
        portfolio value never moves) and a single explanatory note is
        clearer than seven repeated ``"undefined: ..."`` entries.
    """
    num_trades, win_rate = _trade_stats(strategy_results)

    if num_trades == 0:
        strategy_metrics: Dict[str, Any] = {
            "note": "No trades were made in this period — all signals were SELL/HOLD."
        }
    else:
        strategy_metrics = _compute_side_metrics(
            strategy_results["portfolio_value"], risk_free_rate, periods_per_year, var_confidence
        )
        strategy_metrics["num_trades"] = num_trades
        strategy_metrics["win_rate"] = win_rate

    buy_hold_metrics = _compute_side_metrics(
        buy_hold_values, risk_free_rate, periods_per_year, var_confidence
    )

    return {"strategy": strategy_metrics, "buy_and_hold": buy_hold_metrics}
