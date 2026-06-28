"""Interactive Plotly charts for backtest results."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_equity_curve(strategy_values: pd.Series, buy_hold_values: pd.Series) -> go.Figure:
    """Plot strategy portfolio value against the buy-and-hold benchmark.

    Parameters
    ----------
    strategy_values : pandas.Series
        The ``portfolio_value`` column of :func:`backtester.simulate_strategy`'s
        output.
    buy_hold_values : pandas.Series
        Output of :func:`backtester.compute_buy_and_hold`.

    Returns
    -------
    plotly.graph_objects.Figure
        Two lines over time — wherever they diverge is where the
        signal-driven strategy and simply holding the stock disagreed.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=strategy_values.index, y=strategy_values, mode="lines",
            name="Strategy", line=dict(color="#2E86DE", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy_hold_values.index, y=buy_hold_values, mode="lines",
            name="Buy & Hold", line=dict(color="#95A5A6", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Strategy vs. Buy & Hold — Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_signal_markers(ohlcv_df: pd.DataFrame, signal_history_df: pd.DataFrame) -> go.Figure:
    """Plot price with BUY/SELL signal markers at the dates they fired.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        Price history with a ``Close`` column, for the price line itself
        (may include warm-up days before the first backtested signal).
    signal_history_df : pandas.DataFrame
        Output of :func:`backtester.generate_signal_history` — columns
        ``signal`` and ``close``, indexed by date.

    Returns
    -------
    plotly.graph_objects.Figure
        The closing-price line, with up-triangles marking BUY days and
        down-triangles marking SELL days.
    """
    buys = signal_history_df[signal_history_df["signal"] == "BUY"]
    sells = signal_history_df[signal_history_df["signal"] == "SELL"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ohlcv_df.index, y=ohlcv_df["Close"], mode="lines",
            name="Close", line=dict(color="#34495E", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buys.index, y=buys["close"], mode="markers", name="BUY signal",
            marker=dict(symbol="triangle-up", size=11, color="#27AE60"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index, y=sells["close"], mode="markers", name="SELL signal",
            marker=dict(symbol="triangle-down", size=11, color="#E74C3C"),
        )
    )

    fig.update_layout(
        title="Price with Backtested Signal Markers",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
