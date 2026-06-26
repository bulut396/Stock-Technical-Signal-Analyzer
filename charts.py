"""Interactive Plotly charts for price action and technical indicators."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go


def plot_candlestick_with_indicators(
    ohlcv_df: pd.DataFrame,
    bollinger: Dict[str, pd.Series],
    sma_20: pd.Series,
    sma_50: pd.Series,
) -> go.Figure:
    """Build a candlestick chart overlaid with moving averages and Bollinger Bands.

    Parameters
    ----------
    ohlcv_df : pandas.DataFrame
        Price history with ``Open``, ``High``, ``Low``, ``Close`` columns.
    bollinger : dict
        Output of :func:`indicators.bollinger_bands` (keys ``"upper"``,
        ``"middle"``, ``"lower"``). The middle band is not drawn separately
        here when it matches ``sma_20`` — the SMA-20 line already shows it,
        so it isn't computed or plotted twice.
    sma_20 : pandas.Series
        20-period simple moving average.
    sma_50 : pandas.Series
        50-period simple moving average.

    Returns
    -------
    plotly.graph_objects.Figure
        Candlestick chart with SMA-20, SMA-50, and Bollinger upper/lower
        bands overlaid.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=ohlcv_df.index,
            open=ohlcv_df["Open"],
            high=ohlcv_df["High"],
            low=ohlcv_df["Low"],
            close=ohlcv_df["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sma_20.index, y=sma_20, mode="lines",
            name="SMA 20", line=dict(color="#2E86DE", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sma_50.index, y=sma_50, mode="lines",
            name="SMA 50", line=dict(color="#F39C12", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bollinger["upper"].index, y=bollinger["upper"], mode="lines",
            name="Bollinger Upper", line=dict(color="#95A5A6", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bollinger["lower"].index, y=bollinger["lower"], mode="lines",
            name="Bollinger Lower", line=dict(color="#95A5A6", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(149, 165, 166, 0.08)",
        )
    )

    fig.update_layout(
        title="Price with Moving Averages and Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_rsi(rsi_series: pd.Series) -> go.Figure:
    """Build a line chart of RSI with overbought/oversold reference zones.

    Parameters
    ----------
    rsi_series : pandas.Series
        Output of :func:`indicators.rsi`.

    Returns
    -------
    plotly.graph_objects.Figure
        RSI line with dashed reference lines at 40 (oversold) and 60
        (overbought) — matching the signal engine's calibrated thresholds —
        shaded so the extreme zones are visually obvious.
    """
    fig = go.Figure()

    fig.add_hrect(y0=60, y1=100, fillcolor="#E74C3C", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0, y1=40, fillcolor="#27AE60", opacity=0.07, line_width=0)
    fig.add_hline(y=60, line=dict(color="#E74C3C", dash="dash", width=1))
    fig.add_hline(y=40, line=dict(color="#27AE60", dash="dash", width=1))

    fig.add_trace(
        go.Scatter(
            x=rsi_series.index, y=rsi_series, mode="lines",
            name="RSI", line=dict(color="#8E44AD", width=1.5),
        )
    )

    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_macd(macd_dict: Dict[str, pd.Series]) -> go.Figure:
    """Build the classic MACD chart: line, signal line, and histogram.

    Parameters
    ----------
    macd_dict : dict
        Output of :func:`indicators.macd` (keys ``"macd_line"``,
        ``"signal_line"``, ``"histogram"``).

    Returns
    -------
    plotly.graph_objects.Figure
        MACD line and signal line as overlaid lines, with the histogram
        (their difference) as bars beneath them.
    """
    histogram = macd_dict["histogram"]
    bar_colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in histogram]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=histogram.index, y=histogram, name="Histogram",
            marker_color=bar_colors, opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macd_dict["macd_line"].index, y=macd_dict["macd_line"], mode="lines",
            name="MACD Line", line=dict(color="#2E86DE", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=macd_dict["signal_line"].index, y=macd_dict["signal_line"], mode="lines",
            name="Signal Line", line=dict(color="#F39C12", width=1.5),
        )
    )

    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Date",
        yaxis_title="MACD",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
