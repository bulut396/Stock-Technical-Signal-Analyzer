"""stock-technical-signal-analyzer — technical indicators and a rule-based signal.

Fetches OHLCV price history via yfinance, computes RSI/MACD/Bollinger Bands
from scratch, combines them into a transparent BUY/SELL/HOLD signal, and
visualizes everything on an interactive candlestick chart. Also includes a
historical backtest mode that replays the signal engine day-by-day over past
prices and compares a simple signal-driven trading simulation against a
buy-and-hold benchmark. Run with::

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtest_charts import plot_equity_curve, plot_signal_markers
from backtester import (
    compute_buy_and_hold,
    evaluate_backtest,
    generate_signal_history,
    simulate_strategy,
)
from charts import plot_candlestick_with_indicators, plot_macd, plot_rsi
from indicators import bollinger_bands, macd, rsi, sma
from price_fetcher import fetch_ohlcv
from signal_engine import generate_signal

PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y"]
BACKTEST_PERIOD_OPTIONS = ["6mo", "1y", "2y"]
BACKTEST_INITIAL_CAPITAL = 10000.0

_SIGNAL_RENDERERS = {
    "BUY": st.success,
    "SELL": st.error,
    "HOLD": st.info,
}

_METRIC_LABELS = {
    "total_return": "Total Return",
    "annualized_volatility": "Annualized Volatility",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "max_drawdown": "Max Drawdown",
    "calmar_ratio": "Calmar Ratio",
    "historical_var_95": "Historical VaR (95%)",
    "num_trades": "Number of Trades",
    "win_rate": "Win Rate",
}

_METRIC_CAPTIONS = {
    "total_return": "Total gain/loss over the backtest period — higher is better.",
    "annualized_volatility": "How much the portfolio's value swings in a typical year — lower is calmer, but not automatically better or worse on its own.",
    "sharpe_ratio": "Risk-adjusted return — higher is better.",
    "sortino_ratio": "Like Sharpe, but only counts downside swings against it — higher is better.",
    "max_drawdown": "The worst peak-to-trough decline over the period — closer to 0% is better.",
    "calmar_ratio": "Return relative to that worst drawdown — higher is better.",
    "historical_var_95": "The 1-day loss not expected to be exceeded on 95% of days — closer to 0% is better.",
    "num_trades": "How many complete buy-then-sell round trips the strategy made.",
    "win_rate": "Of those round trips, the fraction that ended in a higher price than they started — higher is better.",
}

_METRIC_ORDER = [
    "total_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "calmar_ratio", "historical_var_95", "num_trades", "win_rate",
]
_PERCENT_METRICS = {"total_return", "max_drawdown", "historical_var_95", "win_rate"}


def _render_signal(signal_result: dict) -> None:
    """Render the headline signal badge plus an expandable rule breakdown."""
    signal = signal_result["signal"]
    c1, c2 = st.columns([1, 2])
    c1.metric("Signal", signal)
    c2.metric("Latest close", f"${signal_result['latest_close']:.2f}")
    c2.metric("RSI", f"{signal_result['rsi_value']:.1f}")

    _SIGNAL_RENDERERS[signal](f"**Overall signal: {signal}**")
    st.caption(signal_result["volume_note"])

    with st.expander("Why this signal?"):
        votes = signal_result["votes"]
        st.markdown(
            f"- **RSI rule**: {votes['rsi']} "
            f"(RSI is {signal_result['rsi_value']:.1f}; below 45 votes BUY/"
            "oversold, above 55 votes SELL/overbought)."
        )
        st.markdown(
            f"- **MACD rule**: {votes['macd']} "
            "(votes BUY if the MACD histogram just flipped from negative to "
            "positive, SELL if it just flipped from positive to negative)."
        )
        st.markdown(
            f"- **Bollinger rule**: {votes['bollinger']} "
            "(votes BUY if the latest close sits in the bottom 30% of its "
            "position between the lower and upper band, SELL if it sits in "
            "the top 30%)."
        )
        st.markdown(f"- **Volume note**: {signal_result['volume_note']}")
        st.markdown(
            "Whichever side — BUY or SELL — has more votes wins, as long as "
            "it has at least one; a tie (including no votes either way) is "
            "HOLD."
        )


def run_analysis(ticker: str, period: str) -> None:
    """Fetch price data, compute indicators, generate a signal, and render it."""
    try:
        with st.spinner(f"Fetching price history for {ticker.upper()}..."):
            ohlcv = fetch_ohlcv(ticker, period=period)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        close = ohlcv["Close"]
        sma_20 = sma(close, window=20)
        sma_50 = sma(close, window=50)
        rsi_series = rsi(close)
        macd_result = macd(close)
        bands = bollinger_bands(close)
        signal_result = generate_signal(ohlcv)
    except ValueError as exc:
        st.warning(f"{exc} Try a longer time period.")
        return

    st.subheader(f"{ticker.upper()} — {period}")
    _render_signal(signal_result)

    st.markdown(
        "Candlestick chart of the price, with the 20- and 50-period moving "
        "averages and Bollinger Bands overlaid to show the trend and recent "
        "volatility range."
    )
    st.plotly_chart(
        plot_candlestick_with_indicators(ohlcv, bands, sma_20, sma_50),
        use_container_width=True,
    )

    st.markdown(
        "RSI shows whether the stock has moved unusually far, unusually "
        "fast — the shaded zones mark this app's calibrated overbought "
        "(above 55) and oversold (below 45) thresholds."
    )
    st.plotly_chart(plot_rsi(rsi_series), use_container_width=True)

    st.markdown(
        "MACD tracks momentum — when the blue line crosses the orange "
        "signal line, momentum is shifting, and the bars show how strong "
        "that shift is."
    )
    st.plotly_chart(plot_macd(macd_result), use_container_width=True)


def _format_metric_value(key: str, value) -> str:
    """Format one metric value for the side-by-side table.

    ``risk_toolkit`` calls that turned out to be mathematically undefined
    (e.g. Sharpe with zero-variance returns) come through as plain strings
    starting with ``"undefined: ..."`` — those are shown as-is rather than
    being formatted as numbers.
    """
    if isinstance(value, str):
        return value
    if key in _PERCENT_METRICS:
        return f"{value:.1%}"
    if key == "num_trades":
        return str(int(value))
    return f"{value:.3f}"


def _render_backtest_metrics(evaluation: dict) -> None:
    """Render the strategy-vs-buy-and-hold metrics table and captions."""
    strategy = evaluation["strategy"]
    buy_hold = evaluation["buy_and_hold"]

    if "note" in strategy:
        st.warning(strategy["note"])

    present_keys = [
        key for key in _METRIC_ORDER if key in strategy or key in buy_hold
    ]
    rows = [
        {
            "Metric": _METRIC_LABELS[key],
            "Strategy": _format_metric_value(key, strategy[key]) if key in strategy else "N/A",
            "Buy & Hold": _format_metric_value(key, buy_hold[key]) if key in buy_hold else "N/A",
        }
        for key in present_keys
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    with st.expander("What do these metrics mean?"):
        for key in present_keys:
            st.markdown(f"- **{_METRIC_LABELS[key]}**: {_METRIC_CAPTIONS[key]}")


def run_backtest(ticker: str, period: str) -> None:
    """Fetch history, replay the signal engine over it, simulate, and report."""
    try:
        with st.spinner(f"Fetching {period} of price history for {ticker.upper()}..."):
            ohlcv = fetch_ohlcv(ticker, period=period)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner(
            "Replaying the signal engine day-by-day — this takes longer than "
            "the live signal above, since every historical day recomputes "
            "its own indicators from scratch."
        ):
            signal_history = generate_signal_history(ohlcv)
            strategy_results = simulate_strategy(
                signal_history, initial_capital=BACKTEST_INITIAL_CAPITAL
            )
            buy_hold_values = compute_buy_and_hold(
                ohlcv.loc[signal_history.index], initial_capital=BACKTEST_INITIAL_CAPITAL
            )
            evaluation = evaluate_backtest(strategy_results, buy_hold_values)
    except ValueError as exc:
        st.warning(f"{exc} Try a longer time period.")
        return

    st.subheader(f"{ticker.upper()} — Backtest over {period}")
    st.caption(
        f"Simulated trading from {signal_history.index[0].date()} to "
        f"{signal_history.index[-1].date()}, starting with "
        f"${BACKTEST_INITIAL_CAPITAL:,.0f} of hypothetical capital."
    )

    st.markdown(
        "**Portfolio value over time** — the signal-driven strategy vs. "
        "simply buying and holding the stock for the same period."
    )
    st.plotly_chart(
        plot_equity_curve(strategy_results["portfolio_value"], buy_hold_values),
        use_container_width=True,
    )

    st.markdown(
        "**Where the strategy traded** — green up-triangles are BUY signals, "
        "red down-triangles are SELL signals, at the dates they fired."
    )
    st.plotly_chart(plot_signal_markers(ohlcv, signal_history), use_container_width=True)

    st.markdown("**Performance: strategy vs. buy & hold**")
    _render_backtest_metrics(evaluation)

    st.info(
        "**This is a simplified historical simulation, not a guarantee.** "
        "No transaction costs or slippage are modeled, positions are "
        "all-in/all-out with no partial sizing or leverage, and capital is "
        "entirely hypothetical. A good historical backtest does not predict "
        "future performance — markets change, and this simple rule-based "
        "engine has no way to know that."
    )


def main() -> None:
    """Render the Streamlit page and wire up the sidebar controls."""
    st.set_page_config(page_title="stock-technical-signal-analyzer", page_icon="📈")

    st.title("📈 Stock Technical Signal Analyzer")
    st.write(
        "Enter a stock ticker to see its technical indicators and a simple "
        "rule-based trading signal."
    )
    st.warning(
        "**Educational use only — not financial advice.** This is a "
        "simplified, rule-based technical signal. It can be wrong, and "
        "should never be the sole basis for a real investing or trading "
        "decision."
    )

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker symbol", value="AAPL")
        period = st.selectbox(
            "Period", PERIOD_OPTIONS, index=PERIOD_OPTIONS.index("6mo")
        )
        run = st.button("Analyze", type="primary")

    live_tab, backtest_tab = st.tabs(["Live Signal", "Backtest"])

    with live_tab:
        if not run:
            st.caption("👈 Set your options in the sidebar and click **Analyze**.")
        elif not ticker.strip():
            st.warning("Please enter a ticker symbol to get started (e.g. AAPL).")
        else:
            try:
                run_analysis(ticker, period)
            except Exception:  # noqa: BLE001 - last-resort guard so the app never crashes
                st.error(
                    "Something went wrong while analyzing this ticker. Please "
                    "try again or pick a different ticker/period."
                )

    with backtest_tab:
        st.caption(
            "Simulates how this signal engine would have performed "
            "historically on the ticker above, versus simply buying and "
            "holding for the same period."
        )
        backtest_period = st.selectbox(
            "Backtest period",
            BACKTEST_PERIOD_OPTIONS,
            index=BACKTEST_PERIOD_OPTIONS.index("1y"),
            key="backtest_period",
        )
        run_backtest_clicked = st.button(
            "Run Backtest", type="primary", key="run_backtest_button"
        )

        if not run_backtest_clicked:
            st.caption("Pick a backtest period above and click **Run Backtest**.")
        elif not ticker.strip():
            st.warning(
                "Please enter a ticker symbol in the sidebar to get started "
                "(e.g. AAPL)."
            )
        else:
            try:
                run_backtest(ticker, backtest_period)
            except Exception:  # noqa: BLE001 - last-resort guard so the app never crashes
                st.error(
                    "Something went wrong while backtesting this ticker. "
                    "Please try again or pick a different ticker/period."
                )


if __name__ == "__main__":
    main()
