"""stock-technical-signal-analyzer — technical indicators and a rule-based signal.

Fetches OHLCV price history via yfinance, computes RSI/MACD/Bollinger Bands
from scratch, combines them into a transparent BUY/SELL/HOLD signal, and
visualizes everything on an interactive candlestick chart. Run with::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from charts import plot_candlestick_with_indicators, plot_macd, plot_rsi
from indicators import bollinger_bands, macd, rsi, sma
from price_fetcher import fetch_ohlcv
from signal_engine import generate_signal

PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y"]

_SIGNAL_RENDERERS = {
    "BUY": st.success,
    "SELL": st.error,
    "HOLD": st.info,
}


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
            f"(RSI is {signal_result['rsi_value']:.1f}; below 40 votes BUY/"
            "oversold, above 60 votes SELL/overbought)."
        )
        st.markdown(
            f"- **MACD rule**: {votes['macd']} "
            "(votes BUY if the MACD histogram just flipped from negative to "
            "positive, SELL if it just flipped from positive to negative)."
        )
        st.markdown(
            f"- **Bollinger rule**: {votes['bollinger']} "
            "(votes BUY if the latest close sits in the bottom 20% of its "
            "position between the lower and upper band, SELL if it sits in "
            "the top 20%)."
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
        "fast — the shaded zones mark the classic overbought (above 70) and "
        "oversold (below 30) thresholds."
    )
    st.plotly_chart(plot_rsi(rsi_series), use_container_width=True)

    st.markdown(
        "MACD tracks momentum — when the blue line crosses the orange "
        "signal line, momentum is shifting, and the bars show how strong "
        "that shift is."
    )
    st.plotly_chart(plot_macd(macd_result), use_container_width=True)


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

    if not run:
        st.caption("👈 Set your options in the sidebar and click **Analyze**.")
        return

    if not ticker.strip():
        st.warning("Please enter a ticker symbol to get started (e.g. AAPL).")
        return

    try:
        run_analysis(ticker, period)
    except Exception:  # noqa: BLE001 - last-resort guard so the app never crashes
        st.error(
            "Something went wrong while analyzing this ticker. Please try "
            "again or pick a different ticker/period."
        )


if __name__ == "__main__":
    main()
