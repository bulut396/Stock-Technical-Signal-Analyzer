# stock-technical-signal-analyzer

This app looks at a stock's recent price history and gives you a simple BUY/SELL/HOLD signal, based on classic technical indicators.

## What you need

- Python installed (3.10 or newer).
- That's it. No API key, no account, no payment — the price data is free and requires no sign-up at all.

## How to run it

```bash
git clone https://github.com/bulut396/stock-technical-signal-analyzer.git
cd stock-technical-signal-analyzer
python -m venv venv
venv\Scripts\activate        # or: source venv/bin/activate   on Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## What you'll see

In the sidebar you pick a ticker symbol (like `AAPL`) and a time period, then click **Analyze**. The app shows:

- A signal badge — **BUY**, **SELL**, or **HOLD** — with the latest price and RSI value next to it
- An expandable "Why this signal?" section that explains, in plain words, exactly which rules voted which way
- A candlestick chart of the price, with moving averages and Bollinger Bands drawn on top
- An RSI chart, with the overbought/oversold zones shaded
- A MACD chart, showing the momentum lines and histogram

A warning is always shown on the page reminding you this is for learning, not for making real investment decisions — see the disclaimer below.

## How the signal works

The app looks at three classic technical indicators:

- **RSI** — checks whether the price has moved unusually far, unusually fast. This app votes BUY below 40 and SELL above 60. That's a more sensitive band than the classic textbook 30/70 levels — chosen deliberately so the rule actually fires on real, actively-traded stocks, at the cost of being a less extreme reading than the textbook definition.
- **MACD** — checks whether the price's momentum is shifting up or down.
- **Bollinger Bands** — checks whether the price has pushed to an unusual extreme compared to its recent range.

Each indicator casts one simple vote: BUY, SELL, or no vote. If at least two indicators vote BUY and none vote SELL, the app calls it a BUY. If at least two vote SELL and none vote BUY, it calls it a SELL. Anything else — including mixed votes — is a HOLD. There is no machine learning and no hidden scoring; every part of the decision is shown to you.

This app is a standalone tool, but it's part of a small family of related projects: [`Python-Risk-Management-Library`](https://github.com/bulut396/Python-Risk-Management-Library) for portfolio risk metrics, and [`stock-news-signal-analyzer`](https://github.com/bulut396/stock-news-signal-analyzer) for news-based sentiment. They look at the same kind of question — is this stock looking healthy? — from different angles. None of them depend on each other; each works completely on its own.

## Disclaimer

This app is for educational purposes only. It is **not financial advice**. The signal is a simplified, rule-based calculation and can easily be wrong. Do not make real investing or trading decisions based on this app alone.

## Technical Notes

All indicators (SMA, EMA, RSI with Wilder's smoothing, MACD, Bollinger Bands, and a volume-trend ratio) are implemented directly with pandas/numpy from their standard formulas — no pre-built technical-analysis library is used. Price data comes from `yfinance`, with `auto_adjust=True`.

## License

Released under the [MIT License](LICENSE).
