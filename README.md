# Stock-Technical-Signal-Analyzer

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

The app has two tabs.

In the sidebar you pick a ticker symbol (like `AAPL`) and a time period, then click **Analyze** to see the **Live Signal** tab:

- A signal badge — **BUY**, **SELL**, or **HOLD** — with the latest price and RSI value next to it
- An expandable "Why this signal?" section that explains, in plain words, exactly which rules voted which way
- A candlestick chart of the price, with moving averages and Bollinger Bands drawn on top
- An RSI chart, with the overbought/oversold zones shaded
- A MACD chart, showing the momentum lines and histogram

The **Backtest** tab picks a separate historical window and, on clicking **Run Backtest**, shows how the signal engine would have performed on that ticker historically — see "Backtesting" below.

A warning is always shown on the page reminding you this is for learning, not for making real investment decisions — see the disclaimer below.

## How the signal works

The app looks at three classic technical indicators:

- **RSI** — checks whether the price has moved unusually far, unusually fast. This app votes BUY below 45 and SELL above 55. That's a more sensitive band than the classic textbook 30/70 levels — chosen deliberately so the rule actually fires on real, actively-traded stocks, at the cost of being a less extreme reading than the textbook definition.
- **MACD** — checks whether the price's momentum is shifting up or down.
- **Bollinger Bands** — checks whether the price has pushed to an unusual extreme compared to its recent range. This app votes BUY if the close sits in the bottom 30% of its position between the lower and upper band, SELL if it sits in the top 30%.

Each indicator casts one simple vote: BUY, SELL, or no vote. Whichever side — BUY or SELL — has more votes wins, as long as it has at least one; a tie (including no votes either way) is a HOLD. There is no machine learning and no hidden scoring; every part of the decision is shown to you.

This app is a standalone tool, but it's part of a small family of related projects: [`Python-Risk-Management-Library`](https://github.com/bulut396/Python-Risk-Management-Library) for portfolio risk metrics, and [`stock-news-signal-analyzer`](https://github.com/bulut396/stock-news-signal-analyzer) for news-based sentiment. They look at the same kind of question — is this stock looking healthy? — from different angles. None of them depend on each other; each works completely on its own.

## Backtesting

The **Backtest** tab answers a different question than the live signal: not "what does the signal say today?" but "if I had followed this signal engine's calls historically, how would that have gone?"

It works by replaying the signal engine one day at a time across the chosen historical window, using only the price data that would have actually been available on each day (never peeking ahead), then simulating a simple trading strategy on those historical signals:

- Start with $10,000 of hypothetical capital.
- On a BUY signal, if not already holding the stock, put 100% of the available cash into it.
- On a SELL signal, if currently holding, sell the entire position back to cash.
- On HOLD, do nothing.

The result is compared against simply buying the stock on day one and holding it for the same period — a backtest result means nothing without that comparison. Both are scored with the same set of metrics: total return, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, historical Value at Risk, and (for the strategy) the number of trades and win rate.

**Be honest with yourself about the limitations** — this is a deliberately simplified v1 simulation:

- No transaction costs or trading fees are modeled.
- No slippage (the assumption that you always trade exactly at the closing price) is modeled.
- Positions are all-in/all-out — no partial sizing, no leverage, no scaling in or out.
- Capital is entirely hypothetical; nothing here is a real trade or a real account.

The performance metrics are computed using [`Python-Risk-Management-Library`](https://github.com/bulut396/Python-Risk-Management-Library), a sibling project from the same author — none of that math (Sharpe, Sortino, drawdown, Calmar, VaR) is reimplemented here.

A good-looking historical backtest is not a promise about the future. Markets change, and this simple rule-based engine has no way to know that — it is replaying a fixed set of rules against the past, not learning or adapting.

## Disclaimer

This app is for educational purposes only. It is **not financial advice**. The signal is a simplified, rule-based calculation and can easily be wrong. Do not make real investing or trading decisions based on this app alone.

## Technical Notes

All indicators (SMA, EMA, RSI with Wilder's smoothing, MACD, Bollinger Bands, and a volume-trend ratio) are implemented directly with pandas/numpy from their standard formulas — no pre-built technical-analysis library is used. Price data comes from `yfinance`, with `auto_adjust=True`. Backtest performance metrics come from [`Python-Risk-Management-Library`](https://github.com/bulut396/Python-Risk-Management-Library), installed as a regular dependency via `requirements.txt`.

## License

Released under the [MIT License](LICENSE).
