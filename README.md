# Institutional Quant Terminal

**A Streamlit research terminal for Indian equities, built around one hard question: after the 2025 Tata Motors demerger, does a machine-learning overlay beat buy-and-hold?**

It combines 23 research notebooks, a classical ML ensemble, a Monte-Carlo-dropout LSTM, a GARCH volatility gate and an LLM "robo-advisor" into one app. The research is written up in [`Step_by_Step_Recreation_Guide.md`](Step_by_Step_Recreation_Guide.md) and the two PDF reports in this repo.

> **Not financial advice.** This is a research and learning project. Nothing here is a recommendation to buy or sell anything.

## The story

In January 2025 Tata Motors split into two listed companies, **TMCV.NS** (commercial vehicles) and **TMPV.NS** (passenger vehicles). The old ticker was delisted overnight, so every data pipeline had to be rebuilt around five new series: the two demerged entities, Maruti Suzuki as a peer, NIFTY 50 and NIFTY Auto as benchmarks.

From there the project follows the research arc, one notebook at a time:

| Chapter | Notebooks | What happened |
|---|---|---|
| Get the data | 01-02 | 5 years of daily OHLCV, forward-fill and interpolation, five market regimes tagged (Pre-COVID, COVID crash, recovery, post-COVID, Oct 2024 shock) |
| Engineer features | 03-04 | Technical and statistical indicators |
| Understand the market | 05-07 | EDA, VADER vs TextBlob on financial headlines (VADER won), clustering market phases |
| Build models | 08-10 | Baseline comparison, iterative feature selection, hyperparameter tuning |
| Forecast and test | 11-13 | Prophet forecasts, backtests with 0.1% transaction costs, final investment thesis |
| Go institutional | 14-19 | LSTM with MC-dropout, LangGraph robo-advisor, transfer learning, HMM regime router, multi-asset correlation, RL position manager, intraday microstructure |

## What the research found

- **The edge is small, and that is the honest result.** The best classifier (XGBoost) reaches about 55% directional accuracy. In an efficient market any consistent edge above 51% matters, but it is not a money machine.
- **Fewer features beat more.** Iterative selection cut 30+ features to an optimal 10-15. Volume and rolling volatility carried the most SHAP weight.
- **ML helps risk more than return.** After costs, buy-and-hold is hard to beat. The ML overlay's value is lower drawdowns and better risk-adjusted metrics (Sharpe, Sortino).
- **Forecast bands widen fast.** Prophet changepoints line up with real events (COVID, the EV announcements, Oct 2024), and confidence intervals blow out beyond about 15 days. It is trend guidance, not a trading signal.

## The app

Six pages, launched from `streamlit_app.py`:

| Page | Purpose |
|---|---|
| Home | Landing page and orientation |
| Dashboard | Market overview across a configurable stock universe |
| Quick Analysis | Per-stock technicals, metrics and price charts |
| Deep Report | LLM robo-advisor that synthesises news, signals and risk into a written report (PDF export) |
| Quant Engine | Model predictions and the quant strategy |
| EDA Lab | The exploratory analysis from the notebooks, interactive |

## How it works

```
yfinance + news scrapers ──▶ features ──▶ XGBoost / LightGBM / LSTM (MC-dropout)
                                              │
                      GARCH volatility gate ◀─┘
                                              ▼
                  LangGraph agents + LLM reasoner ──▶ Streamlit pages + PDF report
```

- **Models:** XGBoost, LightGBM, a PyTorch LSTM with Monte-Carlo-dropout uncertainty, Prophet, and an HMM regime router (`models/`, `modules/ml/`)
- **Risk:** GARCH conditional-volatility forecast used as a gate (`agentic_backend.py`)
- **Agents:** LangGraph orchestration, async news scraping and an LLM reasoner through OpenRouter, with Groq or a local LM Studio server as alternatives
- **Caching:** API-call logging and a TTL cache keep LLM and data costs down (`modules/cache_manager.py`)

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

AI features need an API key in a `.env` file:

```env
OPENROUTER_API_KEY=...
# optional
GROQ_API_KEY=...
LM_STUDIO_URL=http://localhost:1234
```

Without a key the app still runs. Only the LLM-powered pages are disabled.

## Repository map

```
streamlit_app.py            app entry point
agentic_backend.py          LSTM, GARCH, scraper and LangGraph engine
config.py                   settings and env vars
modules/ui/                 one file per Streamlit page
modules/ml/                 features, ensemble, sentiment, deep learning
modules/data/               data manager, scrapers, portfolio
notebooks/                  23 research notebooks (01-19)
models/                     trained model artifacts
Step_by_Step_Recreation_Guide.md   full written methodology
```

## Limitations

Transaction costs are modelled at 0.1% per trade, which is lower than real-world brokerage, impact cost and slippage. The post-demerger history is short, so the two new tickers have limited training data. Treat all outputs as research, not signals.

---

Built by [Tuhin Bhattacharya](https://github.com/Karma1028).
