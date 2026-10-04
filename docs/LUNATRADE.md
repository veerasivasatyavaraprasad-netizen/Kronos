# 🌙 LunaTrade

LunaTrade is a multi-agent trading system built on this repository's **Kronos** forecasting model.
300 lightweight analyst workers feed a lead trading brain. A devil's-advocate veto layer, an independent
risk engine, a portfolio brain and a single execution gateway sit between the analysts and the
exchange. A council of LLMs advises, but it never places an order.

> **Research software, not financial advice.** Start in PAPER mode and earn every promotion with evidence.
> No configuration makes losses impossible.

---

## 1. Architecture

```
                         ┌─────────────────────────┐
                         │       HUMAN OWNER       │  dashboard · REST API · voice · Telegram
                         │   YES / NO / EMERGENCY  │
                         └────────────┬────────────┘
                                      ▼
                    ┌─────────────────────────────────┐
                    │        LEAD TRADING BRAIN       │  family scores → conviction (0-100)
                    │  + LLM council review (bounded) │  +5 / −15 conviction at most
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  DEVIL'S ADVOCATE (mandatory)   │  bull case vs bear case → VETO / size cut
                    └───────────────┬─────────────────┘
                                    ▼
          ┌──────────────┐   ┌─────────────┐   ┌───────────────────┐   ┌──────────────────┐
          │ RISK ENGINE  │ → │ PORTFOLIO   │ → │ (HUMAN APPROVAL)  │ → │ EXECUTION        │ → Binance / Alpaca
          │ independent  │   │ BRAIN       │   │ APPROVAL mode     │   │ GATEWAY (only    │
          └──────────────┘   └─────────────┘   └───────────────────┘   │ order placer)    │
                 ▲                                                     └──────────────────┘
          KILL SWITCH (outside the AI layer)       SUPERVISOR 24/7      MEMORY (PostgreSQL)

       ┌──────────────────────────── SPECIALIST NETWORK: 300 workers ────────────────────────────┐
       │ scanners 35 · technical 60 · momentum/vol 25 · order book 20 · news 20 · social 20      │
       │ on-chain 25 · whales 20 · macro 15 · arbitrage 15 · regime 10 · forecast (Kronos) 5     │
       │ risk analysis 10 · strategy evaluation 10 · portfolio analysis 5 · adversarial 5        │
       └─────────────────────────────────────────────────────────────────────────────────────────┘
```

**The rule:** `DATA → ANALYSIS → SIGNAL → CONVICTION → RISK → PORTFOLIO → EXECUTION`. An "LLM says BUY →
exchange order" path does not exist: only `lunatrade/execution/gateway.py` holds broker objects, and a BUY
without a risk-engine authorization is rejected.

### The 300 workers (`python -m lunatrade agents --full`)

The workers are small deterministic statistical analysts, not 300 LLM conversations. Indicators are
memoised per candle, so a full cycle over 15 symbols takes well under a second. Each worker emits a
structured `Signal` (symbol, direction, confidence, setup, expected move, horizon, evidence,
invalidations, context). A signal is evidence and never becomes an order by itself.

| Family | # | What it covers |
|---|---|---|
| scanner | 35 | abnormal volume, relative strength vs BTC, cross-sectional momentum/reversal, new highs/lows, correlation, BTC-dominance proxy, sector rotation, liquidity, funding, open interest, long/short ratio, liquidation flushes |
| technical | 60 | EMA crosses, VWAP trend, ADX, Donchian breakouts, HH/HL structure, MACD, Bollinger/RSI/VWAP/z-score/Keltner/stochastic reversion, S/R bounce, failed breakouts, sweep & reclaim, trend pullbacks, inside bars, candlestick + chart patterns |
| momentum / volatility | 13 + 12 | volume expansion, acceleration, ROC, breakout confirmation, OBV; ATR expansion, squeezes, ATR regime, realized vs long-run vol, Parkinson vol |
| microstructure | 20 | book imbalance, spread z-score, depth asymmetry, taker flow, large trades, microprice, book pressure, liquidity vacuum, VPIN-style toxicity |
| news | 20 | one specialist per event family (ETF flows, regulation, hacks, listings, delistings, unlocks, central banks, macro data, geopolitics, …) plus burst velocity, credible-vs-noise divergence, composite |
| social | 20 | per-community sentiment, mention velocity, organic vs coordinated, bot detection, panic, capitulation, euphoria, Fear & Greed, social/price divergence |
| on-chain | 25 | stablecoin supply/dry powder, chain TVLs, mempool congestion, fee spikes, hashrate, miner stress, bridge and staking flows |
| whale | 20 | exchange in/outflows, net flows, stablecoin mints/burns, stablecoins to exchanges, dormant wallets, accumulation/distribution, deposit clusters, smart money, unlocks, custody noise filter |
| macro | 15 | SPY/QQQ, dollar, yields, gold, VIX level + spike, credit, oil, risk-on/off composites, FOMC/CPI/NFP/PCE proximity |
| arbitrage | 15 | cross-exchange spreads, Coinbase premium, triangular gaps, spot-perp basis, stablecoin depeg, funding carry (signals only) |
| regime | 10 | trend vs range, volatility regime, bull/bear, panic, liquidity, event-driven, Hurst |
| forecast | 5 | **Kronos-small, Kronos-mini** (this repo's model; needs torch), regression trend, Holt smoothing, kNN pattern analogs |
| risk / strategy eval / portfolio / adversarial | 10/10/5/5 | VaR/CVaR, beta, correlation clusters, stop feasibility…; per-strategy hit rates & degradation; concentration/cluster/sector/beta exposure; counter-signals, event risk, crowding, slippage, stop checks |

Workers whose data source is missing (for example no `WHALE_ALERT_API_KEY`) report **OFFLINE**. Their
family then drops out of the score; nothing is faked.

### Decision cycle (`lunatrade/engine.py`)

1. Protective exits come first: stop-loss, take-profit, break-even at +1R, ATR trailing stop, time stop.
2. The account is marked to market, then the kill-switch limits are checked (daily/weekly loss, drawdown).
3. Phase-1 workers run, then the **regime engine** runs, then phase-2 workers (strategy evaluation, portfolio, adversarial).
4. **Strategy memory** scores past signals against what actually happened and the meta-learner updates the weights.
5. The **Lead Brain** groups signals by family. Each signal is weighted by its strategy's learned
   performance and the regime multiplier. Each family is scored in [−100, 100], then the families are
   combined with category weights and coverage. The trade needs ≥3 independent families agreeing, and
   strong dissent costs conviction. The result is **conviction**, which is *not* a probability. A
   calibrated win probability is attached once ≥20 similar trades exist.
6. Optional **LLM council** review, bounded to at most +5 / −15 conviction.
7. **Devil's advocate:** "what would make this trade wrong?" It returns VETO or a size reduction.
8. **Risk engine:** independent checks and risk-budget sizing (1% of equity at the stop by default).
9. **Portfolio brain:** caps correlated clusters and beta-weighted exposure.
10. **Human approval** (APPROVAL mode), then the **execution gateway**: validation, order state machine,
    and reconciliation of unknown states.

Every step is published on the event bus and stored in the database, so `GET /api/explain/{proposal_id}`
shows why a trade happened: category scores, devil's-advocate report, risk checks, orders and outcome.

---

## 2. Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the Kronos workers (optional)
pip install -r requirements-lunatrade.txt
cp .env.example .env        # then edit .env (see below)
python -m lunatrade check   # shows which keys/services are configured - never prints secrets
python -m lunatrade run     # engine + dashboard on http://localhost:8800
```

### API keys: they go in `.env` only

`.env` is git-ignored. Keys never belong in source code, `lunatrade.yaml`, chat messages or commits.

| Variable | Used for |
|---|---|
| `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_SECRET_KEY` | APPROVAL/LIVE orders while `brokers.binance.environment: testnet` (free keys: <https://testnet.binance.vision>) |
| `BINANCE_API_KEY` / `BINANCE_SECRET_KEY` | real account (only with `environment: live` **and** `LUNATRADE_ALLOW_LIVE=yes`) |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | US stocks; `PK…` keys are paper keys |
| `VOICE_API_KEY` + `VOICE_PROVIDER` | speech-to-text provider (`assemblyai`, `deepgram` or `openai`) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OLLAMA_HOST` | LLM council members |
| `LUNATRADE_API_TOKEN`, `LUNATRADE_CONTROL_PIN` | dashboard/API auth and the PIN for sensitive actions |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | owner pings and `/yes <id>` approvals |

**Binance key hygiene (least privilege):** enable only *Spot & Margin Trading*. Keep **withdrawals
disabled** and restrict the key to your server's IP. If a key was ever pasted somewhere public (a chat,
an issue, a screenshot), delete it in Binance → API Management and create a new one.

---

## 3. Modes and the deployment path

| Mode | Market data | Orders |
|---|---|---|
| `RESEARCH` | real | none |
| `PAPER` | real | simulated fills (fees + slippage) |
| `SHADOW` | real | sent to Binance `/api/v3/order/test` (validated, **not executed**); paper ledger |
| `APPROVAL` | real | every order waits for your YES (dashboard / Telegram / voice); testnet by default |
| `LIVE` | real | automatic inside risk limits; Binance live needs `LUNATRADE_ALLOW_LIVE=yes` |

Recommended path: `unit tests → backtest → walk-forward → PAPER → SHADOW → APPROVAL (testnet) → APPROVAL (live, small) → LIVE`.
Promotion is one step at a time and gated by evidence (`promotion:` in the config: ≥50 trades, ≥14 days,
profit factor ≥1.2, expectancy ≥0.05R, drawdown ≤10%). `GET /api/promotion/LIVE` shows the gates, and a
human confirms the switch with the PIN. Demotion is always allowed.

The defaults follow the original plan's recommendations: **spot only** (futures data is read-only
signal input), **auto-discovered liquid coins** with strict volume/spread filters, **paper money** first.

---

## 4. Safety layers

* **Kill switch** (`risk/kill_switch.py`) runs outside the AI layer.
  * Hard trips persist across restarts and need a human reset with the PIN: max daily loss, max weekly
    loss, max drawdown, abnormal slippage, repeated order failures, failed reconciliation, emergency close.
  * Soft halts clear themselves after N healthy supervisor checks: stale market data, exchange/API
    unhealthy, database down.
  * A file named `outputs/lunatrade/STOP` also blocks new orders.
  * No worker, brain or LLM can reset the kill switch. Exits and stop-losses keep working while it is on.
* **Order state machine:** `CREATED → VALIDATED → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED → POSITION_UPDATED`.
  A timeout becomes `UNKNOWN`, and the gateway queries the exchange by `newClientOrderId` instead of
  re-sending, so a lost response can't double a position.
* **Gateway validation:** risk approval, kill switch, duplicates, one working order per symbol, step/tick
  precision, min qty/notional, balance, market hours. It never sells more than LunaTrade itself bought,
  so your existing coins are never touched.
* **Supervisor** (every 15 s): data freshness, WebSocket, brokers, DB, workers, LLMs, CPU/memory, cycle
  latency. A dead feed stops new trades, triggers a reconnect, and trading resumes only after healthy checks.
* **Meta-learning limits:** it may move strategy weights (0.25–2.0, ≤10% per update) and the conviction
  threshold (55–80). It can never change code or risk limits.
* **Accuracy target:** robust risk-adjusted expectancy (Sharpe, Sortino, profit factor, expectancy, avg R,
  tail loss, drawdown), not an inflated win rate.

---

## 5. Operating it

```bash
python -m lunatrade run                 # 24/7 runtime: WebSockets, feeds, engine, supervisor, API
python -m lunatrade cycle               # one cycle on live data, printed
python -m lunatrade backtest --symbols BTCUSDT,ETHUSDT,SOLUSDT --bars 2000 --walk-forward --monte-carlo --stress
python -m lunatrade backtest --synthetic          # offline
python -m lunatrade universe            # liquid symbols right now
python -m lunatrade db sql              # PostgreSQL DDL (lunatrade/memory/migrations/0001_initial.sql)
make luna-test                          # test suite (tests/lt)
docker compose -f docker-compose.lunatrade.yml up -d --build   # postgres + redis + lunatrade
```

**Dashboard** (`http://<host>:8800`): mode, regime, health, equity curve, KPIs, active signals with
conviction bars, positions with close buttons, pending approvals (YES/NO), recent proposals with the
devil's-advocate verdict, the agent network (online/degraded/offline per family), LLM council members,
strategy weights, news intelligence, the event bus, and **Pause / Resume / Emergency close** buttons.
There is also a voice/text box with a microphone button.

**Voice** (no trading access): `voice → speech-to-text → parser → PIN check for sensitive intents → control API`.
Try "What's happening with BTC?", "Stop all new trades", "Close ETH" (PIN) or "Emergency close" (PIN).

**Telegram:** `/status /positions /pnl /stop /resume <pin> /yes <id> /no <id>`. Messages are accepted only
from `TELEGRAM_CHAT_ID`.

**REST API** (bearer token; OpenAPI at `/api/docs`): `/api/status`, `/api/signals`, `/api/assessments`, `/api/regime`,
`/api/portfolio`, `/api/orders`, `/api/trades`, `/api/proposals`, `/api/explain/{id}`, `/api/agents`,
`/api/strategies`, `/api/news`, `/api/events`, `/api/alerts`, `/api/equity`, `/api/approvals[/{id}/approve|reject]`,
`/api/control/{pause,resume,emergency-close,close/{symbol},kill-switch/reset,mode}`, `/api/voice/{command,audio,speak}`,
`/health`, `/metrics` (Prometheus).

**Database** (PostgreSQL in production, SQLite by default): `agents, agent_events, market_data, market_snapshots,
signals, trade_proposals, risk_decisions, approvals, orders, fills, positions, portfolio_snapshots, news_events,
sentiment_events, whale_events, strategy_performance, trade_outcomes, system_alerts, agent_health,
model_versions, backtests, knowledge`.

---

## 6. Backtesting without look-ahead

`backtest/engine.py` runs the **same TradingEngine** with a simulated clock:

* The `MarketView` only shows candles whose close time is ≤ the decision time.
* Feeds only show items received before the decision time.
* Fills happen at the decision price plus slippage, and stops fill at the stop level (worse on gaps).
* A test (`test_no_lookahead_signals_identical_with_or_without_future_data`) proves that adding future
  data changes nothing.

`backtest/robustness.py` adds:

* **Walk-forward:** weights learned in-sample, frozen out-of-sample.
* **Monte Carlo:** trade bootstrap → equity/drawdown percentiles.
* **Stress scenarios:** bull, bear, sideways, flash crash, high volatility, low liquidity, large spread,
  API outage, partial fills, slippage shock.

---

## 7. Code map

```
lunatrade/
  config/            default.yaml + loader (env overrides, secrets only from env)
  core/              types (Signal, TradeProposal, ...), event bus (+Redis/Kafka mirrors), clocks
  data/              market store (time-cut views), Binance REST/WebSocket, universe discovery, feeds, synthetic data
  indicators/        numpy TA library
  intel/             news pipeline, sentiment lexicon, social analysis, whale interpretation, asset dictionary
  workers/           300 workers in 12 modules + registry + strategy groups
  brains/            regime engine, lead brain, devil's advocate
  llm/               providers (Claude, OpenAI-compatible, Gemini, Ollama, rules) + council
  risk/              risk engine, kill switch, portfolio brain
  execution/         order state machine, gateway, position book, brokers (paper, Binance spot, Alpaca)
  memory/            schema, repository, strategy performance + meta-learning, event memory, migrations
  backtest/          engine, metrics, walk-forward, Monte Carlo, stress
  control/           modes/promotion, approvals, notifier/Telegram, control facade
  supervisor/        24/7 health monitor
  voice/             command parser, voice agent (STT/TTS providers)
  api/               FastAPI app + dashboard
  engine.py          one decision cycle (shared by live + backtest)
  orchestrator.py    live runtime
  cli.py             python -m lunatrade ...
```

### What carries over from Kronos

* `model/` + `automation/forecaster.py` power the Kronos forecast workers.
* `automation/notify.py` handles owner alerts.
* `automation/env.py` loads `.env`.
* The existing `automation/live.py` bot still works unchanged.
