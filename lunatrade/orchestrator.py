"""
Live runtime: wires data feeds, the trading engine, supervisor, approvals, Telegram commands and the API,
then runs one decision cycle right after every candle closes, 24/7.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import secrets
import threading
import time
from logging.handlers import RotatingFileHandler

from lunatrade.config import Config, live_trading_allowed, load_config, secret
from lunatrade.control.facade import Control
from lunatrade.control.notifier import Notifier, TelegramCommands
from lunatrade.core.bus import EventBus, build_mirrors
from lunatrade.core.events import Topic
from lunatrade.core.types import Mode
from lunatrade.data.binance_public import BinancePublic, symbol_filters
from lunatrade.data.binance_ws import BinanceStreams
from lunatrade.data.feeds import Collectors, FeedHub, FeedScheduler
from lunatrade.data.market_store import MarketStore
from lunatrade.data.universe import discover
from lunatrade.engine import TradingEngine
from lunatrade.execution.brokers.paper import PaperBroker
from lunatrade.execution.positions import PositionBook
from lunatrade.intel.news_pipeline import NewsPipeline
from lunatrade.llm.council import LLMCouncil
from lunatrade.memory.repository import Repository
from lunatrade.supervisor.health import Supervisor
from lunatrade.voice.service import VoiceAgent

log = logging.getLogger("lunatrade")


def setup_logging(out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("lunatrade")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(out_dir / "lunatrade.log", maxBytes=10_000_000, backupCount=10, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def build_brokers(cfg: Config, mode: Mode, filters_provider=None) -> tuple[dict, str, str]:
    """Return (brokers, crypto_broker_name, stock_broker_name) for a mode."""
    p = cfg.section("paper")
    paper = PaperBroker(float(p.get("starting_equity", 10_000)), float(p.get("fee_bps", 10)), float(p.get("slippage_bps", 5)),
                        float(p.get("partial_fill_probability", 0)), filters_provider=filters_provider)
    brokers, crypto, stock = {"paper": paper}, "paper", "paper"
    env = cfg.get("brokers.binance.environment", "testnet")
    if env != "paper" and mode in (Mode.SHADOW, Mode.APPROVAL, Mode.LIVE):
        from lunatrade.execution.brokers.binance_spot import BinanceSpotBroker

        try:
            if mode == Mode.LIVE and env == "live" and not live_trading_allowed():
                raise RuntimeError("LIVE + binance live requires LUNATRADE_ALLOW_LIVE=yes")
            brokers["binance"] = BinanceSpotBroker(env, validate_only=(mode == Mode.SHADOW))
            crypto = "binance"
        except Exception as e:
            if mode == Mode.SHADOW:
                log.warning("SHADOW mode without Binance validation (%s) - paper fills only", e)
            else:
                raise
    if cfg.get("stocks.enabled") and mode in (Mode.APPROVAL, Mode.LIVE):
        from lunatrade.execution.brokers.alpaca import AlpacaBroker

        brokers["alpaca"] = AlpacaBroker(cfg.get("brokers.alpaca.environment", "paper"))
        stock = "alpaca"
    return brokers, crypto, stock


class Runtime:
    def __init__(self, cfg: Config | None = None, start_threads: bool = True, symbols: list[str] | None = None,
                 public: BinancePublic | None = None):
        self.cfg = cfg or load_config()
        setup_logging(self.cfg.output_dir)
        self.mode = Mode(self.cfg.get("mode", "PAPER"))
        if self.mode == Mode.LIVE and self.cfg.get("brokers.binance.environment") == "live" and not live_trading_allowed():
            raise SystemExit("Refusing to start: mode LIVE on Binance live needs LUNATRADE_ALLOW_LIVE=yes")
        self.repo = Repository(self.cfg.database_url)
        self.bus = EventBus()
        for m in build_mirrors(self.cfg.get("bus.redis_url"), self.cfg.get("bus.kafka_bootstrap")):
            self.bus.add_mirror(m)
        self.notifier = Notifier()
        self.council = LLMCouncil.from_config(self.cfg)
        self.public = public or BinancePublic()
        self.store = MarketStore(self.cfg.get("interval", "15m"), int(self.cfg.get("history_bars", 500)))
        self.feeds = FeedHub(NewsPipeline(llm_council=self.council))
        self.control_pin = secret("LUNATRADE_CONTROL_PIN")
        self.api_token = secret("LUNATRADE_API_TOKEN") or self._token_file()

        self.symbols = symbols or discover(self.public, self.cfg.section("universe"))
        self.stock_symbols = list(self.cfg.get("stocks.symbols", [])) if self.cfg.get("stocks.enabled") else []
        self._filters: dict = {}
        brokers, self.crypto_broker, self.stock_broker = build_brokers(self.cfg, self.mode, self._filters.get)
        book_path = self.cfg.output_dir / f"book_{self.mode.value.lower()}.json"
        self.book = PositionBook(book_path, float(self.cfg.get("paper.starting_equity", 10_000)))
        self.engine = TradingEngine(self.cfg, self.store, self.feeds, brokers, self.book, self.mode,
                                    self.symbols + self.stock_symbols, route=self.route, bus=self.bus, repo=self.repo,
                                    council=self.council, notifier=self.notifier, state_dir=self.cfg.output_dir)
        self.control = Control(self)
        self.voice = VoiceAgent(self.control, self.council, self.cfg.section("voice"))
        self.supervisor = Supervisor(self.engine, self.cfg.section("supervisor"), reconnect=self.reconnect)
        self.collectors = Collectors(self.feeds, self.cfg.section("feeds"), lambda: self.symbols)
        self.scheduler = FeedScheduler(self.collectors, self.cfg.section("feeds"))
        self.telegram = TelegramCommands(self.control.telegram)
        self.streams: BinanceStreams | None = None
        self._stop = threading.Event()
        self.started_at = dt.datetime.utcnow()
        self.repo.model_version("lunatrade", "1.0.0", {"mode": self.mode.value, "interval": self.store.interval,
                                                       "symbols": self.symbols, "workers": len(self.engine.workers)})
        if start_threads:
            self.bootstrap()

    def _token_file(self) -> str:
        p = self.cfg.output_dir / "api_token"
        if not p.exists():
            p.write_text(secrets.token_urlsafe(32))
            os.chmod(p, 0o600)
        return p.read_text().strip()

    def route(self, symbol: str) -> str:
        return self.stock_broker if symbol in self.stock_symbols else self.crypto_broker

    # ------------------------------------------------------------------ data
    def bootstrap(self) -> None:
        interval = self.store.interval
        n = int(self.cfg.get("history_bars", 500))
        try:
            info = self.public.exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] in self.symbols:
                    self._filters[s["symbol"]] = symbol_filters(s)
                    self.store.set_meta(s["symbol"], **self._filters[s["symbol"]])
        except Exception as e:
            log.warning("exchange info unavailable: %s", e)
        for sym in self.symbols:
            try:
                df = self.public.klines(sym, interval, n)
                self.store.load_frame(sym, df)
                self.repo.candles(sym, interval, df.tail(50))
            except Exception as e:
                log.error("bootstrap %s failed: %s", sym, e)
        if self.stock_symbols:
            self._bootstrap_stocks()
        if self.cfg.get("data.websocket", True):
            self.streams = BinanceStreams(self.store, self.symbols, interval, "live",
                                          int(self.cfg.get("data.depth_levels", 20)))
            self.streams.start()

    def _bootstrap_stocks(self) -> None:
        try:
            import yfinance as yf

            yf_int = {"15m": "15m", "5m": "5m", "1h": "60m", "1d": "1d", "30m": "30m"}.get(self.store.interval, "15m")
            for s in self.stock_symbols:
                df = yf.download(s, period="30d", interval=yf_int, progress=False, auto_adjust=False)
                if df is None or df.empty:
                    continue
                if hasattr(df.columns, "levels"):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index().rename(columns=str.lower)
                df = df.rename(columns={df.columns[0]: "open_time"})
                df["open_time"] = df["open_time"].dt.tz_localize(None) if df["open_time"].dt.tz is not None else df["open_time"]
                self.store.load_frame(s, df)
        except Exception as e:
            log.warning("stock bootstrap failed: %s", e)

    def refresh_market_data(self) -> None:
        """REST fallback + derivatives snapshots. Fills any candle the WebSocket missed."""
        interval = self.store.interval
        now_ms = int(time.time() * 1000)
        expected_last_open = (now_ms // self.store.interval_ms - 1) * self.store.interval_ms
        for sym in self.symbols:
            st = self.store.state(sym)
            if not len(st.open_time) or st.open_time[-1] < expected_last_open:
                try:
                    df = self.public.klines(sym, interval, 5)
                    for r in df.itertuples():
                        self.store.upsert_candle(sym, int(r.open_time), r._asdict(), received_ms=now_ms)
                except Exception as e:
                    log.warning("REST refresh %s failed: %s", sym, e)
            if self.cfg.get("data.derivatives_data", True):
                try:
                    d = self.public.derivatives_snapshot(sym)
                    if d:
                        self.store.add_derivatives(sym, d, now_ms)
                except Exception:
                    pass
            if not self.store.ws_connected:
                try:
                    bt = self.public.book_ticker(sym)
                    self.store.add_ticker(sym, float(bt["bidPrice"]), float(bt["askPrice"]), now_ms,
                                          float(bt["bidQty"]), float(bt["askQty"]))
                    dep = self.public.depth(sym, 20)
                    self.store.add_book(sym, dep["bids"], dep["asks"], now_ms)
                except Exception:
                    pass

    def reconnect(self, component: str) -> None:
        if component == "market_data":
            threading.Thread(target=self.refresh_market_data, daemon=True).start()
        elif component.startswith("broker:"):
            b = self.engine.gateway.brokers.get(component.split(":", 1)[1])
            if b is not None and hasattr(b, "sync_time"):
                try:
                    b.sync_time()
                    b.consecutive_errors = 0
                except Exception as e:
                    log.warning("broker reconnect failed: %s", e)

    def sync_live_cash(self) -> None:
        """In APPROVAL/LIVE the account's quote balance is the source of truth for cash."""
        b = self.engine.gateway.brokers.get(self.crypto_broker)
        if b is None or b.name == "paper" or getattr(b, "validate_only", False):
            return
        try:
            self.book.cash = float(b.free_balances().get("USDT", 0.0)) if hasattr(b, "free_balances") else self.book.cash
        except Exception as e:
            log.warning("cash sync failed: %s", e)

    # ------------------------------------------------------------------ control
    def switch_mode(self, mode: Mode, by: str) -> None:
        brokers, self.crypto_broker, self.stock_broker = build_brokers(self.cfg, mode, self._filters.get)
        self.mode = mode
        self.engine.mode = mode
        self.engine.gateway.mode = mode
        self.engine.gateway.brokers = brokers
        self.cfg.set("mode", mode.value)
        self.bus.emit(Topic.MODE_CHANGED, {"mode": mode.value, "by": by}, source="control")
        self.notifier.send(f"Mode changed to {mode.value} by {by}", "control")

    def execute_approved_now(self) -> None:
        view = self.store.view(dt.datetime.utcnow())
        prices = {s: view.price(s) for s in self.engine.symbols if len(view.candles(s))}
        self.engine.execute_approved(prices, dt.datetime.utcnow())

    # ------------------------------------------------------------------ loops
    def _supervisor_loop(self) -> None:
        every = float(self.cfg.get("supervisor.check_interval_seconds", 15))
        while not self._stop.is_set():
            try:
                self.supervisor.check()
                if self.cfg.get("approval.telegram_commands", True):
                    self.telegram.poll()
            except Exception:
                log.exception("supervisor loop error")
            self._stop.wait(every)

    def seconds_to_next_close(self, delay: float = 3.0) -> float:
        step = self.store.interval_ms / 1000
        return step - (time.time() % step) + delay

    def run_cycle(self) -> dict:
        self.refresh_market_data()
        self.sync_live_cash()
        res = self.engine.cycle(dt.datetime.utcnow())
        if self.engine.cycle_count % 4 == 0:
            self.engine.gateway.reconcile_open_orders()
            if self.crypto_broker != "paper":
                self.engine.gateway.reconcile_positions(self.crypto_broker)
        log.info("cycle %d: %d signals, %d proposals, equity %.2f, %.0fms", self.engine.cycle_count, res["signals"],
                 len(res["proposals"]), res["equity"], res["cycle_ms"])
        return res

    def run_forever(self, api: bool = True) -> None:
        threading.Thread(target=self._supervisor_loop, name="supervisor", daemon=True).start()
        self.scheduler.start()
        if api:
            self.start_api()
        self.notifier.send(f"Started in {self.mode.value} mode, {len(self.symbols)} symbols, "
                           f"{len(self.engine.workers)} workers, interval {self.store.interval}.", "info")
        log.info("API token file: %s", self.cfg.output_dir / "api_token")
        try:
            self.run_cycle()
            while not self._stop.is_set():
                wait = self.seconds_to_next_close()
                self._stop.wait(wait)
                if self._stop.is_set():
                    break
                try:
                    self.run_cycle()
                except Exception:
                    log.exception("cycle failed")
                    self.repo.alert("ERROR", "engine", "cycle failed")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def start_api(self) -> None:
        import uvicorn

        from lunatrade.api.app import create_app

        app = create_app(self)
        config = uvicorn.Config(app, host=self.cfg.get("api.host", "0.0.0.0"), port=int(self.cfg.get("api.port", 8800)),
                                log_level="warning")
        threading.Thread(target=uvicorn.Server(config).run, name="api", daemon=True).start()
        log.info("dashboard: http://%s:%s/", self.cfg.get("api.host"), self.cfg.get("api.port"))

    def stop(self) -> None:
        self._stop.set()
        self.scheduler.stop()
        if self.streams:
            self.streams.stop()
        self.notifier.send("Stopped.", "info")
