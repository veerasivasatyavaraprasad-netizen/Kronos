"""
Kronos automation CLI.

    python -m automation.kronos_auto setup-models            # pre-download model weights
    python -m automation.kronos_auto fetch --source yfinance --symbol AAPL --interval 1d --period 2y
    python -m automation.kronos_auto forecast --csv data/sample_a_share_5min.csv --pred-len 24
    python -m automation.kronos_auto run                     # forecast every symbol in config.yaml
    python -m automation.kronos_auto run --every 60          # ...and repeat every 60 minutes
    python -m automation.kronos_auto serve                   # start the web UI
"""
import argparse
import datetime as dt
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "automation" / "config.yaml"
DATA_DIR = ROOT / "data"


def _abs(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- reporting

def plot_forecast(name, hist, pred, paths, out_png, actual=None, show_hist=200):
    """Plot on a bar-index axis so market-closed gaps (nights/weekends) don't distort the chart."""
    hist = hist.tail(show_hist).reset_index(drop=True)
    n_h, n_p = len(hist), len(pred)
    xh, xp = np.arange(n_h), np.arange(n_h, n_h + n_p)
    labels = pd.concat([hist["timestamps"], pred["timestamps"]], ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(xh, hist["close"], color="#1f77b4", lw=1.3, label="History")
    ax1.plot(np.r_[xh[-1], xp], np.r_[hist["close"].iloc[-1], pred["close"]], color="#d62728", lw=1.6,
             label="Kronos forecast (mean)")
    if paths.shape[0] > 1:
        lo, hi = np.percentile(paths[:, :, 3], [10, 90], axis=0)
        ax1.fill_between(xp, lo, hi, color="#d62728", alpha=0.15, label="10-90% band")
    if actual is not None:
        ax1.plot(np.r_[xh[-1], xp], np.r_[hist["close"].iloc[-1], actual["close"]], color="#2ca02c", lw=1.3,
                 label="Actual")
    ax1.axvline(xh[-1], color="grey", ls="--", lw=0.8)
    ax1.set_title(f"{name} - close price forecast")
    ax1.set_ylabel("Close")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.bar(xh, hist["volume"], color="#1f77b4", alpha=0.6)
    ax2.bar(xp, pred["volume"].clip(lower=0), color="#d62728", alpha=0.6)
    ax2.set_ylabel("Volume")
    ax2.grid(alpha=0.3)
    ticks = np.linspace(0, n_h + n_p - 1, 8).astype(int)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([labels.iloc[i].strftime("%Y-%m-%d\n%H:%M") for i in ticks], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def summarize(name, hist, pred, paths):
    last_close = float(hist["close"].iloc[-1])
    end_close = float(pred["close"].iloc[-1])
    change = (end_close / last_close - 1) * 100
    end_paths = paths[:, -1, 3]
    return {
        "name": name,
        "last_timestamp": str(hist["timestamps"].iloc[-1]),
        "last_close": round(last_close, 6),
        "horizon_bars": len(pred),
        "horizon_end": str(pred["timestamps"].iloc[-1]),
        "forecast_close": round(end_close, 6),
        "forecast_change_pct": round(change, 3),
        "forecast_high": round(float(pred["high"].max()), 6),
        "forecast_low": round(float(pred["low"].min()), 6),
        "paths_up_pct": round(float((end_paths > last_close).mean() * 100), 1),
        "signal": "UP" if change > 0.25 else "DOWN" if change < -0.25 else "FLAT",
    }


def write_markdown(results, path, meta):
    lines = ["# Kronos forecast report", "",
             f"- Generated: {meta['generated']}", f"- Model: {meta['model']} on {meta['device']}", "",
             "| Symbol | Last close | Forecast close | Change % | Paths up % | Signal | Backtest MAPE % | Direction ok |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | — | — | — | — | ERROR: {r['error'][:60]} | — | — |")
            continue
        ev = r.get("backtest") or {}
        mape = f"{ev['mape_close_pct']:.2f}" if "mape_close_pct" in ev else "—"
        lines.append(
            f"| {r['name']} | {r['last_close']:.4f} | {r['forecast_close']:.4f} | {r['forecast_change_pct']:+.2f} | "
            f"{r['paths_up_pct']:.0f} | {r['signal']} | {mape} | {ev.get('direction_correct', '—')} |")
    lines += ["", "_Forecasts are model outputs for research only, not financial advice._", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- commands

def cmd_setup_models(args):
    from huggingface_hub import snapshot_download
    from automation.forecaster import MODELS
    keys = list(MODELS) if args.all else [args.model]
    for key in keys:
        for repo in MODELS[key][:2]:
            print(f"Downloading {repo} ...")
            snapshot_download(repo)
    print("Models cached.")


def cmd_fetch(args):
    from automation.data_sources import load_source
    spec = {"source": args.source, "symbol": args.symbol, "interval": args.interval,
            "period": args.period, "limit": args.limit}
    df = load_source(spec)
    DATA_DIR.mkdir(exist_ok=True)
    out = _abs(args.out) if args.out else DATA_DIR / f"{args.symbol.replace('/', '').replace('^', '')}_{args.interval}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


def run_symbols(specs, cfg, evaluate_too=True):
    from automation.data_sources import load_source
    from automation.forecaster import evaluate, forecast, load_predictor, resolve_device

    defaults = cfg.get("defaults", {})
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Loading {cfg.get('model', 'kronos-small')} on {device} ...")
    predictor = load_predictor(cfg.get("model", "kronos-small"), device)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _abs(cfg.get("output_dir", "outputs")) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in specs:
        name = spec["name"]
        p = {**defaults, **{k: v for k, v in spec.items() if k in defaults}}
        kw = dict(temperature=p["temperature"], top_p=p["top_p"],
                  sample_count=p["sample_count"], seed=p.get("seed"))
        print(f"\n== {name} ==")
        try:
            df = load_source(spec)
            df.to_csv(run_dir / f"{name}_input.csv", index=False)
            pred, paths = forecast(predictor, df, p["lookback"], p["pred_len"], **kw)
            pred.to_csv(run_dir / f"{name}_forecast.csv", index=False)
            plot_forecast(name, df, pred, paths, run_dir / f"{name}_forecast.png")
            res = summarize(name, df, pred, paths)

            if evaluate_too and len(df) >= p["lookback"] + p["pred_len"]:
                bpred, bpaths, actual, metrics = evaluate(predictor, df, p["lookback"], p["pred_len"], **kw)
                plot_forecast(f"{name} (backtest)", df.iloc[:-p["pred_len"]], bpred, bpaths,
                              run_dir / f"{name}_backtest.png", actual=actual)
                res["backtest"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
            print(json.dumps(res, indent=2))
        except Exception as e:
            traceback.print_exc()
            res = {"name": name, "error": f"{type(e).__name__}: {e}"}
        results.append(res)

    meta = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
            "model": cfg.get("model", "kronos-small"), "device": device}
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    write_markdown(results, run_dir / "summary.md", meta)

    latest = run_dir.parent / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    print(f"\nReport: {run_dir / 'summary.md'}  (also copied to {latest})")
    return results


def cmd_forecast(args):
    cfg = load_config(args.config)
    cfg["model"] = args.model or cfg.get("model")
    cfg["device"] = args.device or cfg.get("device")
    for k in ("lookback", "pred_len", "sample_count"):
        if getattr(args, k) is not None:
            cfg.setdefault("defaults", {})[k] = getattr(args, k)
    name = args.name or Path(args.csv).stem
    results = run_symbols([{"name": name, "source": "csv", "path": str(_abs(args.csv))}], cfg, not args.no_backtest)
    return 0 if all("error" not in r for r in results) else 1


def cmd_run(args):
    while True:
        cfg = load_config(args.config)
        if args.model:
            cfg["model"] = args.model
        specs = [s for s in cfg.get("symbols", []) if s.get("enabled", True)]
        if args.only:
            specs = [s for s in cfg.get("symbols", []) if s["name"] in args.only]
        if not specs:
            print("No enabled symbols in config.")
            return 1
        results = run_symbols(specs, cfg, not args.no_backtest)
        failed = [r["name"] for r in results if "error" in r]
        if not args.every:
            if failed:
                print(f"Failed: {failed}")
            return 1 if len(failed) == len(results) else 0
        print(f"Sleeping {args.every} minutes ... (Ctrl+C to stop)")
        time.sleep(args.every * 60)


def cmd_serve(args):
    import os
    os.environ.setdefault("KRONOS_DATA_DIR", str(DATA_DIR))
    sys.path.insert(0, str(ROOT / "webui"))
    from webui.app import app
    print(f"Kronos Web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kronos_auto", description="Automated Kronos forecasting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup-models", help="download model weights from Hugging Face")
    s.add_argument("--model", default="kronos-small")
    s.add_argument("--all", action="store_true")
    s.set_defaults(func=cmd_setup_models)

    s = sub.add_parser("fetch", help="download market data into data/")
    s.add_argument("--source", choices=["yfinance", "binance"], default="yfinance")
    s.add_argument("--symbol", required=True)
    s.add_argument("--interval", default="1d")
    s.add_argument("--period", default="2y", help="yfinance period, e.g. 60d, 1y, 5y")
    s.add_argument("--limit", type=int, default=1000, help="binance bar count")
    s.add_argument("--out")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("forecast", help="forecast a single CSV file")
    s.add_argument("--csv", required=True)
    s.add_argument("--name")
    s.add_argument("--model")
    s.add_argument("--device")
    s.add_argument("--lookback", type=int)
    s.add_argument("--pred-len", dest="pred_len", type=int)
    s.add_argument("--samples", dest="sample_count", type=int)
    s.add_argument("--no-backtest", action="store_true")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("run", help="forecast all enabled symbols from the config")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--model")
    s.add_argument("--only", nargs="+", help="run only these symbol names (even if disabled)")
    s.add_argument("--every", type=float, help="repeat every N minutes")
    s.add_argument("--no-backtest", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7070)
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
