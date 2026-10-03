"""Model loading, forecasting and evaluation on normalized K-line data."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODELS = {
    "kronos-mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "kronos-base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_predictor(model_key: str = "kronos-small", device: str = "auto",
                   model_path: str = None, tokenizer_path: str = None) -> KronosPredictor:
    """Load a released model by key, or a fine-tuned one via model_path/tokenizer_path."""
    if model_path:
        model_id = str(ROOT / model_path) if (ROOT / model_path).exists() else model_path
        tokenizer_id = str(ROOT / tokenizer_path) if tokenizer_path and (ROOT / tokenizer_path).exists() else tokenizer_path
        if not tokenizer_id:
            raise ValueError("tokenizer_path is required together with model_path")
        max_ctx = 512
    elif model_key in MODELS:
        model_id, tokenizer_id, max_ctx = MODELS[model_key]
    else:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(MODELS)}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    model.eval()
    return KronosPredictor(model, tokenizer, device=resolve_device(device), max_context=max_ctx)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def future_timestamps(ts: pd.Series, n: int) -> pd.DatetimeIndex:
    """Extend a timestamp series by n steps using its dominant bar interval."""
    ts = pd.to_datetime(ts).reset_index(drop=True)
    step = ts.diff().dropna().tail(200).mode().iloc[0]
    # Daily bars without weekends (stocks) -> continue on business days.
    if step == pd.Timedelta(days=1) and not (ts.dt.dayofweek >= 5).any():
        return pd.bdate_range(ts.iloc[-1] + pd.Timedelta(days=1), periods=n)
    return pd.date_range(ts.iloc[-1] + step, periods=n, freq=step)


def forecast(predictor: KronosPredictor, df: pd.DataFrame, lookback: int, pred_len: int,
             temperature=1.0, top_p=0.9, sample_count=1, y_timestamp=None, seed=None, verbose=False):
    """
    Forecast `pred_len` bars after the last `lookback` rows of df.

    Returns (mean_df, paths) where paths has shape (sample_count, pred_len, 6) so the
    caller can draw an uncertainty band.
    """
    lookback = min(lookback, len(df), predictor.max_context)
    hist = df.tail(lookback).reset_index(drop=True)
    x_ts = hist["timestamps"]
    y_ts = pd.Series(y_timestamp if y_timestamp is not None else future_timestamps(x_ts, pred_len))

    if seed is not None:
        set_seed(seed)
    n = max(1, int(sample_count))
    # Repeat the series in one batch to get n independent paths in a single pass.
    preds = predictor.predict_batch(
        df_list=[hist[FEATURES]] * n,
        x_timestamp_list=[x_ts] * n,
        y_timestamp_list=[y_ts] * n,
        pred_len=pred_len, T=temperature, top_p=top_p, sample_count=1, verbose=verbose,
    )
    paths = np.stack([p[FEATURES].values for p in preds])
    mean_df = pd.DataFrame(paths.mean(axis=0), columns=FEATURES)
    mean_df.insert(0, "timestamps", pd.to_datetime(y_ts.values))
    return mean_df, paths


def evaluate(predictor, df, lookback, pred_len, **kw):
    """Backtest on the most recent window: hide the last pred_len bars and score the forecast."""
    if len(df) < lookback + pred_len:
        raise ValueError(f"Need {lookback + pred_len} rows for evaluation, have {len(df)}")
    train, actual = df.iloc[:-pred_len], df.iloc[-pred_len:].reset_index(drop=True)
    pred, paths = forecast(predictor, train, lookback, pred_len, y_timestamp=actual["timestamps"], **kw)
    err = pred["close"].values - actual["close"].values
    last = train["close"].iloc[-1]
    metrics = {
        "mae_close": float(np.abs(err).mean()),
        "mape_close_pct": float((np.abs(err) / np.abs(actual["close"].values)).mean() * 100),
        "direction_correct": bool(np.sign(pred["close"].iloc[-1] - last) == np.sign(actual["close"].iloc[-1] - last)),
        "step_direction_accuracy_pct": float(
            (np.sign(np.diff(np.r_[last, pred["close"].values])) ==
             np.sign(np.diff(np.r_[last, actual["close"].values]))).mean() * 100),
    }
    return pred, paths, actual, metrics
