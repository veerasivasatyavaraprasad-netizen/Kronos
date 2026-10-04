"""Map workers to strategy groups so performance memory can weight whole strategies, not just families."""
from __future__ import annotations

GROUP_RULES = [
    ("TREND", ("ema_cross", "vwap_trend", "adx_trend", "structure_hh", "macd", "trend_pullback", "trend_strength",
               "donchian")),
    ("MEAN_REVERSION", ("bollinger", "rsi_extreme", "vwap_deviation", "zscore", "keltner", "stoch", "xs_rank_reversal")),
    ("MOMENTUM", ("volume_expansion", "acceleration", "roc_", "breakout_confirmation", "obv", "relative_strength",
                  "xs_rank_momentum", "new_highs")),
    ("STRUCTURE", ("sr_bounce", "failed_breakout", "sweep_reclaim", "inside_bar")),
    ("PATTERN", ("engulfing", "hammer", "three_soldiers", "doji", "morning_evening", "double_top", "flag", "gap_scan")),
    ("VOLATILITY", ("vol_expansion", "squeeze", "realized_vs_long")),
    ("MICROSTRUCTURE", ("book_", "depth_", "taker_flow", "large_trades", "microprice")),
    ("POSITIONING", ("funding", "oi_change", "long_short", "liquidation", "basis", "coinbase_premium")),
    ("FORECAST", ("kronos", "linear_trend", "holt", "knn")),
]
FAMILY_GROUP = {"news": "NEWS", "social": "SOCIAL", "onchain": "ONCHAIN", "whale": "WHALE", "macro": "MACRO",
                "arbitrage": "POSITIONING"}
GROUPS = [g for g, _ in GROUP_RULES] + ["NEWS", "SOCIAL", "ONCHAIN", "WHALE", "MACRO", "ROTATION"]


def group_of(worker_id: str) -> str:
    family, _, name = worker_id.partition(".")
    for group, keys in GROUP_RULES:
        if any(k in name for k in keys):
            return group
    if family in FAMILY_GROUP:
        return FAMILY_GROUP[family]
    if family == "scanner":
        return "ROTATION"
    return family.upper()
