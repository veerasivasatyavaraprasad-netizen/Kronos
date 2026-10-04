"""Asset dictionary for entity extraction: names/tickers/aliases -> base asset."""
from __future__ import annotations

import re

ASSET_ALIASES = {
    "BTC": ["bitcoin", "btc", "xbt", "satoshi", "spot bitcoin etf", "ibit", "gbtc"],
    "ETH": ["ethereum", "eth", "ether", "vitalik", "spot ether etf"],
    "SOL": ["solana", "sol"],
    "BNB": ["bnb", "binance coin", "bnb chain"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "DOT": ["polkadot", "dot"],
    "LINK": ["chainlink", "link"],
    "TRX": ["tron", "trx"],
    "TON": ["toncoin", "ton"],
    "MATIC": ["polygon", "matic", "pol"],
    "LTC": ["litecoin", "ltc"],
    "SHIB": ["shiba inu", "shib"],
    "UNI": ["uniswap", "uni"],
    "ATOM": ["cosmos", "atom"],
    "NEAR": ["near protocol"],
    "APT": ["aptos", "apt"],
    "ARB": ["arbitrum", "arb"],
    "OP": ["optimism"],
    "SUI": ["sui"],
    "PEPE": ["pepe"],
    "FIL": ["filecoin", "fil"],
    "INJ": ["injective", "inj"],
    "AAVE": ["aave"],
    "MKR": ["maker", "makerdao", "mkr"],
    "USDT": ["tether", "usdt"],
    "USDC": ["usdc", "circle"],
}

# Words that mean "the whole crypto market" -> applies to every symbol (scaled down)
MARKET_TERMS = ["crypto market", "cryptocurrency market", "digital assets", "crypto industry", "altcoins",
                "stablecoin", "defi", "sec", "federal reserve", "fed ", "fomc", "interest rate", "inflation",
                "cpi", "regulation", "crypto"]

_PATTERNS = {a: re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(x) for x in al) + r")(?![a-z0-9])")
             for a, al in ASSET_ALIASES.items()}
# Short tickers that are also common words need the $ or uppercase form to count
_AMBIGUOUS = {"DOT", "LINK", "UNI", "OP", "SUI", "TON", "SOL", "APT", "ARB", "ATOM", "FIL", "INJ", "POL", "NEAR"}


def extract_assets(text: str) -> list[str]:
    low = f" {text.lower()} "
    found = []
    for asset, pat in _PATTERNS.items():
        m = pat.search(low)
        if not m:
            continue
        if asset in _AMBIGUOUS and len(m.group(1)) <= 4:
            # require $SOL / SOL (uppercase) in the original text
            if not re.search(rf"(\${asset}\b|\b{asset}\b)", text):
                continue
        found.append(asset)
    return found


def mentions_market(text: str) -> bool:
    low = f" {text.lower()} "
    return any(t in low for t in MARKET_TERMS)


def base_of(symbol: str, quote: str = "USDT") -> str:
    for q in (quote, "USDT", "USDC", "FDUSD", "BUSD", "USD", "BTC"):
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol[: -len(q)]
    return symbol
