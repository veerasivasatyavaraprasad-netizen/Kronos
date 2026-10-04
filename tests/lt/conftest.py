import pytest

from lunatrade.config import load_config
from lunatrade.data.market_store import MarketStore
from lunatrade.data.synthetic import correlated_universe
from lt.helpers import SYMS


@pytest.fixture
def cfg(tmp_path):
    return load_config(use_env=False, overrides={"output_dir": str(tmp_path / "out"), "llm": {"providers": []},
                                                 "storage": {"database_url": f"sqlite:///{tmp_path / 'lt.db'}"}})


@pytest.fixture(scope="session")
def data():
    return correlated_universe(SYMS, 450, "15m", seed=11)


@pytest.fixture
def store(data):
    s = MarketStore("15m", 1000)
    for sym, df in data.items():
        s.load_frame(sym, df)
    return s
