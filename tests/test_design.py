import pytest

from abtest.design import breakeven, sample_size
from abtest.economics import incremental_profit


def test_sample_size_preregistration():
    assert sample_size(0.10, 0.005, 0.05, 0.8) == 57760


def test_breakeven_preregistration():
    arpu, conv = breakeven(6000, 2500, 0.30, 4_000_000, 2_000_000)
    assert arpu == pytest.approx(9.13, abs=0.005)
    assert conv == pytest.approx(0.00365, abs=0.00001)


def test_incremental_profit():
    users, cost = 6000 * 365, 6_000_000
    be, _ = breakeven(6000, 2500, 0.30, 4_000_000, 2_000_000)
    at_be = incremental_profit(be, (be - 5, be + 5), users, 0.30, cost)
    assert at_be.breakeven_arpu == pytest.approx(be)
    assert at_be.profit == pytest.approx(0, abs=1e-6)
    assert at_be.profit_low == pytest.approx(-5 * users * 0.30)
    r = incremental_profit(12.0, (2.0, 22.0), users, 0.30, cost)
    assert r.profit == pytest.approx(12 * users * 0.30 - cost)
    assert r.above_breakeven
    assert not incremental_profit(8.0, (0, 16), users, 0.30, cost).above_breakeven
