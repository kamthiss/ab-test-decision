from dataclasses import dataclass

from abtest.design import breakeven


@dataclass(frozen=True)
class Profit:
    profit: float
    profit_low: float
    profit_high: float
    breakeven_arpu: float
    above_breakeven: bool


def incremental_profit(arpu_diff, arpu_ci, users_per_year, margin, cost):
    """Годовая прибыль от внедрения: точечная оценка, интервал и флаг выше безубыточности."""
    be, _ = breakeven(users_per_year, 1, margin, cost, 0, days=1)
    lo, hi = (d * users_per_year * margin - cost for d in arpu_ci)
    return Profit(arpu_diff * users_per_year * margin - cost, lo, hi, be, arpu_diff > be)
