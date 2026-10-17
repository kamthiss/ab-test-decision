import math

from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def sample_size(p1, mde, alpha=0.05, power=0.8):
    """Размер группы для разницы двух долей, нормальная аппроксимация без пула."""
    p2 = p1 + mde
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / mde**2)


def sample_size_sm(p1, mde, alpha=0.05, power=0.8):
    """Размер группы через statsmodels (Cohen's h)."""
    h = proportion_effectsize(p1 + mde, p1)
    n = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0)
    return math.ceil(n)


def duration(n_per_group, daily_users, share=0.5):
    """Длительность в днях: точная и с округлением вверх до целых недель."""
    days = n_per_group / (daily_users * share)
    return days, math.ceil(days / 7) * 7


def breakeven(daily_users, aov, margin, one_off, yearly, days=365):
    """Безубыточный прирост ARPU в ₽ на пользователя и эквивалент по конверсии в долях."""
    users = daily_users * days
    arpu = (one_off + yearly * days / 365) / margin / users
    return arpu, arpu / aov
