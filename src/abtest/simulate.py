from dataclasses import dataclass

import numpy as np
from scipy import stats

from abtest.stats import diff_proportions, holm, srm_test


@dataclass(frozen=True)
class Clustered:
    diff: np.ndarray
    p_session: np.ndarray
    p_user: np.ndarray
    var_session: np.ndarray
    var_user: np.ndarray


@dataclass(frozen=True)
class SRMBug:
    n_t: np.ndarray
    diff: np.ndarray
    p_value: np.ndarray
    srm: np.ndarray


def power_empirical(effects, p=0.1, n=63000, reps=5000, alpha=0.05, seed=0):
    """Доля значимых z-тестов при истинной разнице долей effects, по одной на эффект."""
    rng = np.random.default_rng(seed)
    eff = np.atleast_1d(np.asarray(effects, float))[:, None]
    xc = rng.binomial(n, p, (len(eff), reps))
    xt = rng.binomial(n, p + eff, (len(eff), reps))
    return (diff_proportions(xc, n, xt, n).p_value < alpha).mean(axis=1)


def power_analytic(effects, p=0.1, n=63000, alpha=0.05):
    """Мощность двустороннего z-теста с пулом, нормальная аппроксимация."""
    d = np.asarray(effects, float)
    p2 = p + d
    pool = (p + p2) / 2
    se0 = np.sqrt(2 * pool * (1 - pool) / n)
    se1 = np.sqrt((p * (1 - p) + p2 * (1 - p2)) / n)
    q = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf((d - q * se0) / se1) + stats.norm.cdf((-d - q * se0) / se1)


def daily_tests(effect=0.0, p=0.1, daily=6000, days=21, reps=5000, seed=0):
    """Накопленные оценки разницы и p-значения после каждого дня набора, shape (reps, days)."""
    rng = np.random.default_rng(seed)
    per = daily // 2
    n = per * np.arange(1, days + 1)
    xc = rng.binomial(per, p, (reps, days)).cumsum(axis=1)
    xt = rng.binomial(per, p + effect, (reps, days)).cumsum(axis=1)
    r = diff_proportions(xc, n, xt, n)
    return r.diff, r.p_value


def peeking_fpr(pvalues, alpha=0.05):
    """Доля повторов с хотя бы одним p < alpha при k равномерных проверках, k = 1..days."""
    days = pvalues.shape[1]
    hit = pvalues < alpha
    out = []
    for k in range(1, days + 1):
        looks = np.round(np.linspace(days / k, days, k)).astype(int) - 1
        out.append(hit[:, looks].any(axis=1).mean())
    return np.array(out)


def first_stop(diff, pvalues, alpha=0.05):
    """День первой проверки с p < alpha (0, если не было) и оценка разницы в этот день."""
    hit = pvalues < alpha
    idx = hit.argmax(axis=1)
    stopped = hit.any(axis=1)
    est = diff[np.arange(len(diff)), idx]
    return np.where(stopped, idx + 1, 0), np.where(stopped, est, np.nan)


def _ratio_var(num, den):
    n = num.shape[1]
    mx, my = den.mean(axis=1), num.mean(axis=1)
    cov = ((num - my[:, None]) * (den - mx[:, None])).sum(axis=1) / (n - 1)
    vx, vy = den.var(axis=1, ddof=1), num.var(axis=1, ddof=1)
    return (vy / mx**2 - 2 * my * cov / mx**3 + my**2 * vx / mx**4) / n


def clustered_aa(p=0.1, icc=0.1, m_max=8, n=63000, reps=2000, seed=0, batch=100):
    """A/A с повторными сессиями: z-тест по сессиям против дельта-метода по пользователям."""
    rng = np.random.default_rng(seed)
    ab = 1 / icc - 1
    out = []
    for k in range(0, reps, batch):
        size = (2, min(batch, reps - k), n)
        m = rng.integers(1, m_max + 1, size)
        y = rng.binomial(m, rng.beta(p * ab, (1 - p) * ab, size))
        x, s = y.sum(axis=2), m.sum(axis=2)
        r = diff_proportions(x[0], s[0], x[1], s[1])
        var_user = _ratio_var(y[0], m[0]) + _ratio_var(y[1], m[1])
        out.append((r.diff, r.p_value, r.se**2, var_user))
    diff, p_s, var_s, var_u = (np.concatenate(c) for c in zip(*out))
    p_u = 2 * stats.norm.sf(np.abs(diff) / np.sqrt(var_u))
    return Clustered(diff, p_s, p_u, var_s, var_u)


def design_effect(icc, m):
    """Design effect для кластеров размера m: 1 + (m - 1) * ICC."""
    return 1 + (m - 1) * icc


def srm_bug(p=0.1, seg_share=0.12, seg_p=0.03, lost=0.5, n=63000, reps=5000, seed=0):
    """Нет эффекта, но в treatment теряется доля lost сегмента с низкой конверсией."""
    rng = np.random.default_rng(seed)
    p_rest = (p - seg_share * seg_p) / (1 - seg_share)
    seg = rng.binomial(n, seg_share, (2, reps))
    rest = n - seg
    seg_t = rng.binomial(seg[1], 1 - lost)
    n_t = rest[1] + seg_t
    xc = rng.binomial(seg[0], seg_p) + rng.binomial(rest[0], p_rest)
    xt = rng.binomial(seg_t, seg_p) + rng.binomial(rest[1], p_rest)
    r = diff_proportions(xc, n, xt, n_t)
    srm = np.array([srm_test(n, k).srm for k in n_t])
    return SRMBug(n_t, r.diff, r.p_value, srm)


def fwer(m=5, rho=0.0, reps=5000, alpha=0.05, seed=0):
    """FWER без поправки и с Холмом для m метрик без эффекта с попарной корреляцией z-статистик rho."""
    rng = np.random.default_rng(seed)
    cov = np.full((m, m), rho)
    np.fill_diagonal(cov, 1)
    z = rng.multivariate_normal(np.zeros(m), cov, reps)
    p = 2 * stats.norm.sf(np.abs(z))
    raw = (p < alpha).any(axis=1).mean()
    adj = np.mean([holm(row, alpha).reject.any() for row in p])
    return raw, adj


def _lognormal(cv, mean):
    s2 = np.log1p(cv**2)
    return np.log(mean) - s2 / 2, np.sqrt(s2)


def _arpu_moments(n, cv, reps, rng, p, mean):
    mu, sigma = _lognormal(cv, mean)
    k = rng.binomial(n, p, reps)
    x = rng.lognormal(mu, sigma, k.sum())
    rep = np.repeat(np.arange(reps), k)
    s = np.bincount(rep, x, reps)
    ss = np.bincount(rep, x**2, reps)
    return s / n, (ss - s**2 / n) / (n - 1)


def welch_coverage(n, cv, reps=5000, p=0.1, mean=2500, alpha=0.05, seed=0, max_draws=5 * 10**6):
    """Покрытие нуля 95%-м CI Уэлча в A/A по ARPU = покупка × логнормальная сумма."""
    rng = np.random.default_rng(seed)
    batch = max(1, int(max_draws / (n * p)))
    covered = []
    for k in range(0, reps, batch):
        b = min(batch, reps - k)
        (ma, va), (mb, vb) = (_arpu_moments(n, cv, b, rng, p, mean) for _ in range(2))
        va, vb = va / n, vb / n
        df = (va + vb) ** 2 / ((va**2 + vb**2) / (n - 1))
        q = stats.t.ppf(1 - alpha / 2, df)
        covered.append(np.abs(mb - ma) <= q * np.sqrt(va + vb))
    return np.concatenate(covered).mean()


def mean_coverage(n, cv, reps=5000, p=0.1, mean=2500, alpha=0.05, seed=0, max_draws=5 * 10**6):
    """Покрытие истинного ARPU 95%-м t-интервалом для среднего одной группы."""
    rng = np.random.default_rng(seed)
    batch = max(1, int(max_draws / (n * p)))
    q = stats.t.ppf(1 - alpha / 2, n - 1)
    covered = []
    for k in range(0, reps, batch):
        m, v = _arpu_moments(n, cv, min(batch, reps - k), rng, p, mean)
        covered.append(np.abs(m - p * mean) <= q * np.sqrt(v / n))
    return np.concatenate(covered).mean()


def arpu_sample(n, cv, rng, p=0.1, mean=2500):
    """Суммы покупок у покупателей; остальные n - len пользователей имеют ARPU 0."""
    mu, sigma = _lognormal(cv, mean)
    return rng.lognormal(mu, sigma, rng.binomial(n, p))


def boot_means(x, n, n_boot, rng):
    """Бутстреп-средние по n пользователям, из которых ненулевые только x; ресэмплируются лишь покупатели."""
    k = rng.binomial(n, len(x) / n, n_boot)
    c = np.concatenate([[0], np.cumsum(x[rng.integers(0, len(x), k.sum())])])
    end = np.cumsum(k)
    return (c[end] - c[end - k]) / n


def boot_coverage(n, cv, reps=500, n_boot=1000, p=0.1, mean=2500, alpha=0.05, seed=0):
    """Покрытие нуля перцентильным бутстреп-CI в A/A по ARPU."""
    rng = np.random.default_rng(seed)
    covered = np.empty(reps, bool)
    for i in range(reps):
        a, b = arpu_sample(n, cv, rng, p, mean), arpu_sample(n, cv, rng, p, mean)
        d = boot_means(b, n, n_boot, rng) - boot_means(a, n, n_boot, rng)
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        covered[i] = lo <= 0 <= hi
    return covered.mean()
