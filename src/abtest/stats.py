from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SRM:
    chi2: float
    p_value: float
    srm: bool


@dataclass(frozen=True)
class SMD:
    by_category: pd.Series
    max_abs: float


@dataclass(frozen=True)
class PropDiff:
    p_c: float
    p_t: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float
    rel: float
    rel_ci_low: float
    rel_ci_high: float


@dataclass(frozen=True)
class MeanDiff:
    diff: float
    se: float
    df: float
    ci_low: float
    ci_high: float
    t: float
    p_value: float


@dataclass(frozen=True)
class BootDiff:
    diff: float
    se: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class RatioDiff:
    ratio_c: float
    ratio_t: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float


@dataclass(frozen=True)
class Holm:
    p_adj: np.ndarray
    reject: np.ndarray


@dataclass(frozen=True)
class Harm:
    z: float
    p_value: float
    harm: bool


def srm_test(n_control, n_treatment, expected_share=0.5, threshold=0.001):
    """Хи-квадрат на соответствие фактического сплита ожидаемому."""
    obs = np.array([n_control, n_treatment])
    exp = obs.sum() * np.array([expected_share, 1 - expected_share])
    chi2 = ((obs - exp) ** 2 / exp).sum()
    p = stats.chi2.sf(chi2, 1)
    return SRM(chi2, p, p < threshold)


def smd_categorical(df, column, group_col, control="control", treatment="treatment"):
    """Стандартизованная разница долей treatment - control по каждой категории."""
    shares = pd.crosstab(df[column], df[group_col], normalize="columns")
    pc, pt = shares[control], shares[treatment]
    smd = (pt - pc) / np.sqrt((pc * (1 - pc) + pt * (1 - pt)) / 2)
    return SMD(smd, smd.abs().max())


def diff_proportions(x_c, n_c, x_t, n_t, alpha=0.05):
    """Разница долей: CI Вальда без пула, z-тест с пулом, относительный прирост с CI дельта-методом."""
    pc, pt = x_c / n_c, x_t / n_t
    var_c, var_t = pc * (1 - pc) / n_c, pt * (1 - pt) / n_t
    diff = pt - pc
    se = np.sqrt(var_c + var_t)
    q = stats.norm.ppf(1 - alpha / 2)
    pool = (x_c + x_t) / (n_c + n_t)
    z = diff / np.sqrt(pool * (1 - pool) * (1 / n_c + 1 / n_t))
    p = 2 * stats.norm.sf(abs(z))
    rel = pt / pc - 1
    se_rel = np.sqrt(var_t / pc**2 + pt**2 * var_c / pc**4)
    return PropDiff(
        pc, pt, diff, se, diff - q * se, diff + q * se, z, p,
        rel, rel - q * se_rel, rel + q * se_rel,
    )


def welch_diff(a, b, alpha=0.05):
    """Разница средних b - a, CI и p по Уэлчу."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = b.mean() - a.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t = diff / se
    q = stats.t.ppf(1 - alpha / 2, df)
    p = 2 * stats.t.sf(abs(t), df)
    return MeanDiff(diff, se, df, diff - q * se, diff + q * se, t, p)


def _boot(x, stat, n_boot, rng, max_cells=10**7):
    batch = max(1, max_cells // len(x))
    out = []
    for k in range(0, n_boot, batch):
        idx = rng.integers(0, len(x), (min(batch, n_boot - k), len(x)))
        out.append(stat(x[idx], axis=1))
    return np.concatenate(out)


def bootstrap_diff(a, b, stat=np.mean, n_boot=10000, alpha=0.05, seed=0):
    """Перцентильный бутстреп-CI разницы stat(b) - stat(a); stat должен принимать axis."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = _boot(b, stat, n_boot, rng) - _boot(a, stat, n_boot, rng)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return BootDiff(stat(b) - stat(a), d.std(ddof=1), lo, hi)


def _ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    mx, my = den.mean(), num.mean()
    cov = np.cov(num, den, ddof=1)
    var = (cov[0, 0] / mx**2 - 2 * my * cov[0, 1] / mx**3 + my**2 * cov[1, 1] / mx**4) / len(num)
    return num.sum() / den.sum(), var


def ratio_delta(num_c, den_c, num_t, den_t, alpha=0.05):
    """Разница ratio-метрик sum(num)/sum(den) по пользователям, дисперсия дельта-методом."""
    rc, vc = _ratio(num_c, den_c)
    rt, vt = _ratio(num_t, den_t)
    diff = rt - rc
    se = np.sqrt(vc + vt)
    q = stats.norm.ppf(1 - alpha / 2)
    z = diff / se
    return RatioDiff(rc, rt, diff, se, diff - q * se, diff + q * se, z, 2 * stats.norm.sf(abs(z)))


def holm(pvalues, alpha=0.05):
    """Поправка Холма: скорректированные p и флаги отклонения H0."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1)
    out = np.empty(m)
    out[order] = adj
    return Holm(out, out <= alpha)


def one_sided_harm(diff, se, alpha=0.05, worse="higher"):
    """Односторонний z-тест на ухудшение guardrail; worse задаёт плохое направление."""
    z = diff / se
    p = {"higher": stats.norm.sf, "lower": stats.norm.cdf}[worse](z)
    return Harm(z, p, p < alpha)
