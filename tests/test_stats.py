import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from abtest.stats import (
    _boot,
    bootstrap_diff,
    diff_proportions,
    holm,
    one_sided_harm,
    ratio_delta,
    smd_categorical,
    srm_test,
    welch_diff,
)


def test_srm_matches_scipy():
    r = srm_test(62863, 63137)
    ref = stats.chisquare([62863, 63137])
    assert r.chi2 == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.p_value == pytest.approx(0.44, abs=0.005)
    assert not r.srm


def test_srm_flags_imbalance():
    assert srm_test(62000, 64000).srm


def test_smd_categorical():
    df = pd.DataFrame({
        "platform": ["ios"] * 50 + ["web"] * 50 + ["ios"] * 60 + ["web"] * 40,
        "variant": ["control"] * 100 + ["treatment"] * 100,
    })
    r = smd_categorical(df, "platform", "variant")
    smd = 0.1 / np.sqrt((0.25 + 0.24) / 2)
    assert r.by_category["ios"] == pytest.approx(smd)
    assert r.by_category["web"] == pytest.approx(-smd)
    assert r.max_abs == pytest.approx(smd)


def test_diff_proportions_reference():
    r = diff_proportions(1000, 10000, 1100, 10000)
    assert r.diff == pytest.approx(0.01)
    assert r.se == pytest.approx(0.004335, abs=1e-6)
    assert r.ci_low == pytest.approx(0.00150, abs=1e-5)
    assert r.ci_high == pytest.approx(0.01850, abs=1e-5)


def test_diff_proportions_matches_statsmodels_and_manual():
    x_c, n_c, x_t, n_t = 6280, 63000, 6510, 63100
    r = diff_proportions(x_c, n_c, x_t, n_t)
    z, p = proportions_ztest([x_t, x_c], [n_t, n_c])
    assert r.z == pytest.approx(z)
    assert r.p_value == pytest.approx(p)
    pc, pt = x_c / n_c, x_t / n_t
    se = np.sqrt(pc * (1 - pc) / n_c + pt * (1 - pt) / n_t)
    assert r.ci_low == pytest.approx(pt - pc - 1.959964 * se)
    assert r.ci_high == pytest.approx(pt - pc + 1.959964 * se)
    assert r.rel == pytest.approx(pt / pc - 1)
    assert r.rel_ci_low < r.rel < r.rel_ci_high


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.exponential(250, 3000)
    b = rng.exponential(270, 2500)
    r = welch_diff(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert r.t == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.df == pytest.approx(ref.df)
    assert r.ci_low == pytest.approx(ci.low)
    assert r.ci_high == pytest.approx(ci.high)


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    r = holm(p)
    reject, p_adj, _, _ = multipletests(p, alpha=0.05, method="holm")
    np.testing.assert_allclose(r.p_adj, p_adj)
    np.testing.assert_array_equal(r.reject, reject)


def test_ratio_delta_one_order_equals_welch():
    rng = np.random.default_rng(2)
    rev_c = rng.lognormal(7.8, 0.5, 6000)
    rev_t = rng.lognormal(7.75, 0.5, 6300)
    r = ratio_delta(rev_c, np.ones_like(rev_c), rev_t, np.ones_like(rev_t))
    w = welch_diff(rev_c, rev_t)
    width = w.ci_high - w.ci_low
    assert r.diff == pytest.approx(w.diff)
    assert r.se == pytest.approx(w.se, rel=0.01)
    assert r.ci_low == pytest.approx(w.ci_low, abs=0.01 * width)
    assert r.ci_high == pytest.approx(w.ci_high, abs=0.01 * width)


def test_bootstrap_close_to_welch():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0.2, 1, 2000)
    r = bootstrap_diff(a, b)
    w = welch_diff(a, b)
    width = w.ci_high - w.ci_low
    assert r.diff == pytest.approx(w.diff)
    assert r.ci_low == pytest.approx(w.ci_low, abs=0.05 * width)
    assert r.ci_high == pytest.approx(w.ci_high, abs=0.05 * width)


def test_bootstrap_batches_match_single_pass():
    x = np.random.default_rng(4).normal(0, 1, 500)
    full = _boot(x, np.median, 2000, np.random.default_rng(0))
    batched = _boot(x, np.median, 2000, np.random.default_rng(0), max_cells=7 * len(x))
    np.testing.assert_allclose(batched, full)


def test_one_sided_harm():
    assert one_sided_harm(0.02, 0.01, worse="higher").harm
    assert not one_sided_harm(0.02, 0.01, worse="lower").harm
    assert one_sided_harm(-0.02, 0.01, worse="lower").harm
    assert one_sided_harm(1.0, 1.0).p_value == pytest.approx(stats.norm.sf(1.0))


def test_aa_coverage():
    rng = np.random.default_rng(42)
    n, p, reps = 10000, 0.1, 500
    xc = rng.binomial(n, p, reps)
    xt = rng.binomial(n, p, reps)
    covered = [
        (r := diff_proportions(c, n, t, n)).ci_low <= 0 <= r.ci_high
        for c, t in zip(xc, xt)
    ]
    assert 0.93 <= np.mean(covered) <= 0.97
