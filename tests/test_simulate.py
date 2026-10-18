import numpy as np
import pytest

from abtest.simulate import (
    boot_means,
    clustered_aa,
    daily_tests,
    first_stop,
    fwer,
    peeking_fpr,
    power_analytic,
    power_empirical,
    srm_bug,
)
from abtest.stats import bootstrap_diff


def test_aa_false_positive_rate():
    assert 0.035 <= power_empirical([0], reps=2000)[0] <= 0.065


def test_power_high_effect():
    assert power_empirical([0.008], reps=1000)[0] > 0.95


def test_power_analytic_preregistration():
    assert power_analytic(0) == pytest.approx(0.05)
    assert power_analytic(0.005) == pytest.approx(0.83, abs=0.005)


def test_holm_controls_fwer():
    raw, adj = fwer(reps=2000)
    assert raw > adj
    assert raw == pytest.approx(1 - 0.95**5, abs=0.03)


def test_peeking_inflates():
    _, p = daily_tests(reps=2000)
    curve = peeking_fpr(p)
    assert curve[-1] > 0.15 > 0.07 > curve[0]


def test_first_stop():
    diff = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = np.array([[0.5, 0.01, 0.01], [0.5, 0.5, 0.5]])
    day, est = first_stop(diff, p)
    np.testing.assert_array_equal(day, [2, 0])
    assert est[0] == 2.0 and np.isnan(est[1])


def test_clustered_session_analysis_inflated():
    r = clustered_aa(n=5000, reps=400)
    assert (r.p_session < 0.05).mean() > (r.p_user < 0.05).mean()


def test_srm_bug():
    r = srm_bug(reps=300)
    assert r.srm.all()
    assert r.diff.mean() == pytest.approx(0.0045, abs=0.0005)


def test_boot_means_matches_full_bootstrap():
    rng = np.random.default_rng(0)
    n = 2000
    a, b = np.zeros(n), np.zeros(n)
    a[:200] = rng.lognormal(7, 1, 200)
    b[:200] = rng.lognormal(7, 1, 200)
    d = boot_means(b[:200], n, 4000, rng) - boot_means(a[:200], n, 4000, rng)
    ref = bootstrap_diff(a, b, n_boot=4000)
    width = ref.ci_high - ref.ci_low
    lo, hi = np.quantile(d, [0.025, 0.975])
    assert lo == pytest.approx(ref.ci_low, abs=0.05 * width)
    assert hi == pytest.approx(ref.ci_high, abs=0.05 * width)
