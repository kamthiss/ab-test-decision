from pathlib import Path

import numpy as np
import pandas as pd

from generator import config as cfg

STEPS = np.array(["session_start", "view_product", "add_to_cart", "view_checkout", "payment_start", "purchase"])
DAY = 86400


def draw(rng, probs, n):
    return rng.choice(list(probs), n, p=list(probs.values()))


def mult(values, probs, mults):
    norm = sum(probs[k] * mults[k] for k in probs)
    return pd.Series(values).map(mults).to_numpy() / norm


def to_time(sec):
    return (pd.Timestamp(cfg.START) + pd.to_timedelta(sec, unit="s")).floor("s")


def main():
    """Генерирует assignments, events, orders в data/."""
    rng = np.random.default_rng(cfg.SEED)
    n_days = len(pd.date_range(cfg.START, cfg.END))
    n = n_days * cfg.DAILY
    window = cfg.WINDOW_DAYS * DAY
    data_end = (pd.Timestamp(cfg.DATA_END) - pd.Timestamp(cfg.START)).total_seconds()

    assigned = (np.arange(n_days)[:, None] * DAY + np.sort(rng.uniform(0, DAY, (n_days, cfg.DAILY)), axis=1)).ravel()
    treat = rng.random(n) < cfg.P_TREATMENT
    variant = np.where(treat, "treatment", "control")
    platform = draw(rng, cfg.PLATFORM, n)
    country = draw(rng, cfg.COUNTRY, n)
    source = draw(rng, cfg.SOURCE, n)

    p_buy = (
        cfg.BUY_BASE
        * mult(platform, cfg.PLATFORM, cfg.PLATFORM_MULT)
        * mult(source, cfg.SOURCE, cfg.SOURCE_MULT)
        + cfg.BUY_LIFT * treat
    )
    u = rng.random(n)
    stage = (u < cfg.CART).astype(int) + (u < cfg.CHECKOUT) + (u < p_buy + cfg.PAY_EXTRA) + (u < p_buy)

    k = np.minimum(1 + rng.poisson(np.array(cfg.SESSION_LAM)[stage]), cfg.MAX_SESSIONS)
    s_user = np.repeat(np.arange(n), k)
    pos = np.arange(len(s_user)) - np.repeat(np.cumsum(k) - k, k)
    off = rng.uniform(0, window - cfg.SESSION_TAIL_S, len(s_user))
    off[pos == 0] = 0
    off = off[np.lexsort((off, s_user))]
    last = pos == np.repeat(k - 1, k)
    s_start = assigned[s_user] + off
    s_nev = np.where(last, 2 + stage[s_user], 1 + (rng.random(len(s_user)) < cfg.VIEW_PRODUCT))
    s_cap = np.full(len(s_user), float(cfg.GAP_MAX_S))

    late = np.flatnonzero(rng.random(n) < cfg.LATE_SHARE)
    lo = assigned[late] + window
    avail = data_end - lo
    s_user = np.concatenate([s_user, late])
    s_start = np.concatenate([s_start, lo + rng.uniform(0, 0.5, len(late)) * avail])
    s_nev = np.concatenate([s_nev, np.full(len(late), len(STEPS))])
    s_cap = np.concatenate([s_cap, 0.4 * avail / (len(STEPS) - 1)])
    order = np.argsort(s_user, kind="stable")
    s_user, s_start, s_nev, s_cap = s_user[order], s_start[order], s_nev[order], s_cap[order]

    first = np.cumsum(s_nev) - s_nev
    e_sess = np.repeat(np.arange(len(s_user)), s_nev)
    step = np.arange(len(e_sess)) - np.repeat(first, s_nev)
    gap = np.minimum(rng.exponential(cfg.GAP_MEAN_S, len(e_sess)), s_cap[e_sess])
    gap[step == 0] = 0
    cs = np.cumsum(gap)
    e_time = s_start[e_sess] + cs - np.repeat(cs[first], s_nev)
    e_user = s_user[e_sess]

    is_co = step == 3
    load = np.full(len(e_sess), np.nan)
    load[is_co] = np.round(rng.lognormal(np.log(cfg.LOAD_MEDIAN_MS), cfg.LOAD_SIGMA, is_co.sum()))

    is_buy = step == 5
    o_user = e_user[is_buy]
    sigma = np.sqrt(np.log(1 + cfg.AOV_CV**2))
    mu = np.log(pd.Series(variant[o_user]).map(cfg.AOV).to_numpy()) - sigma**2 / 2
    revenue = np.round(rng.lognormal(mu, sigma))
    refunded = rng.random(len(o_user)) < cfg.REFUND

    assignments = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "variant": variant,
        "assigned_at": to_time(assigned),
        "platform": platform,
        "country": country,
        "source": source,
    })
    events = pd.DataFrame({
        "event_id": np.arange(1, len(e_sess) + 1),
        "user_id": e_user + 1,
        "session_id": e_sess + 1,
        "event_time": to_time(e_time),
        "event_type": STEPS[step],
        "page_load_ms": pd.array(load, dtype="Int64"),
    })
    orders = pd.DataFrame({
        "order_id": np.arange(1, len(o_user) + 1),
        "user_id": o_user + 1,
        "created_at": to_time(e_time[is_buy]),
        "revenue": revenue.astype(int),
        "refunded": refunded,
    })

    out = Path("data")
    out.mkdir(exist_ok=True)
    for name, df in [("assignments", assignments), ("events", events), ("orders", orders)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,}")


if __name__ == "__main__":
    main()
