import sys

import pandas as pd

WINDOW = pd.Timedelta(days=7)
DATA_END = pd.Timestamp("2026-03-30")

a = pd.read_csv("data/assignments.csv", parse_dates=["assigned_at"])
ev = pd.read_csv("data/events.csv", parse_dates=["event_time"])
orders = pd.read_csv("data/orders.csv", parse_dates=["created_at"])

ev = ev.merge(a[["user_id", "variant", "assigned_at"]], on="user_id", how="left")
orders = orders.merge(a[["user_id", "variant", "assigned_at"]], on="user_id", how="left")
ev["in_win"] = ev.event_time < ev.assigned_at + WINDOW
orders["in_win"] = orders.created_at < orders.assigned_at + WINDOW
w = orders[orders.in_win]


def reached(event_type):
    return set(ev.loc[ev.in_win & (ev.event_type == event_type), "user_id"])


cart, pay, buy = reached("add_to_cart"), reached("payment_start"), reached("purchase")
purchases = ev.loc[ev.event_type == "purchase", ["user_id", "event_time"]].drop_duplicates()
matched = orders.merge(purchases, left_on=["user_id", "created_at"], right_on=["user_id", "event_time"], how="left")

checks = {
    "пользователь в одном варианте": a.user_id.is_unique and a.variant.isin(["control", "treatment"]).all(),
    "у событий и заказов есть назначение": ev.assigned_at.notna().all() and orders.assigned_at.notna().all(),
    "нет событий до назначения": (ev.event_time >= ev.assigned_at).all(),
    "нет заказов до назначения": (orders.created_at >= orders.assigned_at).all(),
    "данные до 2026-03-29 включительно": ev.event_time.max() < DATA_END and orders.created_at.max() < DATA_END,
    "воронка вложенная: покупка ⊂ оплата ⊂ корзина": buy <= pay <= cart,
    "у каждого заказа есть событие purchase": matched.event_time.notna().all(),
    "один заказ на покупателя в окне": w.user_id.is_unique,
}
for name, ok in checks.items():
    print(f"[{'OK' if ok else 'FAIL'}] {name}")

a["cart"] = a.user_id.isin(cart)
a["pay"] = a.user_id.isin(pay)
a["buy"] = a.user_id.isin(w.user_id)
a["rev"] = a.user_id.map(w.groupby("user_id").revenue.sum()).fillna(0)
g = a.groupby("variant")
late = orders[~orders.in_win].groupby("variant").size()

pct = "{:.2%}".format
rub = "{:,.1f}".format
rows = [
    ("пользователей", g.size(), "{:,}".format),
    ("доля корзины", g.cart.mean(), pct),
    ("доля оплаты", g.pay.mean(), pct),
    ("конверсия 7д", g.buy.mean(), pct),
    ("средний чек, ₽", w.groupby("variant").revenue.mean(), rub),
    ("ARPU 7д, ₽", g.rev.mean(), rub),
    ("доля возвратов", w.groupby("variant").refunded.mean(), pct),
    ("медиана загрузки, мс", ev[ev.event_type == "view_checkout"].groupby("variant").page_load_ms.median(), "{:.0f}".format),
    ("заказов вне окна", late, "{:,}".format),
    ("заказов вне окна / польз.", late / g.size(), pct),
]
print(f"\n{'':<28}{'control':>12}{'treatment':>12}")
for name, s, fmt in rows:
    print(f"{name:<28}{fmt(s['control']):>12}{fmt(s['treatment']):>12}")

sys.exit(0 if all(checks.values()) else 1)
