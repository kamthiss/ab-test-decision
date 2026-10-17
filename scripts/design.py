from abtest.design import breakeven, duration, sample_size, sample_size_sm

DAILY = 6000
P0 = 0.10
AOV = 2500
MARGIN = 0.30
ONE_OFF = 4_000_000
YEARLY = 2_000_000
ALPHA = 0.05
POWER = 0.8
MDE = 0.005

n_norm = sample_size(P0, MDE, ALPHA, POWER)
n_sm = sample_size_sm(P0, MDE, ALPHA, POWER)
n = max(n_norm, n_sm)
days, wk_days = duration(n, DAILY)
total = wk_days * DAILY
arpu0 = P0 * AOV
d_arpu, d_conv = breakeven(DAILY, AOV, MARGIN, ONE_OFF, YEARLY)

print(f"n на группу, формула без пула: {n_norm:,}")
print(f"n на группу, statsmodels:      {n_sm:,} ({(n_sm / n_norm - 1) * 100:+.2f}%)")
print(f"n на группу, итог:             {n:,}")
print(f"длительность: {days:.2f} дн. -> {wk_days} дн. ({wk_days // 7} нед.)")
print(f"пользователей всего: {total:,}, на группу: {total // 2:,}")
print(f"безубыточность ARPU: +{d_arpu:.2f} ₽ на пользователя ({d_arpu / arpu0:.2%} от {arpu0:.0f} ₽)")
print(f"эквивалент по конверсии: +{d_conv * 100:.3f} п.п.")
