SEED = 42

START = "2026-03-02"
END = "2026-03-22"
DATA_END = "2026-03-30"
DAILY = 6000
P_TREATMENT = 0.5
WINDOW_DAYS = 7

PLATFORM = {"ios": 0.40, "android": 0.45, "web": 0.15}
COUNTRY = {"RU": 0.55, "KZ": 0.15, "BY": 0.12, "UZ": 0.10, "AM": 0.08}
SOURCE = {"organic": 0.45, "paid": 0.30, "referral": 0.15, "email": 0.10}

PLATFORM_MULT = {"ios": 1.15, "android": 0.90, "web": 0.95}
SOURCE_MULT = {"organic": 1.00, "paid": 0.85, "referral": 1.20, "email": 1.30}

BUY_BASE = 0.100
BUY_LIFT = 0.008
PAY_EXTRA = 0.030
CHECKOUT = 0.20
CART = 0.30

AOV = {"control": 2500, "treatment": 2300}
AOV_CV = 1.0
REFUND = 0.05
LATE_SHARE = 0.02

LOAD_MEDIAN_MS = 1800
LOAD_SIGMA = 0.45

SESSION_LAM = [0.6, 1.0, 1.3, 1.6, 2.0]
MAX_SESSIONS = 7
VIEW_PRODUCT = 0.7
GAP_MEAN_S = 60
GAP_MAX_S = 1500
SESSION_TAIL_S = 3 * 3600
