import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from math import sqrt
import os

# ---------------------------
# 1) Download monthly data
# ---------------------------
START = "2010-01-01"
END = None          # None = up to today
INTERVAL = "1mo"

# Tickers:
#  - GOOG: primary asset to forecast (Y)
#  - AAPL, SPY: returns factors
#  - ^VIX: volatility index (level; we use change)
#  - ^IRX: 13-week T-bill rate (index level, ~annualized %)
#  - ^TNX: 10-year Treasury yield (index level, ~10x percent; e.g., 20 -> 2.0%)
TICKERS = ["GOOG", "AAPL", "SPY", "^VIX", "^IRX", "^TNX"]

print("Downloading data from Yahoo Finance...")
raw = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    interval=INTERVAL,
    auto_adjust=True,
    group_by="ticker",
    progress=False,
)

# Helper to extract a single column series for each ticker
def close_series(d, ticker):
    if ticker not in d.columns.get_level_values(0):
        raise ValueError(f"Ticker {ticker} not found in downloaded data.")
    # yfinance returns a multi-index (ticker, field)
    s = d[(ticker, "Close")].copy()
    s.name = ticker
    return s

GOOG = close_series(raw, "GOOG")
AAPL = close_series(raw, "AAPL")
SPY  = close_series(raw, "SPY")
VIX  = close_series(raw, "^VIX")
IRX  = close_series(raw, "^IRX")
TNX  = close_series(raw, "^TNX")

# Align and drop rows with all-NaN
prices = pd.concat([GOOG, AAPL, SPY, VIX, IRX, TNX], axis=1).dropna(how="all")

# ---------------------------
# 2) Construct monthly features
# ---------------------------
# Returns for equities (GOOG, AAPL, SPY)
ret_GOOG = prices["GOOG"].pct_change().rename("Y_asset_return")
ret_AAPL = prices["AAPL"].pct_change().rename("X_AAPL_return")
ret_SPY  = prices["SPY"].pct_change().rename("X_SPY_return")

# Level changes (not pct) for rates/vol (stationarity-friendly)
# For IRX and TNX, the units are index points; we use first differences.
dVIX = prices["^VIX"].diff().rename("X_dVIX")
dIRX = prices["^IRX"].diff().rename("X_dIRX")
dTNX = prices["^TNX"].diff().rename("X_dTNX")

df = pd.concat([ret_GOOG, ret_SPY, ret_AAPL, dIRX, dTNX, dVIX], axis=1).dropna()
df.index.name = "date"

# Keep at least 100 obs
if len(df) < 100:
    raise RuntimeError(f"Not enough observations after cleaning: {len(df)} (<100). Try an earlier START date.")

# Save dataset
out_dir = os.path.abspath(".")
csv_path = os.path.join(out_dir, "project_dataset_hw4_real.csv")
dta_path = os.path.join(out_dir, "project_dataset_hw4_real.dta")
df.to_csv(csv_path)
try:
    df.to_stata(dta_path, write_index=True)
except Exception as e:
    print(f"Stata export failed (optional): {e}")

print(f"Dataset saved:\n- CSV: {csv_path}\n- Stata: {dta_path}")

# ---------------------------
# 3) ARMA model selection
# ---------------------------
y = df["Y_asset_return"].copy()

candidates = []
for p, q in product(range(0, 4), range(0, 4)):
    if p == 0 and q == 0:
        continue
    try:
        m = ARIMA(y, order=(p,0,q)).fit()
        candidates.append({"p": p, "q": q, "aic": m.aic, "bic": m.bic, "model": m})
    except Exception:
        pass

if not candidates:
    raise RuntimeError("No ARMA models successfully fit. Try changing p/q ranges or differencing.")

best = min(candidates, key=lambda c: c["bic"])
best_p, best_q, best_model = best["p"], best["q"], best["model"]
print(f"Best ARMA order by BIC: ({best_p},0,{best_q})  BIC={best_model.bic:.3f}  AIC={best_model.aic:.3f}")

with open(os.path.join(out_dir, "best_arma_model_summary.txt"), "w") as f:
    f.write(best_model.summary().as_text())

# ---------------------------
# 4) Ex-post 5-step forecast
# ---------------------------
h = 5
train = y.iloc[:-h]
test  = y.iloc[-h:]

best_train = ARIMA(train, order=(best_p,0,best_q)).fit()
pred_best = best_train.get_forecast(steps=h).predicted_mean
rmsfe_best = sqrt(np.mean((test - pred_best)**2))

ar1_train = ARIMA(train, order=(1,0,0)).fit()
pred_ar1 = ar1_train.get_forecast(steps=h).predicted_mean
rmsfe_ar1 = sqrt(np.mean((test - pred_ar1)**2))

print(f"RMSFE (Best ARMA) over last {h}: {rmsfe_best:.8f}")
print(f"RMSFE (AR(1))      over last {h}: {rmsfe_ar1:.8f}")

plt.figure(figsize=(10,5))
y.iloc[-40:].plot(label="Actual (last 40)", linewidth=2)
pred_best.rename("Ex-post forecast (Best ARMA)").plot(style="--")
pred_ar1.rename("Ex-post forecast (AR(1))").plot(style=":")
plt.title("Ex-Post 5-Step Forecast vs Actual")
plt.xlabel("Date"); plt.ylabel("Return"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(out_dir, "expost_5step_forecast_real.png"), dpi=160)
plt.close()

# ---------------------------
# 5) Ex-ante 2-step forecast
# ---------------------------
full = ARIMA(y, order=(best_p,0,best_q)).fit()
fc = full.get_forecast(steps=2)
mean_fc = fc.predicted_mean
ci_fc   = fc.conf_int(alpha=0.05)

# Reindex to future months (month-end)
last = y.index[-1]
future_idx = pd.date_range(last + pd.offsets.MonthEnd(1), periods=2, freq="M")
mean_fc.index = future_idx
ci_fc.index   = future_idx

# Plot
plt.figure(figsize=(10,5))
y.iloc[-60:].plot(label="Actual (last 60)", linewidth=2)
mean_fc.rename("Ex-ante forecast (2 steps)").plot(style="--")
plt.fill_between(mean_fc.index, ci_fc.iloc[:,0], ci_fc.iloc[:,1], alpha=0.2, label="95% CI")
plt.title("Ex-Ante 2-Step Forecast Beyond Sample")
plt.xlabel("Date"); plt.ylabel("Return"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(out_dir, "exante_2step_forecast_real.png"), dpi=160)
plt.close()

# ---------------------------
# 6) Key metrics file
# ---------------------------
metrics = {
    "best_order": (best_p,0,best_q),
    "best_model_BIC": float(best_model.bic),
    "best_model_AIC": float(best_model.aic),
    "RMSFE_best_expost_5": float(rmsfe_best),
    "RMSFE_ar1_expost_5": float(rmsfe_ar1),
    "files": {
        "csv": csv_path,
        "dta": dta_path,
        "model_summary": "best_arma_model_summary.txt",
        "fig_expost": "expost_5step_forecast_real.png",
        "fig_exante": "exante_2step_forecast_real.png"
    }
}
with open(os.path.join(out_dir, "metrics_real.json"), "w") as f:
    import json
    json.dump(metrics, f, indent=2)

print("Done. Files written:")
for k, v in metrics["files"].items():
    print(f" - {k}: {v}")
