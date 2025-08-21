from datetime import date
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm

# ---- Config ----
TICKERS = ["MLI", "CWCO", "SEDG", "^GSPC"]
START = "2024-08-01"
END = date.today().isoformat()  # today's date
OUT_DIR = "plots"               # where figures will be saved
os.makedirs(OUT_DIR, exist_ok=True)


# We keep raw OHLCV, but will use Adjusted Close for returns
data = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    auto_adjust=False,     # keep raw; 'Adj Close' is returned separately
    progress=False,
    group_by='ticker'
)

def get_series(df, ticker, field="Adj Close"):
    """Return a (date-indexed) Series for the given ticker & field."""
    # multi-index (ticker, field) when multiple tickers requested
    if isinstance(df.columns, pd.MultiIndex):
        return df[(ticker, field)].dropna()
    # single ticker edge case
    return df[field].dropna()

# Build Adjusted Close price DataFrame (columns=tickers)
prices = pd.DataFrame({t: get_series(data, t, "Adj Close") for t in TICKERS})
prices = prices.dropna(how="all")
prices.head()

returns = prices.pct_change().dropna()
returns.head()

# Stata-like summary: N mean sd min p25 p50 p75 max
def stata_like_summary(df: pd.DataFrame) -> pd.DataFrame:
    q = df.quantile([0.25, 0.5, 0.75])
    out = pd.DataFrame({
        "N": df.count(),
        "Mean": df.mean(),
        "SD": df.std(ddof=1),
        "Min": df.min(),
        "p25": q.loc[0.25],
        "p50": q.loc[0.5],
        "p75": q.loc[0.75],
        "Max": df.max()
    })
    return out

print("=== Descriptive Stats: Prices (Adjusted Close) ===")
desc_prices = stata_like_summary(prices).round(4)
print(desc_prices)

print("\n=== Descriptive Stats: Simple Daily Returns ===")
desc_returns = stata_like_summary(returns).round(6)
print(desc_returns)

# Optional: save to CSV / Excel (for your submission)
desc_prices.to_csv("descriptive_stats_prices.csv")
desc_returns.to_csv("descriptive_stats_returns.csv")

with pd.ExcelWriter("descriptive_stats.xlsx", engine="xlsxwriter") as writer:
    desc_prices.to_excel(writer, sheet_name="Prices_AdjClose")
    desc_returns.to_excel(writer, sheet_name="Returns_Simple")

def plot_series(series: pd.Series, title: str, ylabel: str, outfile: str = None, show: bool = True):
    plt.figure()
    series.plot()  # default matplotlib
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    if show:
        plt.show()
    plt.close()

for t in TICKERS:
    # Price (Adjusted Close)
    plot_series(
        prices[t].dropna(),
        title=f"{t} — Adjusted Close (Daily)",
        ylabel="Price (USD)",
        outfile=os.path.join(OUT_DIR, f"{t}_price.png"),
        show=True
    )

    # Simple Returns
    if t in returns.columns:
        plot_series(
            returns[t].dropna(),
            title=f"{t} — Simple Daily Returns",
            ylabel="Return",
            outfile=os.path.join(OUT_DIR, f"{t}_return.png"),
            show=True
        )
prices.to_csv("prices_adjclose.csv")
returns.to_csv("returns_simple.csv")
