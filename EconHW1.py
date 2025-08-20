from datetime import date
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ---- Config ----
TICKERS = ["MLI", "CWCO", "SEDG", "^GSPC"]
START = "2024-08-01"
END = date.today().isoformat()  # today's date
OUT_DIR = "plots"               # where figures will be saved
os.makedirs(OUT_DIR, exist_ok=True)