
# Midterm Live‑Coding Playbook (PS1 + PS2) — Paste‑Ready

**Purpose:** Fast execution during an offline, no‑internet, live‑coding midterm. Focus on the exact patterns used in PS1/PS2 and the five lectures. Copy/paste into Google Doc, adjust fonts, export as PDF.

---

## Exam Operating Model (offline hygiene)
- **Disable all AI helpers** in your IDE (Anaconda/VS Code Copilot popups, etc.).
- **Allowed during exam:** past homeworks (PS1, PS2) and **all 5 lecture notes** — not the internet. Mirror the patterns below and you’ll be fine.
- **Local datasets:** if needed, keep `chess_sample.csv` in the working dir. Adjust `index_col` exactly as shown below.
- **Imports to assume:** `numpy as np`, `pandas as pd`, optionally `matplotlib.pyplot as plt`, `from scipy.optimize import linprog` (PS2 LP).

---

## 0) Quick Patterns You’ll Reuse (hot keys)

### Lists & list comprehensions
```python
# List from 0 to <100> in steps of 4
x_values = list(range(0, 100, 4))

# Map/transform (fast during exam)
plus_one = [x + 1 for x in x_values]
roots = [x**0.5 for x in x_values]
```
_PS1 used range/LC extensively. If you try raw list arithmetic like `x_list * y_list`, you’ll get a TypeError. Convert to NumPy first._

### NumPy vectorization
```python
import numpy as np
X = np.array([2, 3, 4])
Y = np.array([10, 20, 30])

X * Y           # elementwise product
np.mean(X)      # 3.0 etc.
```
_List ops that fail on Python lists (e.g., `list + 1` or `list * list`) work once you convert to arrays. Useful for PS1 vectorization tasks._

### Pandas: read, group, percentage, conditional update
```python
import pandas as pd

# Load with first column as index
df = pd.read_csv("chess_sample.csv", index_col=0)

# Group counts (size) and percentages
total_by_player = df.groupby("player").size()                  # counts
share_by_outcome = df.groupby("outcome").size() / len(df)      # shares

# Faster percentage via value_counts if you don't need groupby shape:
share_by_outcome_v2 = df["outcome"].value_counts(normalize=True)

# Conditional update (draw -> win/loss depending on ratings)
import numpy as np
cond = (df["outcome"] == 0.5) & (df["rating"] < df["opp_rating"])
df.loc[cond, "outcome"] = 0.0
cond = (df["outcome"] == 0.5) & (df["rating"] >= df["opp_rating"])
df.loc[cond, "outcome"] = 1.0

# Epoch seconds -> readable datetime
df["start_time"] = pd.to_datetime(df["start_time"], unit="s")
df["end_time"]   = pd.to_datetime(df["end_time"],   unit="s")
```
_Those cover PS1 3.2–3.4 patterns._

### Matplotlib quick plot (when allowed)
```python
import matplotlib.pyplot as plt
plt.hist(total_by_player)   # or .plot(kind="hist") on pandas
plt.show()
```

---

## 1) PS1 Patterns (replicate + extend)

### 1.1 Basic Python: list/range/list‑comp
- Create step sequences and transform with list comprehension.
```python
x_values = list(range(0, 100, 4))
plus_one = [x + 1 for x in x_values]
sqrt_x   = [x**0.5 for x in x_values]
```
- If you hit errors adding/multiplying lists, switch to NumPy arrays and retry.

### 1.2 NumPy: arrays, vector ops, norms (if asked)
```python
import numpy as np
a = np.random.normal(loc=0, scale=2, size=5)    # ~N(0, 2)
b = np.random.uniform(low=-1, high=1, size=5)   # U(-1, 1)

inner = np.dot(a, b)        # a^T b
l2 = np.linalg.norm(a)      # sqrt(sum a_i^2)
```

### 1.3 Pandas warmup: build DataFrame / select columns
```python
import pandas as pd
df = pd.DataFrame([[1,6],[4,5]], columns=["one","two"])
df["one"].head()
df[["one","two"]].head()
```

### 1.4 PS1 Chess mini‑project (I/O + transforms)
```python
# Load
df = pd.read_csv("chess_sample.csv", index_col=0)

# Inspect
df.head()
df.info()     # dtypes

# Q3.2 — fix epoch seconds to readable timestamps
df["start_time"] = pd.to_datetime(df["start_time"], unit="s")
df["end_time"]   = pd.to_datetime(df["end_time"],   unit="s")

# Q3.3 — total games per player
total_games = df.groupby("player").size()

# Q3.4 — share of outcomes
share = df["outcome"].value_counts(normalize=True)

# Optional: recode draws using ratings rule
is_draw = df["outcome"].eq(0.5)
df.loc[ is_draw & (df["rating"] <  df["opp_rating"]), "outcome"] = 0.0
df.loc[ is_draw & (df["rating"] >= df["opp_rating"]), "outcome"] = 1.0
share_after = df["outcome"].value_counts(normalize=True)

# Optional viz (if permitted)
# import matplotlib.pyplot as plt
# total_games.plot(kind="hist"); plt.show()
```

**Quality bar / pitfalls**
- If counts are wrong, check `index_col=0` on read, missing headers, or stray whitespace in column names.
- Prefer `groupby().size()` for counts; `value_counts(normalize=True)` for shares.
- Datetime conversion: your numeric timestamps look like `1.505e+09`, which are **seconds** since epoch, so use `unit="s"`.

---

## 2) PS2 — Linear Programming (hedge‑fund allocation)

**Shape the LP to match your exam narrative. The code skeleton below mirrors PS2 Q1.1–1.2 patterns.**

### 2.1 Model sketch
- Decision variables (sample naming; keep the exact PS2 ordering you used):
  - `x1` = yearly annuity investment (same each of years 1–4)
  - `x2, x3, x4, x5` = bank balances at start of years 1–4 (can be negative up to a limit)
  - `x6` = tech startup (only from year 2, cap 75,000)
  - `x7, x8` = real estate (years 3 and/or 4)
- Objective (end of year 4 payout, sample from PS2):
  - `1.40 * 4*x1 + 1.05*x5 + 2.50*x6 + 1.10*x7 + 1.10*x8`
- Core flow constraints (restate your PS2 equalities verbatim in matrices):
  - `x1 + x2 = 150000`
  - `x1 - 1.05*x2 + x3 + x6 = 0`
  - `x1 - 1.05*x3 + x4 + x7 = 0`
  - `x1 - 1.05*x4 + x5 + x8 = 0`
- Bounds:
  - Bank can be negative to −25k; annuity/startup/RE non‑negative; startup ≤ 75k.

### 2.2 Implementation template (SciPy linprog, maximize via sign flip)
```python
import numpy as np
from scipy.optimize import linprog

# Coeff vector c (maximize -> minimize -c)
# order: [x1, x2, x3, x4, x5, x6, x7, x8]
c = np.array([1.40*4, 0, 0, 0, 1.05, 2.50, 1.10, 1.10])

# Equality constraints A_eq x = b_eq
A_eq = np.array([
    [1,  1,     0,     0,     0, 0, 0, 0],          # x1 + x2 = 150000
    [1, -1.05,  1,     0,     0, 1, 0, 0],          # x1 - 1.05 x2 + x3 + x6 = 0
    [1,  0,    -1.05,  1,     0, 0, 1, 0],          # x1 - 1.05 x3 + x4 + x7 = 0
    [1,  0,     0,    -1.05,  1, 0, 0, 1],          # x1 - 1.05 x4 + x5 + x8 = 0
])
b_eq = np.array([150000, 0, 0, 0])

# Inequalities (A_ub x <= b_ub) — none besides bounds in this toy; leave empty
A_ub = None
b_ub = None

# Bounds (lower, upper) per variable
bounds = [
    (0, None),        # x1 >= 0
    (-25000, None),   # x2
    (-25000, None),   # x3
    (-25000, None),   # x4
    (-25000, None),   # x5
    (0, 75000),       # x6 cap
    (0, None),        # x7
    (0, None),        # x8
]

res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
              options={"tol": 1e-14})
assert res.success, res.message

opt_x = res.x
opt_val = -res.fun
opt_x, opt_val
```
_Tips: put all equalities into `A_eq`, keep income/flow signs consistent; flip sign on `c` to turn max into min; tweak `tol` if solver complains._

**Quality bar / pitfalls**
- If solver returns infeasible, triple‑check the year‑to‑year flow equations and bounds.
- Keep the annuity “same each year” by modeling it as **one** variable that appears in each yearly flow equation (as above).
- Interpret result with a small helper to label variables for clarity:
```python
names = ["x1_annuity","x2_bank_y1","x3_bank_y2","x4_bank_y3","x5_bank_y4","x6_startup","x7_RE_y3","x8_RE_y4"]
dict(zip(names, opt_x))
```

---

## 3) PS2 — Agent‑based simulation (Schelling‑style)

**Goal (Q2.2)**: On a 2D grid with two agent types and empty cells, iteratively move unhappy agents until everyone is happy. Track iteration count.

### 3.1 Definitions
- Grid `G` with values `{0: empty, 1: group A, 2: group B}`.
- Neighborhood: 8 adjacent cells (Moore neighborhood).
- Happiness rule: agent is “happy” if `(similar_neighbors / total_neighbors) >= threshold`.
- Movement: unhappy agent jumps to a random empty cell.

### 3.2 Vectorized neighbor counts (fast approach with `np.roll`)
```python
import numpy as np
rng = np.random.default_rng(0)

def neighbor_counts(mask):
    # count neighbors of SAME type using 8 shifts
    total = np.zeros_like(mask, dtype=int)
    for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        total += np.roll(np.roll(mask, dx, axis=0), dy, axis=1)
    return total

def step(G, threshold=0.5):
    # masks for A/B presence
    A = (G == 1).astype(int)
    B = (G == 2).astype(int)
    # similar neighbor counts
    simA = neighbor_counts(A)
    simB = neighbor_counts(B)
    # total occupied neighbors
    occ = neighbor_counts((G != 0).astype(int))
    # happiness (avoid divide by zero where occ==0 -> treat as happy)
    happyA = (occ == 0) | (simA / occ >= threshold)
    happyB = (occ == 0) | (simB / occ >= threshold)

    unhappy_mask = ((G == 1) & ~happyA) | ((G == 2) & ~happyB)
    empties = np.argwhere(G == 0)
    movers  = np.argwhere(unhappy_mask)

    if len(movers) == 0:
        return G, 0

    # move each unhappy to a random empty cell
    rng.shuffle(empties)
    count = min(len(movers), len(empties))
    # reassign first <count> movers into first <count> empties
    G2 = G.copy()
    for k in range(count):
        (i,j) = movers[k]
        (u,v) = empties[k]
        G2[u,v] = G[i,j]     # move agent
        G2[i,j] = 0          # leave empty
    return G2, count

def run_until_stable(G0, threshold=0.5, max_iters=500):
    G = G0.copy()
    for t in range(1, max_iters+1):
        G, moved = step(G, threshold=threshold)
        if moved == 0:
            return G, t
    return G, max_iters

# Example init
def make_grid(n=30, p_empty=0.1, p_A=0.45):
    N = n*n
    n_empty = int(N * p_empty)
    n_A = int((N - n_empty) * p_A)
    n_B = N - n_empty - n_A
    arr = np.array([0]*n_empty + [1]*n_A + [2]*n_B, dtype=int)
    rng.shuffle(arr)
    return arr.reshape(n, n)

G0 = make_grid(n=30, p_empty=0.1, p_A=0.5)
Gf, iters = run_until_stable(G0, threshold=0.6, max_iters=500)
iters
```
_Pattern hits PS2 Q2.2 spec: define “happy”, move unhappy until no changes, and track iteration count. Add prints/plots if allowed._

**Quality bar / pitfalls**
- Always cap iterations with `max_iters` to avoid infinite loops.
- If you need speed, move only a **random subset** of unhappy agents each iteration to reduce thrashing.
- Use one RNG (as shown) to keep reproducible behavior if the grader re‑runs it.

---

## 4) Debug checklist (fast triage during exam)
- Import errors: re‑type the exact import names (`from scipy.optimize import linprog`).
- Off‑by‑one bugs in `range(...)`: remember stop is **exclusive**.
- Pandas group results look weird? Confirm you’re grouping the correct column and not mixing strings with numbers due to bad CSV parsing.
- LP fails to converge? Re‑sign the objective or fix bounds/equalities; switch equality rows that were accidentally put in `A_ub`.

---

## 5) Minimal “one‑pager” to keep on screen

```python
# Lists
x_values = list(range(0, 100, 4))
y = [x+1 for x in x_values]

# NumPy
import numpy as np
X = np.array(x_values); Y = np.array(y)
X * Y; np.linalg.norm(X)

# Pandas
import pandas as pd
df = pd.read_csv("chess_sample.csv", index_col=0)
df["start_time"] = pd.to_datetime(df["start_time"], unit="s")
tot = df.groupby("player").size()
share = df["outcome"].value_counts(normalize=True)

# Recode draws by ratings
is_draw = df["outcome"].eq(0.5)
df.loc[is_draw & (df["rating"] <  df["opp_rating"]), "outcome"] = 0.0
df.loc[is_draw & (df["rating"] >= df["opp_rating"]), "outcome"] = 1.0

# LP
from scipy.optimize import linprog
c = np.array([1.40*4,0,0,0,1.05,2.50,1.10,1.10])
A_eq = np.array([[1,1,0,0,0,0,0,0],
                 [1,-1.05,1,0,0,1,0,0],
                 [1,0,-1.05,1,0,0,1,0],
                 [1,0,0,-1.05,1,0,0,1]])
b_eq = np.array([150000,0,0,0])
bounds = [(0,None),(-25000,None),(-25000,None),(-25000,None),
          (-25000,None),(0,75000),(0,None),(0,None)]
res = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, options={"tol":1e-14})
opt_x, opt_val = res.x, -res.fun

# Schelling step (neighbor via rolls)
def neighbor_counts(m):
    t = np.zeros_like(m, int)
    for dx,dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        t += np.roll(np.roll(m, dx, 0), dy, 1)
    return t
```
That’s the 80/20.

---

### Sources tied to your course materials
- PS1 chess tasks: reading CSV, time transform, groupby counts and shares, and draw‑recode rule align with your PS1 notebook. 
- PS2 LP (annuity/bank/startup/real‑estate, equality flows, bounds, tol in `linprog`) are lifted from the PS2 guidance.
- Lecture 3/4 reinforce `groupby`/aggregation patterns and SciPy `linprog` setup.

> Drop this whole section into your doc as “Appendix: patterns & references” and keep the code above in the main body for speed.
