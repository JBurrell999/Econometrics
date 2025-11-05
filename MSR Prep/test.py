import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm

df = pd.read_csv('/Users/jjburrell/Econometrics/Econometrics/MSR Prep/sample_school_curriculum_data.csv')
df.sample(n=20).head()
df.describe().T
df.dropna()
df.isna().mean().sort_values()
df.duplicated().sum

df = df.drop_duplicates()
df = df.dropna(subset= ['school_id'])
df['date_submitted'] = pd.to_datetime(df['date_submitted'])
df['year'] = df['date_submitted'].dt.year

df['subject_changed'] = df['subject_changed'].str.lower().str.strip()

