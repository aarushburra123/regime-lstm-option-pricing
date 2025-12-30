
import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'
df = pd.read_csv(os.path.join(DATA_DIR, 'spy_vix_combined.csv'), index_col=0, parse_dates=True)
print(f"Loaded df shape: {df.shape}")
print(df.head())
print(df.tail())

# Mimic lag features
for i in range(1, 11):
    df[f'Return_lag_{i}'] = df['Returns_SPY'].shift(i)

# Mimic rolling
df['MA_60'] = df['Close_SPY'].rolling(window=60).mean()

# Mimic Target
df['Forward_5d_return'] = df['Close_SPY'].shift(-5) / df['Close_SPY'] - 1

# Mimic Volume merge
spy_prices_path = os.path.join(DATA_DIR, 'spy_prices.csv')
if os.path.exists(spy_prices_path):
    print(f"Reading {spy_prices_path}")
    spy_full = pd.read_csv(spy_prices_path, index_col=0, parse_dates=True)
    print(f"Spy full columns: {spy_full.columns}")
    print(spy_full.head())
    
    if 'Volume' in spy_full.columns:
        print("Merging Volume...")
        df['Volume'] = spy_full['Volume']
    else:
        print("Volume column not found in spy_prices")
else:
    print("spy_prices.csv not found")

# Check NaNs
print("\nNaNs before dropna:")
print(df.isna().sum())

df_clean = df.dropna()
print(f"\nShape after dropna: {df_clean.shape}")

if len(df_clean) == 0:
    print("ALL DATA DROPPED!")
    # Check which column is causing this
    for col in df.columns:
        if df[col].isna().all():
            print(f"Column {col} is ALL NaNs")
