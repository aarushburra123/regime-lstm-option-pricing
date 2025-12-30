"""
Feature Engineering Pipeline

Generates features for the regime-switching LSTM model.
Outputs: data/features_engineered.csv, models/feature_scaler.pkl

Replaces requirements of 02_feature_engineering.ipynb
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_feature_engineering(data_dir='data', models_dir='models'):
    print("Starting Feature Engineering...")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Combined Data
    input_path = os.path.join(data_dir, 'spy_vix_combined.csv')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from {input_path}")
    
    # 2. Add Volume Data (handling potential CSV formatting issues)
    spy_prices_path = os.path.join(data_dir, 'spy_prices.csv')
    if os.path.exists(spy_prices_path):
        # Read with index_col=0 but don't parse dates yet
        spy_full = pd.read_csv(spy_prices_path, index_col=0)
        
        # Clean up the index
        rows_to_drop = [idx for idx in spy_full.index if str(idx) in ['Ticker', 'Date', 'nan']]
        if rows_to_drop:
            spy_full = spy_full.drop(rows_to_drop)
        
        spy_full.index = pd.to_datetime(spy_full.index)
        
        if 'Volume' in spy_full.columns:
            # Convert to numeric, forcing errors to NaN
            spy_full['Volume'] = pd.to_numeric(spy_full['Volume'], errors='coerce')
            df['Volume'] = spy_full['Volume']
            print("Merged Volume data")
    else:
        print("WARNING: spy_prices.csv not found, skipping Volume features")
        df['Volume'] = 0

    # 3. Create Features
    
    # Lags (Returns)
    for i in range(1, 11):
        df[f'Return_lag_{i}'] = df['Returns_SPY'].shift(i)
        
    # Lags (Prices - normalized? No, usually raw prices are bad features for LSTM unless normalized)
    # The notebook used raw prices for lags? "Price_lag_1"
    # Better to use Returns or Log Returns. The notebook listed "Price_lag_1". 
    # Let's keep consistency with notebook plan but maybe check scaler.
    for i in range(1, 6):
        df[f'Price_lag_{i}'] = df['Close_SPY'].shift(i)

    # Moving Averages
    for window in [5, 20, 60]:
        df[f'MA_{window}'] = df['Close_SPY'].rolling(window=window).mean()
        # Return MA
        df[f'Return_MA_{window}'] = df['Returns_SPY'].rolling(window=window).mean()
        
    # Volatility Windows
    for window in [20, 60]:
        df[f'Vol_{window}d'] = df['Returns_SPY'].rolling(window=window).std() * np.sqrt(252)

    # Price vs MA
    df['Price_to_MA20'] = df['Close_SPY'] / df['MA_20']
    df['Price_to_MA60'] = df['Close_SPY'] / df['MA_60']
    df['Distance_from_MA20_pct'] = (df['Close_SPY'] - df['MA_20']) / df['MA_20']
    
    # VIX Features
    df['VIX_level'] = df['Close_VIX']
    df['VIX_change'] = df['Close_VIX'].diff()
    df['VIX_pct_change'] = df['Close_VIX'].pct_change()
    df['VIX_MA_10'] = df['Close_VIX'].rolling(window=10).mean()
    df['VIX_MA_60'] = df['Close_VIX'].rolling(window=60).mean()
    df['VIX_Vol_60'] = df['Close_VIX'].rolling(window=60).std()
    df['VIX_zscore'] = (df['Close_VIX'] - df['VIX_MA_60']) / df['VIX_Vol_60']
    
    # Regime Features
    df['Regime_binary'] = (df['Regime'] == 'High').astype(int)
    # Days in regime
    df['Regime_change'] = df['Regime_binary'].diff().ne(0)
    df['Days_in_regime'] = df.groupby(df['Regime_change'].cumsum()).cumcount() + 1
    
    # Volume Features (if available)
    if 'Volume' in df.columns:
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_spike'] = (df['Volume'] > 2 * df['Volume_MA_20']).astype(int)
        
    # 4. Target Variable
    # 5-day forward return: (Price_t+5 - Price_t) / Price_t
    df['Forward_5d_return'] = df['Close_SPY'].shift(-5) / df['Close_SPY'] - 1
    df['Direction_label'] = (df['Forward_5d_return'] > 0).astype(int)
    
    # 5. Clean and Save
    print(f"Shape before dropna: {df.shape}")
    
    # Identifying columns to keep (drop intermediate calculation columns if needed? No, keep all)
    
    # Drop rows with NaNs (due to lags/rolling)
    df_clean = df.dropna()
    print(f"Shape after dropna: {df_clean.shape}")
    
    if len(df_clean) == 0:
        raise ValueError("All data dropped! Check feature generation.")

    # 6. Scaling
    # Define columns NOT to scale
    no_scale_cols = ['Direction_label', 'Regime_binary', 'Regime_transition', 
                     'VIX_spike', 'VIX_extreme', 'Volume_spike', 'Regime', 'Forward_5d_return', 
                     'Regime_change'] # Also string 'Regime'
    
    # Select numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [col for col in numeric_cols if col not in no_scale_cols]
    
    print(f"Scaling {len(scale_cols)} features...")
    
    # Fit scaler on TRAINING PART ONLY (first 70%) to avoid leakage
    train_size = int(len(df_clean) * 0.7)
    scaler = StandardScaler()
    scaler.fit(df_clean.iloc[:train_size][scale_cols])
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Transform all data (we save the Scaled version? 
    # Creating a new dataframe with scaled values makes loading easier)
    df_scaled = df_clean.copy()
    df_scaled[scale_cols] = scaler.transform(df_clean[scale_cols])
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'features_engineered.csv')
    df_scaled.to_csv(output_path)
    print(f"Saved engineered features to {output_path}")

if __name__ == "__main__":
    run_feature_engineering()
