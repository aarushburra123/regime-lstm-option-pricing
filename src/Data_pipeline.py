"""
Data Pipeline for SPY Options and VIX Historical Data

This module handles downloading, cleaning, and preparing data for my future research project.

#Comments are AI assisted, but written and reviewed by me
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataPipeline:
    """
    Handles data preprocessing for SPY options and VIX.
    """
    
    def __init__(self, start_date='2020-01-01', end_date='2024-12-31', data_dir='../data'):
        """
        Initialize data pipeline.
        
        Args:
            start_date (str): Start date for data collection (YYYY-MM-DD)
            end_date (str): End date for data collection (YYYY-MM-DD)
            data_dir (str): Directory to save data files
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def download_spy_prices(self):
        """
        Download SPY daily price data.
        
        Returns:
            pd.DataFrame: SPY price data with columns [Open, High, Low, Close, Volume]
        """
        print(f"Downloading SPY price data from {self.start_date} to {self.end_date}...")
        
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
        
        # Calculate returns
        # Handle potential MultiIndex columns
        if isinstance(spy.columns, pd.MultiIndex):
            close_col = [col for col in spy.columns if 'Close' in str(col)][0]
        else:
            close_col = 'Close'

        spy['Returns'] = spy[close_col].pct_change()
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, 'spy_prices.csv')
        spy.to_csv(filepath)
        print(f"✓ Saved SPY prices to {filepath}")
        
        return spy
    
    def download_vix_data(self):
        """
        Download VIX (volatility index) data.
        
        Returns:
            pd.DataFrame: VIX data
        """
        print(f"Downloading VIX data from {self.start_date} to {self.end_date}...")
        
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, 'vix.csv')
        vix.to_csv(filepath)
        print(f"✓ Saved VIX data to {filepath}")
        
        return vix
    
    def get_realized_volatility(self, returns, window=30):
        """
        Calculate realized volatility from returns.
        
        Args:
            returns (pd.Series): Daily returns
            window (int): Rolling window size
            
        Returns:
            pd.Series: Annualized realized volatility
        """
        return returns.rolling(window).std() * np.sqrt(252)
    
    def merge_data(self, spy, vix):
        """
        Merge SPY and VIX data, align dates.
        
        Args:
            spy (pd.DataFrame): SPY price data
            vix (pd.DataFrame): VIX data
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        print("Merging SPY and VIX data...")
        
        # Flatten column names if they're MultiIndex (from yfinance)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                          for col in spy.columns.values]
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                          for col in vix.columns.values]
        
        print(f"  SPY columns after flattening: {spy.columns.tolist()}")
        print(f"  VIX columns after flattening: {vix.columns.tolist()}")
        
        # Select relevant columns
        # SPY should have 'Close' or 'Close_SPY' and 'Returns'
        spy_close_col = [col for col in spy.columns if 'Close' in col][0]
        # Find the Returns column (might be 'Returns' or have a suffix after flattening)
        spy_returns_col = [col for col in spy.columns if 'Returns' in col][0]
        
        spy_subset = spy[[spy_close_col, spy_returns_col]].copy()
        spy_subset.columns = ['Close_SPY', 'Returns_SPY']
        
        # VIX should have 'Close' or 'Close_^VIX'
        vix_close_col = [col for col in vix.columns if 'Close' in col][0]
        vix_subset = vix[[vix_close_col]].copy()
        vix_subset.columns = ['Close_VIX']
        
        # Merge on date index
        combined = pd.merge(
            spy_subset,
            vix_subset,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        print(f"  Columns after merge: {combined.columns.tolist()}")
        
        # Add regime labels
        combined['Regime'] = combined['Close_VIX'].apply(
            lambda x: 'High' if x >= 20 else 'Low'
        )
        
        # Add realized volatility
        combined['RealizedVol_30d'] = self.get_realized_volatility(combined['Returns_SPY'], 30)
        
        # Drop NaN
        combined = combined.dropna()
        
        # Save
        filepath = os.path.join(self.data_dir, 'spy_vix_combined.csv')
        combined.to_csv(filepath)
        print(f"✓ Saved combined data to {filepath}")
        print(f"✓ Total rows: {len(combined)}")
        print(f"✓ Date range: {combined.index[0]} to {combined.index[-1]}")
        
        # Print regime distribution
        regime_counts = combined['Regime'].value_counts()
        print(f"\nRegime Distribution:")
        print(f"  Low Volatility (VIX < 20): {regime_counts.get('Low', 0)} days ({regime_counts.get('Low', 0)/len(combined)*100:.1f}%)")
        print(f"  High Volatility (VIX ≥ 20): {regime_counts.get('High', 0)} days ({regime_counts.get('High', 0)/len(combined)*100:.1f}%)")
        
        return combined
    
    def run_full_pipeline(self):
        """
        Execute complete data pipeline.
        
        Returns:
            pd.DataFrame: Final combined dataset
        """
        print("=" * 60)
        print("Starting Data Pipeline")
        print("=" * 60)
        
        # Download data
        spy = self.download_spy_prices()
        vix = self.download_vix_data()
        
        # Merge
        combined = self.merge_data(spy, vix)
        
        print("=" * 60)
        print("Data Pipeline Complete!")
        print("=" * 60)
        
        return combined


if __name__ == "__main__":
    # Run pipeline
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level to project root
    data_dir = os.path.join(project_root, 'data')
    
    pipeline = DataPipeline(
        start_date='2020-01-01',
        end_date='2024-12-31',
        data_dir=data_dir  # ← Now it will always use the correct path
    )

    
    data = pipeline.run_full_pipeline()
    
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nData info:")
    print(data.info())