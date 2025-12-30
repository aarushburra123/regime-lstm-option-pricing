"""
Dataset utilities for Regime-Switching LSTM

Creates PyTorch-compatible datasets with proper time-based train/val/test splits.
Handles sequence creation for LSTM input and prevents forward-looking bias.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
from typing import Tuple, List, Optional


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series sequences.
    
    Creates sliding window sequences of features for LSTM input.
    Each sample is a tuple of (sequence, target) where:
    - sequence: (seq_length, num_features) tensor
    - target: scalar (0 or 1 for direction)
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 20, regime: Optional[np.ndarray] = None):
        """
        Args:
            features: 2D array of shape (n_samples, n_features)
            targets: 1D array of shape (n_samples,) with binary labels
            sequence_length: Number of timesteps in each sequence
            regime: Optional 1D array with regime labels for each sample
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.regime = torch.LongTensor(regime) if regime is not None else None
        
        # Number of valid sequences (need at least seq_length samples)
        self.n_sequences = len(features) - sequence_length + 1
        
        if self.n_sequences <= 0:
            raise ValueError(f"Not enough data for sequence_length={sequence_length}. "
                           f"Got {len(features)} samples, need at least {sequence_length}.")
    
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        The target corresponds to the LAST timestep in the sequence.
        """
        # Get sequence of features
        seq_start = idx
        seq_end = idx + self.sequence_length
        sequence = self.features[seq_start:seq_end]
        
        # Target is for the last timestep in the sequence
        target = self.targets[seq_end - 1]
        
        if self.regime is not None:
            regime = self.regime[seq_end - 1]
            return sequence, target, regime
        
        return sequence, target


def create_time_based_split(df: pd.DataFrame, 
                            train_ratio: float = 0.70,
                            val_ratio: float = 0.15,
                            forward_buffer: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/val/test split with forward-looking buffer.
    
    CRITICAL: The target at time t uses data from t+5 (forward-looking).
    So we need a buffer between splits to prevent data leakage.
    
    Args:
        df: DataFrame with features and target
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        forward_buffer: Number of rows to skip between splits (default 5 for 5-day forward target)
    
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    
    # Calculate split points with buffer
    train_end = int(n * train_ratio)
    val_start = train_end + forward_buffer  # Skip buffer rows
    val_end = int(n * (train_ratio + val_ratio))
    test_start = val_end + forward_buffer  # Skip buffer rows
    test_end = n - forward_buffer  # Exclude last rows (no complete target)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:test_end].copy()
    
    # Verify no date overlap
    if hasattr(df, 'index') and hasattr(df.index, 'max'):
        assert train_df.index.max() < val_df.index.min(), "Train/val overlap detected!"
        assert val_df.index.max() < test_df.index.min(), "Val/test overlap detected!"
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


def prepare_features_and_target(df: pd.DataFrame,
                                 target_col: str = 'Direction_label',
                                 exclude_cols: List[str] = None,
                                 scaler_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract features and target from DataFrame.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
        scaler_path: Path to pre-fitted scaler (for val/test data)
    
    Returns:
        features: 2D numpy array
        targets: 1D numpy array
        regime: 1D numpy array (or None if no Regime_binary column)
        feature_names: List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['Direction_label', 'Forward_5d_return', 'Regime']
    
    # Get feature columns (numeric only, excluding target-related columns)
    feature_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    
    # Get regime if available
    regime = None
    if 'Regime_binary' in df.columns:
        regime = df['Regime_binary'].values.astype(np.int64)
    
    return features, targets, regime, feature_cols


def create_dataloaders(df: pd.DataFrame,
                       sequence_length: int = 20,
                       batch_size: int = 32,
                       train_ratio: float = 0.70,
                       val_ratio: float = 0.15,
                       scaler_path: str = None,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train/val/test DataLoaders from DataFrame.
    
    This is the main entry point for preparing data for LSTM training.
    
    Args:
        df: DataFrame with features and target (from features_engineered.csv)
        sequence_length: Number of timesteps per sequence
        batch_size: Batch size for DataLoaders
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        scaler_path: Path to pre-fitted scaler (use this, DO NOT re-fit)
        num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, feature_names
    """
    # Split data
    train_df, val_df, test_df = create_time_based_split(
        df, train_ratio, val_ratio, forward_buffer=5
    )
    
    # Prepare features and targets
    train_X, train_y, train_regime, feature_names = prepare_features_and_target(train_df)
    val_X, val_y, val_regime, _ = prepare_features_and_target(val_df)
    test_X, test_y, test_regime, _ = prepare_features_and_target(test_df)
    
    print(f"\nFeature count: {len(feature_names)}")
    print(f"Sequence length: {sequence_length}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_X, train_y, sequence_length, train_regime)
    val_dataset = TimeSeriesDataset(val_X, val_y, sequence_length, val_regime)
    test_dataset = TimeSeriesDataset(test_X, test_y, sequence_length, test_regime)
    
    print(f"\nDataset sizes (after sequence creation):")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")
    print(f"  Test:  {len(test_dataset)} sequences")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, feature_names


def create_regime_specific_dataloaders(df: pd.DataFrame,
                                       sequence_length: int = 20,
                                       batch_size: int = 32,
                                       train_ratio: float = 0.70,
                                       val_ratio: float = 0.15) -> dict:
    """
    Create separate DataLoaders for low-vol and high-vol regimes.
    
    Used for training regime-specific expert models.
    
    Args:
        df: DataFrame with features, target, and Regime_binary column
    
    Returns:
        dict with keys 'low_vol' and 'high_vol', each containing
        (train_loader, val_loader, test_loader)
    """
    if 'Regime_binary' not in df.columns:
        raise ValueError("DataFrame must have Regime_binary column")
    
    # Split by regime
    low_vol_df = df[df['Regime_binary'] == 0].copy()
    high_vol_df = df[df['Regime_binary'] == 1].copy()
    
    print(f"Low Vol samples: {len(low_vol_df)}")
    print(f"High Vol samples: {len(high_vol_df)}")
    
    # Create loaders for each regime
    low_vol_loaders = create_dataloaders(
        low_vol_df, sequence_length, batch_size, train_ratio, val_ratio
    )
    
    high_vol_loaders = create_dataloaders(
        high_vol_df, sequence_length, batch_size, train_ratio, val_ratio
    )
    
    return {
        'low_vol': low_vol_loaders[:3],  # (train, val, test)
        'high_vol': high_vol_loaders[:3],
        'feature_names': low_vol_loaders[3]
    }


def load_engineered_features(data_dir: str = 'data') -> pd.DataFrame:
    """
    Load the engineered features CSV file.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        DataFrame with engineered features
    """
    filepath = os.path.join(data_dir, 'features_engineered.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Features file not found: {filepath}\n"
            "Please run notebooks/02_feature_engineering.ipynb first."
        )
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


if __name__ == "__main__":
    # Test with dummy data
    print("Testing dataset utilities...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 200
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')
    df_dummy = pd.DataFrame({
        'Close_SPY': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'Returns_SPY': np.random.randn(n_samples) * 0.01,
        'Return_lag_1': np.random.randn(n_samples) * 0.01,
        'Return_lag_2': np.random.randn(n_samples) * 0.01,
        'VIX_level': 20 + np.random.randn(n_samples) * 5,
        'Regime_binary': np.random.randint(0, 2, n_samples),
        'Direction_label': np.random.randint(0, 2, n_samples),
        'Forward_5d_return': np.random.randn(n_samples) * 0.02,
    }, index=dates)
    
    # Test split
    print("\n" + "="*50)
    print("Testing time-based split:")
    train_df, val_df, test_df = create_time_based_split(df_dummy)
    
    # Test DataLoader creation
    print("\n" + "="*50)
    print("Testing DataLoader creation:")
    train_loader, val_loader, test_loader, feature_names = create_dataloaders(
        df_dummy, sequence_length=10, batch_size=16
    )
    
    # Test a batch
    print("\n" + "="*50)
    print("Testing batch retrieval:")
    for batch in train_loader:
        if len(batch) == 3:
            sequences, targets, regimes = batch
            print(f"  Sequence shape: {sequences.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Regimes shape: {regimes.shape}")
        else:
            sequences, targets = batch
            print(f"  Sequence shape: {sequences.shape}")
            print(f"  Targets shape: {targets.shape}")
        break
    
    print("\n✓ All tests passed!")
