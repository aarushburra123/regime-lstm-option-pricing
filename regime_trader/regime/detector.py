"""Market regime detection."""
import pandas as pd
import numpy as np


class RegimeDetector:
    """Detects market volatility regimes."""
    
    def __init__(self, vix_threshold: float = 20.0):
        self.vix_threshold = vix_threshold
    
    def detect_vix_regime(self, vix_level: float) -> str:
        """Simple VIX threshold-based regime detection.
        
        Args:
            vix_level: Current VIX level
            
        Returns:
            'low_vol' or 'high_vol'
        """
        return 'high_vol' if vix_level >= self.vix_threshold else 'low_vol'
    
    def label_dataframe(self, df: pd.DataFrame, 
                        vix_col: str = 'Close_VIX') -> pd.DataFrame:
        """Add regime labels to dataframe.
        
        Args:
            df: DataFrame with VIX data
            vix_col: Name of VIX column
            
        Returns:
            DataFrame with 'Regime' column added
        """
        df = df.copy()
        df['Regime'] = df[vix_col].apply(self.detect_vix_regime)
        return df
