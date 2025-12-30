"""Tests for regime detection."""
import pandas as pd
import numpy as np
from regime_trader.regime.detector.py import RegimeDetector


def test_vix_regime_detection():
    """Test basic VIX threshold regime detection."""
    detector = RegimeDetector(vix_threshold=20.0)
    
    # Test low vol
    assert detector.detect_vix_regime(15.0) == 'low_vol'
    
    # Test high vol
    assert detector.detect_vix_regime(25.0) == 'high_vol'
    
    # Test boundary
    assert detector.detect_vix_regime(20.0) == 'high_vol'


def test_dataframe_labeling():
    """Test regime labeling on DataFrame."""
    detector = RegimeDetector(vix_threshold=20.0)
    
    # Create test data
    df = pd.DataFrame({
        'Close_VIX': [10, 15, 25, 30, 18]
    })
    
    # Label
    result = detector.label_dataframe(df)
    
    # Verify
    assert 'Regime' in result.columns
    assert result.iloc[0]['Regime'] == 'low_vol'
    assert result.iloc[2]['Regime'] == 'high_vol'
    assert len(result) == len(df)
