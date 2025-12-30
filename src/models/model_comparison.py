"""
Model Comparison Utilities

Tools to compare multiple models on the same test set and rank them by performance.
"""

import pandas as pd
import torch
from typing import Dict, List, Optional
from src.training.evaluate import evaluate_model
from src.models.baselines import BaselineModel

def compare_models(models_dict: Dict[str, any], 
                  test_loader, 
                  test_X_df: Optional[pd.DataFrame] = None, 
                  test_y_np: Optional[object] = None,
                  device: str = 'cpu') -> pd.DataFrame:
    """
    Compare multiple models (PyTorch and Baselines) on the same test set.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
                     (can be PyTorch nn.Module or BaselineModel)
        test_loader: PyTorch DataLoader for LSTM models
        test_X_df: DataFrame features for Baseline models
        test_y_np: Numpy array targets for Baseline models
        device: Device for PyTorch inference
    
    Returns:
        DataFrame with metrics for each model, sorted by accuracy
    """
    results = []
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        
        # Check if it's a PyTorch model
        if isinstance(model, torch.nn.Module):
            metrics = evaluate_model(model, test_loader, device)
            metrics['type'] = 'LSTM'
        # Check if it's a Baseline model
        elif isinstance(model, BaselineModel):
            if test_X_df is None or test_y_np is None:
                print(f"Skipping {name}: test_X_df and test_y_np required for baselines")
                continue
            metrics = model.evaluate(test_X_df, test_y_np)
            metrics['type'] = 'Baseline'
        else:
            print(f"Skipping {name}: Unknown model type {type(model)}")
            continue
            
        metrics['model'] = name
        results.append(metrics)
    
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'type', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    return df
