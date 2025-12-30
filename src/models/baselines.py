"""
Baseline Models for SPY Directional Prediction

These baselines establish a performance floor for the LSTM models to beat.
Expected accuracy for random baseline: ~50%
Target for LogisticRegression: 52-55%
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BaselineModel(ABC):
    """Abstract base class for all baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineModel':
        """Fit the model on training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities. Default: returns 0.5 for all.
        Override in subclasses that support probability estimation.
        """
        predictions = self.predict(X)
        # Convert to pseudo-probabilities
        return predictions.astype(float)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate model on test data.
        
        Returns:
            dict with accuracy, precision, recall, f1, and optionally auc
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'model': self.name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC only if we have valid probabilities (not all same value)
        if len(np.unique(y_proba)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auc'] = 0.5
        else:
            metrics['auc'] = 0.5
        
        return metrics


class RandomBaseline(BaselineModel):
    """
    Random predictions: 50% UP, 50% DOWN.
    Expected accuracy: ~50%
    """
    
    def __init__(self, random_state: int = 42):
        super().__init__("RandomBaseline")
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomBaseline':
        """No training needed for random baseline."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Random 0/1 predictions."""
        return self.rng.randint(0, 2, size=len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Random probabilities between 0 and 1."""
        return self.rng.random(size=len(X))


class MomentumBaseline(BaselineModel):
    """
    Momentum strategy: If SPY went UP in the last N days, predict UP.
    Uses the sum of recent returns as a signal.
    """
    
    def __init__(self, lookback_col: str = 'Return_MA_5'):
        super().__init__("MomentumBaseline")
        self.lookback_col = lookback_col
        self.threshold = 0.0  # Will be set during fit
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'MomentumBaseline':
        """
        Fit by finding optimal threshold on training data.
        For simplicity, use 0 as threshold (positive returns → UP).
        """
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """If average recent return is positive, predict UP (1)."""
        if isinstance(X, pd.DataFrame) and self.lookback_col in X.columns:
            signals = X[self.lookback_col].values
        else:
            # Fallback: use first column or assume it's already the signal
            signals = X[:, 0] if isinstance(X, np.ndarray) else X.iloc[:, 0].values
        
        return (signals > self.threshold).astype(int)


class MeanReversionBaseline(BaselineModel):
    """
    Mean reversion strategy: If SPY went UP recently, predict DOWN (expect reversal).
    Opposite of momentum.
    """
    
    def __init__(self, lookback_col: str = 'Return_MA_5'):
        super().__init__("MeanReversionBaseline")
        self.lookback_col = lookback_col
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'MeanReversionBaseline':
        """No complex fitting needed."""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """If average recent return is positive, predict DOWN (0) - expect reversal."""
        if isinstance(X, pd.DataFrame) and self.lookback_col in X.columns:
            signals = X[self.lookback_col].values
        else:
            signals = X[:, 0] if isinstance(X, np.ndarray) else X.iloc[:, 0].values
        
        # Opposite of momentum: positive recent returns → predict DOWN (0)
        return (signals <= 0).astype(int)


class NaivePersistenceBaseline(BaselineModel):
    """
    Naive persistence: Predict the same direction as the last observed 5-day return.
    If the most recent 5-day return was UP, predict UP.
    
    This is a simple but meaningful baseline - markets often have short-term momentum.
    """
    
    def __init__(self, return_col: str = 'Returns_SPY'):
        super().__init__("NaivePersistenceBaseline")
        self.return_col = return_col
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'NaivePersistenceBaseline':
        """No training needed."""
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict same direction as recent 5-day cumulative return.
        Uses Return_lag_1 through Return_lag_5 to compute recent return direction.
        """
        if isinstance(X, pd.DataFrame):
            # Sum of returns over last 5 days
            lag_cols = [f'Return_lag_{i}' for i in range(1, 6)]
            available_cols = [c for c in lag_cols if c in X.columns]
            
            if available_cols:
                cumulative_return = X[available_cols].sum(axis=1)
                return (cumulative_return > 0).astype(int).values
        
        # Fallback: use first column
        signals = X[:, 0] if isinstance(X, np.ndarray) else X.iloc[:, 0].values
        return (signals > 0).astype(int)


class LogisticRegressionBaseline(BaselineModel):
    """
    Logistic Regression baseline using scikit-learn.
    This is the main baseline to beat - typically achieves 52-55% accuracy.
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        super().__init__("LogisticRegressionBaseline")
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionBaseline':
        """Fit logistic regression on features."""
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 (UP)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Return feature coefficients sorted by absolute value."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance


def evaluate_all_baselines(X_train: pd.DataFrame, y_train: np.ndarray,
                           X_test: pd.DataFrame, y_test: np.ndarray,
                           feature_cols: list = None) -> pd.DataFrame:
    """
    Train and evaluate all baseline models.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training labels
        X_test: Test features (DataFrame)
        y_test: Test labels
        feature_cols: List of feature column names for LogisticRegression
    
    Returns:
        DataFrame with performance metrics for each baseline
    """
    # If feature_cols not provided, use numeric columns excluding target-like columns
    if feature_cols is None:
        exclude = ['Direction_label', 'Forward_5d_return', 'Regime']
        feature_cols = [c for c in X_train.columns if c not in exclude and X_train[c].dtype in ['float64', 'int64']]
    
    # Initialize baselines
    baselines = [
        RandomBaseline(),
        MomentumBaseline(),
        MeanReversionBaseline(),
        NaivePersistenceBaseline(),
        LogisticRegressionBaseline()
    ]
    
    results = []
    
    for baseline in baselines:
        print(f"Training {baseline.name}...")
        
        # LogisticRegression needs numeric features only
        if isinstance(baseline, LogisticRegressionBaseline):
            X_train_model = X_train[feature_cols]
            X_test_model = X_test[feature_cols]
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Fit and evaluate
        baseline.fit(X_train_model, y_train)
        metrics = baseline.evaluate(X_test_model, y_test)
        results.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Convert to DataFrame and sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    return results_df


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing baseline models...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    
    X_dummy = pd.DataFrame({
        'Return_MA_5': np.random.randn(n_samples) * 0.01,
        'Return_lag_1': np.random.randn(n_samples) * 0.01,
        'Return_lag_2': np.random.randn(n_samples) * 0.01,
        'Return_lag_3': np.random.randn(n_samples) * 0.01,
        'Return_lag_4': np.random.randn(n_samples) * 0.01,
        'Return_lag_5': np.random.randn(n_samples) * 0.01,
        'VIX_level': 20 + np.random.randn(n_samples) * 5,
    })
    y_dummy = np.random.randint(0, 2, n_samples)
    
    # Split
    split = int(0.7 * n_samples)
    X_train, X_test = X_dummy.iloc[:split], X_dummy.iloc[split:]
    y_train, y_test = y_dummy[:split], y_dummy[split:]
    
    # Evaluate
    results = evaluate_all_baselines(X_train, y_train, X_test, y_test)
    print("\nResults:")
    print(results.to_string(index=False))
