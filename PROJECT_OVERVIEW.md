PROJECT OVERVIEW: Regime-Switching LSTM for SPY Directional Prediction

Project Goal
Build a machine learning system that predicts the 5-day forward direction (UP/DOWN) of SPY (S&P 500 ETF) using a regime-switching mixture-of-experts LSTM architecture. The hypothesis is that market behavior differs fundamentally between low and high volatility regimes, and separate expert models will outperform a single unified model.

Technical Approach
Problem Formulation:
Task: Binary classification (predict if SPY will be higher or lower in 5 trading days)

Input features: Historical SPY prices, returns, VIX levels, rolling statistics, regime indicators

Target variable: Direction_label = 1 if SPY goes up in next 5 days, 0 if down

Evaluation metric: Classification accuracy, precision/recall, Sharpe ratio of resulting trading strategy

Architecture: Mixture-of-Experts (MoE)
text
Input Features
    ↓
[Gating Network] → Regime Weights (Low Vol %, High Vol %)
    ↓
Expert 1 (LSTM) ← Trained on Low Vol data (VIX < 20)
Expert 2 (LSTM) ← Trained on High Vol data (VIX ≥ 20)
    ↓
Weighted Combination → Final Prediction (UP/DOWN)
Regime Definition:
Low Volatility Regime: VIX < 20 (50.7% of days in dataset)

High Volatility Regime: VIX ≥ 20 (49.3% of days in dataset)

Current Project Status
✅ Completed (Phase 1: Data Infrastructure)
Data Pipeline (src/Data_pipeline.py)

Downloads SPY and VIX data from Yahoo Finance (2020-2024)

Calculates daily returns and 30-day realized volatility

Classifies each day into Low/High volatility regime

Outputs: spy_prices.csv, vix.csv, spy_vix_combined.csv

Dataset: 1,227 trading days (2020-02-14 to 2024-12-30)

Exploratory Data Analysis (notebooks/01_data_explore.ipynb)

Visualized SPY price movements colored by regime

Plotted VIX timeline (showing COVID crash spike to 85)

Analyzed VIX vs Realized Volatility correlation (0.799)

Confirmed regime balance is roughly 50/50

📂 Current File Structure:
text
regime-lstm-option-pricing/
├── src/
│   └── Data_pipeline.py          ✅ Complete
├── notebooks/
│   └── 01_data_explore.ipynb     ✅ Complete
├── data/
│   ├── spy_prices.csv            ✅ Generated
│   ├── vix.csv                   ✅ Generated
│   └── spy_vix_combined.csv      ✅ Generated
└── results/
    ├── spy_price_by_regime.png   ✅ Generated
    ├── vix_over_time.png         ✅ Generated
    └── vix_vs_realized_vol.png   ✅ Generated
Next Steps (Phase 2: Feature Engineering - IMMEDIATE)
Create: notebooks/02_feature_engineering.ipynb
Features to Engineer:

Lag Features (capture momentum/mean reversion):

Return_lag_1 through Return_lag_10 (past 10 days of returns)

Price_lag_1 through Price_lag_5

Rolling Statistics (capture trends):

MA_5, MA_20, MA_60 (moving averages)

Vol_20d, Vol_60d (rolling volatility)

Return_MA_5, Return_MA_20 (average recent returns)

VIX Features (volatility regime info):

VIX_level (current VIX)

VIX_change (daily change in VIX)

VIX_MA_10 (10-day moving average)

VIX_pct_change (percentage change)

Regime Features:

Regime_binary (0=Low, 1=High)

Days_in_regime (consecutive days in current regime)

Regime_transition (1 if regime changed today, 0 otherwise)

Volume Features:

Volume_ratio (today's volume / 20-day avg volume)

Volume_MA_20

Target Variable:

Forward_5d_return = SPY_close(t+5) / SPY_close(t) - 1

Direction_label = 1 if Forward_5d_return > 0, else 0

Output: data/features_engineered.csv (ready for modeling)

Phase 3: Baseline Models (After Feature Engineering)
Create: src/models/baselines.py
Implement simple baselines to establish performance floor:

RandomBaseline: Predict 0 or 1 randomly (expected accuracy: 50%)

MomentumBaseline: If SPY up last 5 days → predict UP

MeanReversionBaseline: If SPY up last 5 days → predict DOWN

LogisticRegressionBaseline: Simple logistic regression on features

GARCHBaseline: Use GARCH(1,1) volatility forecast for direction

Success Criteria: Beat 50% random baseline, establish target accuracy (aim for 52-55%)

Phase 4: LSTM Architecture (Q1 2026)
Create: src/models/regime_lstm.py
Components:

LowVolExpertLSTM

2-layer LSTM (64-32 hidden units)

Trained ONLY on Low Vol regime data

Input: sequence of last N days of features

Output: probability of UP direction

HighVolExpertLSTM

Same architecture as LowVolExpert

Trained ONLY on High Vol regime data

Hypothesis: different patterns in high vol periods

GatingNetwork

Small feedforward network

Input: current regime features (VIX level, recent returns)

Output: weights for each expert [w_low, w_high] (sum to 1)

Learns which expert to trust based on current conditions

MixtureOfExpertsModel

Combines expert predictions using gating weights

Final_prediction = w_low * LowExpert + w_high * HighExpert

End-to-end trainable

Training Strategy:

Phase 1: Pre-train experts separately on regime-specific data

Phase 2: Freeze experts, train gating network

Phase 3: Fine-tune entire system end-to-end

Phase 5: Training Pipeline (Q1 2026)
Create: src/training/train.py and src/training/evaluate.py
Train/Val/Test Split (TIME-BASED, no data leakage):

Train: 2020-02 to 2022-12 (70% of data)

Validation: 2023-01 to 2023-06 (15%)

Test: 2023-07 to 2024-12 (15%)

Hyperparameters to Tune:

Sequence length (lookback window: 10, 20, 30, 60 days)

LSTM hidden units (32, 64, 128)

Number of LSTM layers (1, 2, 3)

Dropout rate (0.2, 0.3, 0.5)

Learning rate (1e-3, 1e-4)

Batch size (32, 64, 128)

Evaluation Metrics:

Classification accuracy (target: > 53%)

Precision, Recall, F1 score

ROC-AUC

Trading strategy metrics:

Sharpe ratio (annualized risk-adjusted return)

Maximum drawdown

Win rate

Profit factor

Phase 6: Backtesting & Strategy (Q2 2026)
Create: src/backtesting/strategy.py
Simple Directional Trading Strategy:

If model predicts UP (prob > 0.55) → Go long SPY

If model predicts DOWN (prob > 0.55) → Stay in cash (or short)

If uncertain (prob between 0.45-0.55) → No position

Risk Management:

Position sizing based on prediction confidence

Stop-loss rules

Maximum drawdown limits

Performance Comparison:

LSTM MoE vs Baselines

LSTM MoE vs Buy-and-Hold SPY

LSTM MoE vs regime-naive single LSTM

Phase 7: Production Deployment (Q3-Q4 2026)
Real-time inference API (FastAPI or Flask)

Daily data pipeline (auto-update with latest data)

Model versioning and monitoring

Live paper trading (Alpaca or Interactive Brokers API)

Dashboard (Streamlit/Gradio for predictions and performance tracking)

Key Research Questions to Validate
Does regime-switching improve predictions?

Compare MoE vs single LSTM trained on all data

What features matter most?

Feature importance analysis

Ablation studies (remove features one by one)

Is the VIX=20 threshold optimal?

Try dynamic thresholds (percentiles, HMM-based regime detection)

How far ahead can we predict?

Test 1-day, 3-day, 5-day, 10-day horizons

Does the model generalize across market cycles?

Performance in bull markets vs bear markets vs sideways

Success Criteria (Minimum Viable Results)
Academic success: Accuracy > 52% on out-of-sample test set

Practical success: Positive Sharpe ratio (> 0.5) in backtesting

Regime hypothesis validation: MoE outperforms single LSTM by > 1% accuracy

Technical Stack
Data: Python (pandas, yfinance)

ML Framework: PyTorch (LSTM implementation)

Visualization: matplotlib, seaborn

Experiment tracking: Weights & Biases or MLflow

Backtesting: backtrader or vectorbt

Deployment: FastAPI, Docker

Timeline (2025-2026)
Now - Jan 2026: Feature engineering, baseline models

Jan - Mar 2026: LSTM development and initial training

Apr - Jun 2026: Hyperparameter tuning, regime analysis

Jul - Sep 2026: Backtesting and strategy development

Oct - Dec 2026: Production deployment and live paper trading
