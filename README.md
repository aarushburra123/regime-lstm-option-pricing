# Regime-Conditional LSTM for Option Pricing

**Research Project: Aarush Burra, Academies of Loudoun (AOS), 2026

---

## Research Question

I want to study the effect of **regime-conditional LSTM neural networks** on **option pricing accuracy for short-dated S&P 500 options** using **historical options data (2020-2024) and Monte Carlo simulation**.

---

## Background

Accurate option pricing is critical for hedge funds, market makers, and risk managers. Traditional models like Black-Scholes assume constant volatility, which fails during market crises (e.g., March 2020 COVID crash). This project compares four volatility forecasting methods and evaluates their impact on option pricing accuracy:

1. **GARCH(1,1)** – Standard econometric baseline
2. **HAR-RV** – Heterogeneous Autoregressive Realized Volatility
3. **Single-Regime LSTM** – One LSTM trained on all data
4. **Regime-Conditional LSTM** – Separate LSTMs for low-volatility (VIX < 20) and high-volatility (VIX ≥ 20) periods

**Novel Contribution:** Most research measures volatility forecast accuracy (RMSE, MAE). This project directly measures **option pricing accuracy** (MAPE vs. market prices) and quantifies when regime-conditional models provide value.

---

## Project Status

**Current Phase:** Data Pipeline (Week 1)

**Timeline:**
- ✅ Dec 2025: Project setup, environment configuration
- 🔄 Jan 2026: Data collection and cleaning
- ⏳ Feb-Mar 2026: Baseline models (GARCH, HAR-RV)
- ⏳ Apr-May 2026: LSTM models (single-regime, regime-conditional)
- ⏳ Jun-Jul 2026: Monte Carlo option pricing and backtesting
- ⏳ Aug 2026: Results analysis and writeup

---

## Repository Structure

