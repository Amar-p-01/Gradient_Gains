# Gradient Gains 📈

### Hybrid Machine Learning System for Stock Price Forecasting

**Gradient Gains** is an interactive machine-learning based financial forecasting application that predicts short-term stock prices using a **hybrid ensemble of Gradient Boosting and Extra Trees regressors**. The system combines historical market data, temporal lag features, return statistics, volatility, and technical indicators to generate **3-day recursive forecasts** for US stocks, Indian NSE stocks, and cryptocurrencies.

The project is built with **Python, Scikit-Learn, Streamlit, yFinance, Pandas, NumPy, and Plotly**.

---

## 🚀 Key Features

- **Multi-market support** for US stocks, Indian NSE stocks, and cryptocurrencies
- **Live market data** fetched using the `yfinance` API
- **21-day lag-based feature engineering** to capture recent price dynamics
- Feature extraction using:
  - Log returns
  - 5/10/20-day moving averages
  - 5-day volatility
  - Historical closing prices
- **Hybrid ensemble forecasting** using:
  - Gradient Boosting Regressor
  - Extra Trees Regressor
- **Weighted Voting Regressor** with a 2:1 GBR-to-ETR weighting
- **Recursive multi-step forecasting** for the next 3 trading days
- **RSI-based technical signals** for BUY/HOLD/SELL interpretation
- Interactive **Streamlit dashboard**
- Interactive **Plotly visualizations**
- Automatic handling of weekends and invalid ticker/API failures

---

## 🧠 Machine Learning Approach

### 1. Data Collection

Historical market data is retrieved through `yfinance`, using approximately two years of historical price data.

The pipeline transforms raw closing prices into supervised learning samples using a **21-day sliding window**:

```text
Close(t-20) ... Close(t-2) Close(t-1) → Close(t)
```

This allows the model to learn short-term temporal dependencies in price movements.

### 2. Feature Engineering

Additional market features are constructed to provide information beyond raw prices:

| Feature | Purpose |
|---|---|
| Lagged Prices | Capture recent price patterns |
| Log Returns | Represent relative price movement |
| Moving Averages | Capture short/medium-term trends |
| 5-Day Volatility | Represent recent market risk |
| RSI | Identify overbought/oversold conditions |

---

## 🤖 Hybrid Ensemble Model

The forecasting model combines two complementary tree-based algorithms:

### Gradient Boosting Regressor

Gradient Boosting sequentially learns from the errors made by previous trees, allowing it to capture nonlinear relationships and reduce prediction bias.

### Extra Trees Regressor

Extra Trees introduces additional randomization when constructing decision trees, producing a diverse ensemble that helps reduce variance and improve robustness.

### Weighted Voting

The final prediction is calculated using a weighted combination:

```text
Final Prediction = (2 × GBR Prediction + 1 × ETR Prediction) / 3
```

This gives greater influence to Gradient Boosting while using Extra Trees as a stabilizing component.

---

## 🔄 Recursive 3-Day Forecasting

Because the model predicts one step at a time, multi-day forecasting is performed recursively.

```text
Historical Data
      ↓
Predict Day 1
      ↓
Append Prediction
      ↓
Recalculate Features
      ↓
Predict Day 2
      ↓
Append Prediction
      ↓
Predict Day 3
```

The system also accounts for **non-trading days such as weekends** when generating future prediction dates.

---

## 📊 Technical Signal Engine

The application calculates the **14-day Relative Strength Index (RSI)** as an additional technical-analysis signal.

```text
RSI < 30       → BUY
30 ≤ RSI ≤ 70  → HOLD
RSI > 70       → SELL
```

The RSI signal is presented alongside the ML forecast rather than being used as a direct model prediction feature.

---

## 🖥️ Interactive Dashboard

The Streamlit application provides:

- Ticker selection
- Quick-access stock/crypto selections
- Current market price
- Predicted next closing price
- Expected percentage movement
- 3-day forecast
- Historical vs predicted price visualization
- RSI and trading signal
- Interactive Plotly charts

---

## ⚙️ Model Configuration

Example model configuration:

```text
Gradient Boosting:
    n_estimators = 300
    learning_rate = 0.03

Extra Trees:
    n_estimators = 400
    max_depth = 7

Ensemble:
    GBR weight = 2
    ETR weight = 1
```

The relatively constrained Extra Trees depth is intended to limit overfitting while maintaining model diversity.

---

## 🏗️ Technology Stack

**Programming:** Python

**Machine Learning:** Scikit-Learn

**Data:** Pandas, NumPy, yFinance

**Visualization:** Plotly

**Web Application:** Streamlit

---

## ⚠️ Limitations

Financial markets are highly stochastic and influenced by information that cannot be inferred from historical prices alone. Therefore, the forecasts should be treated as **experimental model outputs rather than financial advice**.

The current system also does not incorporate:

- News sentiment
- Fundamental financial metrics
- Macroeconomic indicators
- Transaction costs
- Full historical backtesting
- Market regime detection

---

## 🔮 Future Improvements

Potential extensions include:

- **News and social-media sentiment analysis**
- MACD, Bollinger Bands and additional technical indicators
- Automated **walk-forward backtesting**
- Hyperparameter optimization
- LSTM/Transformer-based temporal models
- Fundamental and macroeconomic features
- Portfolio-level forecasting
- Model performance tracking over time

---

## 🎯 Project Objective

The primary objective of Gradient Gains is to demonstrate how **ensemble machine learning, temporal feature engineering, and recursive forecasting** can be combined into an end-to-end financial prediction system with an accessible interactive interface.

> **Note:** This project is intended for educational and experimental purposes and does not constitute financial advice.
