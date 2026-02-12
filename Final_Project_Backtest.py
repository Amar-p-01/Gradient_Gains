import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP FUNCTIONS (Same as App)
# ==========================================
def clean_columns(df):
    df.columns = [col.lower() for col in df.columns]
    return df

def create_lags(df):
    df = df.copy()
    target_cols = ["open", "high", "low", "close", "volume"]
    for col in target_cols:
        if col in df.columns:
            for i in range(21):
                df[f"{col}_lag_{i}"] = df[col].shift(i)
    return df

def create_features(df):
    # Log Returns
    for i in [1, 5, 10, 20]:
        if f"close_lag_{i}" in df.columns:
            df[f"logret_{i}"] = np.log(df["close_lag_0"] / df[f"close_lag_{i}"])
    # Moving Averages
    for i in [5, 10, 20]:
        cols = [f"close_lag_{j}" for j in range(i) if f"close_lag_{j}" in df.columns]
        if cols:
            ma = df[cols].mean(axis=1)
            df[f"price_ma_{i}"] = df["close_lag_0"] / ma
    # Volatility
    cols = [f"close_lag_{i}" for i in range(5) if f"close_lag_{i}" in df.columns]
    if cols:
        df["vol5"] = df[cols].std(axis=1)
    return df

# ==========================================
# 2. CALCULATION ENGINE (New XGB+RF Logic)
# ==========================================
def calculate_single_mape(ticker, test_days=30):
    try:
        # Fetch Data (3 Years for better training for XGBoost)
        df = yf.download(ticker, period="3y", interval="1d", progress=False)
        
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        df = clean_columns(df)
        df = create_lags(df)
        df = create_features(df)
        
        # Target
        df["target_price"] = df["close_lag_0"].shift(-1)
        df["y"] = np.log(df["target_price"] / df["close_lag_0"])
        
        # Features
        features = []
        for i in range(10):
            if f"close_lag_{i}" in df.columns: features.append(f"close_lag_{i}")
        for i in [1, 5, 10, 20]:
            if f"logret_{i}" in df.columns: features.append(f"logret_{i}")
        for i in [5, 10, 20]:
            if f"price_ma_{i}" in df.columns: features.append(f"price_ma_{i}")
        if "vol5" in df.columns: features.append("vol5")
        
        df_clean = df.dropna().reset_index(drop=True)
        if len(df_clean) < 200: return None # Need more data for XGB

        predictions = []
        actuals = []
        
        # Walk-Forward Validation (Last 30 Days)
        test_indices = range(len(df_clean) - test_days, len(df_clean))
        
        for i in test_indices:
            # Training Window: Last 300 days (XGBoost needs more data than GradientBoosting)
            train_subset = df_clean.iloc[i-300 : i]
            
            X_train = train_subset[features]
            y_train = train_subset["y"]
            
            X_test = df_clean.iloc[[i]][features]
            actual_price = df_clean.iloc[i]["target_price"]
            
            # --- NEW MODEL ARCHITECTURE ---
            xgb_model = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=5, 
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
            rf_model = RandomForestRegressor(
                n_estimators=400, max_depth=10, min_samples_leaf=5, 
                random_state=42, n_jobs=-1
            )
            model = VotingRegressor([("xgb", xgb_model), ("rf", rf_model)], weights=[2, 1])
            
            model.fit(X_train, y_train)
            
            pred_log_ret = model.predict(X_test)[0]
            current_close = df_clean.iloc[i]["close_lag_0"]
            predicted_price = current_close * np.exp(pred_log_ret)
            
            predictions.append(predicted_price)
            actuals.append(actual_price)
            
        # Calc MAPE
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        return mape

    except Exception as e:
        return None

# ==========================================
# 3. TABLE PRINTER
# ==========================================
def print_grid_table(df):
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    w_stock = 18
    w_mape = 12
    h_line = "+" + "-"*(w_stock+2) + "+" + "-"*(w_mape+2) + "+"
    
    print(h_line)
    print(f"| {BOLD}{'STOCK NAME'.ljust(w_stock)}{RESET} | {BOLD}{'MAPE (%)'.ljust(w_mape)}{RESET} |")
    print(h_line.replace("-", "="))
    
    for _, row in df.iterrows():
        stock = row['Stock']
        mape = row['MAPE (%)']
        
        # Color Logic: XGBoost is sharper, so we expect better accuracy
        if isinstance(mape, (int, float)) and mape < 1.8:
            mape_str = f"{GREEN}{mape:.3f}{RESET}"
        else:
            mape_str = f"{mape}"
            
        print(f"| {stock.ljust(w_stock)} | {str(mape_str).ljust(w_mape + (len(mape_str)-len(str(mape))))} |")
        print(h_line)

# ==========================================
# 4. EXECUTION
# ==========================================
stocks_to_test = [
    "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "TATASTEEL.NS",
    "BTC-USD"
]

print(f"\nComputing Accuracy for New XGBoost + RF Model (Last 30 Days)...\n")
results = []

for ticker in stocks_to_test:
    mape = calculate_single_mape(ticker, test_days=30)
    if mape is not None:
        print(f"✅ {ticker} Done")
        results.append({"Stock": ticker, "MAPE (%)": round(mape, 3)})
    else:
        print(f"❌ {ticker} Failed")
        results.append({"Stock": ticker, "MAPE (%)": "Error"})

df_results = pd.DataFrame(results).sort_values(by="MAPE (%)", key=lambda x: pd.to_numeric(x, errors='coerce')).reset_index(drop=True)

print("\n\nFINAL RANKING (XGBoost + Random Forest):")
print_grid_table(df_results)