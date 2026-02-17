import pandas as pd
import numpy as np
import os 

#loading raw data from csv
# return df of that stock
def load_stock_data(ticker, data_folder = "stock_data"):
    file_path = os.path.join(data_folder, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    else:
        print(f"File not found: {file_path}")
        return None

# read tickers from file
def get_tickers(filename):
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file.readlines() if line.strip() and not line.startswith("#")]
    except Exception as e:
        print(f"Error while reading tickers from {filename}: {e}")
        return []


#============== Feature Engineering ==============

#- Trends
def trend_indicator(df, window_size=20):
    df[f"SMA_{window_size}"] = df["Close"].rolling(window=window_size).mean()  # Simple Moving Average
    df[f"EMA_{window_size}"] = df["Close"].ewm(span=window_size, adjust=False).mean() # Exponential Moving Average
    df["MACD"] = df["Close"].ewm(span = 12).mean() - df["Close"].ewm(span = 26).mean() # Moving Average Convergence Divergence
    return df
    
# - Momentum
def momentum_indicators(df): # 2 week
    gain = df["Close"].diff().apply(lambda x: max(x,0)).rolling(14).mean()
    loss = df["Close"].diff().apply(lambda x: abs(min(x,0))).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs)) # Relative Strength Index
    df["MOM"] = df["Close"].diff(periods=10) # Momentum
    return df

# - Volatility
def volatility_indicators(df, window_size=20):
   rolling_std = df["Close"].rolling(window=window_size).std()
   rolling_mean = df["Close"].rolling(window=window_size).mean()
   df["Bollinger_Bands_Upper"] = rolling_mean + 2 * rolling_std # Bollinger Bands Upper
   df["Bollinger_Bands_Lower"] = rolling_mean - 2 * rolling_std # Bollinger Bands Lower
   df["ATR_14"] = df[["High", "Low", "Close"]].apply( 
    lambda x: max(x["High"] - x["Low"], abs(x["High"] - x["Close"]), abs(x["Low"] - x["Close"])), axis=1
    ).rolling(window=14).mean() # Average True Range
   return df

# - Volume
def volume_indicators(df):
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["OBV"] = (df["Volume"] * ((df["Close"].diff() > 0).astype(int) - (df["Close"].diff() < 0).astype(int))).cumsum()
    return df

# - Lag Returns
def compute_lag_returns(df, lag_periods= [1,5,10]):
    for lag in lag_periods:
        df[f"Lag_{lag}"] = df["Close"].shift(lag)
    return df

# - combine all features
def all_features(df):
    df = trend_indicator(df)
    df = momentum_indicators(df)
    df = volatility_indicators(df)
    df = volume_indicators(df)
    df = compute_lag_returns(df)
    return df

# -- ALIAS FOR MODEL PIPELINE --
#def recalc_indicators(df):
#   return all_features(df)

# ============== DATA PROCESSING =================

def save_transformed_data(df, ticker, output_folder = "processed_data"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f"{ticker}_features.csv")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

def process_stock_data(ticker):
    df = load_stock_data(ticker)
    if df is not None:
        df = all_features(df)
        df = df.dropna()
        save_transformed_data(df, ticker)
    else:
        print(f"Failed to process data for {ticker}")

def main():
    output_folder = "processed_data"
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            print(f"Cleared output folder: {output_folder}")
    
    tickers = get_tickers("tickers.txt")
    if tickers:
        for ticker in tickers:
            print(f"Processing {ticker}")
            process_stock_data(ticker)
    else:
        print("No tickers found in tickers.txt")

if __name__ == "__main__":
    main()