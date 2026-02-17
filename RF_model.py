import pandas as pd
import numpy as np
import os
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

US_MARKET_HOLIDAYS_2025 = pd.to_datetime([
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
    '2025-05-26', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
])

def adjust_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except Exception:
        parts = date_str.split('-')
        if len(parts) != 3:
            return None
        try:
            year, month, day = map(int, parts)
            day = min(day, calendar.monthrange(year, month)[1])
            return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
        except Exception:
            return None

    
def get_trading_days(start_date, end_date):
    start_dt = adjust_date(start_date)
    end_dt = adjust_date(end_date)
    if start_dt is None or end_dt is None or start_dt > end_dt:
        return pd.DatetimeIndex([])
    all_bdays = pd.date_range(start=start_dt, end=end_dt, freq='B')
    return pd.DatetimeIndex([d for d in all_bdays if d not in US_MARKET_HOLIDAYS_2025])


def train_model_for_ticker(ticker, train_start, train_end, test_size=0.2, random_state=42):
    csv_path = f'processed_data/{ticker}_features.csv'
    df = pd.read_csv(csv_path)

    date_col = next((col for col in df.columns if col.strip().lower() == "date"), None)
    if not date_col:
        raise ValueError(f"Missing 'date' column in {csv_path}")

    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    df['target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    df_for_training = df.copy()
    df_for_training.drop(columns=["date"], inplace=True)
    X = df_for_training.select_dtypes(include=[np.number]).drop(columns=['target'])
    y = df_for_training['target']
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, df, mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred), feature_columns

def simulate_future_ohlcv(last_row):
    close = last_row['Close']
    open_price = close + np.random.normal(0, 0.2)
    high = max(open_price, close) + abs(np.random.normal(0.3, 0.1))
    low = min(open_price, close) - abs(np.random.normal(0.3, 0.1))
    volume = last_row['Volume'] * np.random.uniform(0.95, 1.05)
    return open_price, high, low, volume

def predict_iteratively(ticker, model, df, trading_days, recalc_indicators_fn, feature_columns):
    if 'date' not in df.columns:
        print(f"[{ticker}] No date column found.")
        return
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    last_known_date = df['date'].max()
    future_days = [d for d in trading_days if pd.notna(last_known_date) and d > last_known_date]
    if not future_days:
        print(f"[{ticker}] No future trading days.")
        return

    rolling_window = df.tail(20).copy()
    if 'target' in rolling_window.columns:
        rolling_window.drop(columns=['target'], inplace=True)

    predictions = []
    for day in future_days:
        last_row = rolling_window.iloc[-1].copy()
        open_price, high, low, volume = simulate_future_ohlcv(last_row)
        new_row = last_row.copy()
        new_row['date'] = day
        new_row['Open'] = open_price
        new_row['High'] = high
        new_row['Low'] = low
        new_row['Volume'] = volume
        new_row['Close'] = last_row['Close']
        rolling_window = pd.concat([rolling_window, new_row.to_frame().T], ignore_index=True)
        rolling_window = recalc_indicators_fn(rolling_window)
        latest_row = rolling_window.iloc[-1]
        X_input = pd.DataFrame([latest_row[feature_columns].values], columns=feature_columns)
        predicted_close = model.predict(X_input)[0]
        rolling_window.at[rolling_window.index[-1], 'Close'] = predicted_close
        predictions.append(rolling_window.iloc[-1].copy())

    output_df = pd.DataFrame(predictions)
    os.makedirs('predictions', exist_ok=True)
    output_csv = f'predictions/{ticker}_iterative_predictions.csv'
    output_df.to_csv(output_csv, index=False)
    print(f"[{ticker}] Predictions saved to {output_csv}")

def process_ticker(ticker, train_ranges, num_days, recalc_indicators):
    if ticker not in train_ranges:
        return f"[{ticker}] Skipped: No training range."

    train_start_date, train_end_date = train_ranges[ticker]
    if train_start_date is None or train_end_date is None:
        return f"[{ticker}] Invalid training range."

    trading_days = get_trading_days(train_start_date, '2025-12-31')
    future_trading_days = trading_days[trading_days > train_end_date][:num_days]

    if future_trading_days.empty:
        return f"[{ticker}] No future trading days."

    try:
        model, df, mse, rmse, mae, r2, feature_columns = train_model_for_ticker(
            ticker, train_start_date, train_end_date
        )
        print(f"[{ticker}] MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")
        predict_iteratively(ticker, model, df, future_trading_days, recalc_indicators, feature_columns)
        return f"[{ticker}] Done."
    except Exception as e:
        return f"[{ticker}] Failed with error: {e}"

def main():
    from features import recalc_indicators
    num_days = int(input("How many future trading days would you like to predict? ").strip())

    if os.path.exists('predictions'):
        for file in os.listdir('predictions'):
            file_path = os.path.join('predictions', file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    train_ranges = {}
    with open('ticker_date_ranges.txt') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                ticker, start, end = parts
                train_ranges[ticker] = (adjust_date(start), adjust_date(end))

    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_ticker, ticker, train_ranges, num_days, recalc_indicators): ticker for ticker in tickers}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    for r in results:
        print(r)

if __name__ == '__main__':
    main()