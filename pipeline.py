import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

def get_tickers(filename): # Returns a list of tickers from a file
  try:
    with open(filename, "r") as file:
      output = [stripped for line in file.readlines() if (stripped := line.strip()) and not stripped.startswith("#")]
      return output
  except FileNotFoundError:
    print(f"Error reading tickers from {filename}")
    return []

def get_polygon_data(ticker): # single ticker info
    url = (
      f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
      f"2000-01-01/{pd.Timestamp.now().strftime('%Y-%m-%d')}?adjusted=true&sort=asc"
      f"&limit=50000&apiKey={API_KEY}"
    )
    try:
        response = requests.get(url)
        if response.status_code == 429: # Rate limit exceeded
          print(f"Rate limit exceeded for {ticker}, pausing for 60 seconds")
          time.sleep(60)
          get_polygon_data(ticker); # Pause for a min and then add this ticker
          return

        elif response.status_code != 200: # other error
          print(f"Error getting data for {ticker}: {response.status_code}")
          return
        
        #else:
        data = response.json()
        #print("Data:", data.get("results", [])[:1] if isinstance(data, dict) else data[:1])
        if "results" not in data:
          print(f"No results for {ticker}")
          return
        
        df = pd.DataFrame(data["results"])
        df.rename(columns={
          "t": "Date", 
          "o": "Open", 
          "h": "High", 
          "l": "Low", 
          "c": "Close", 
          "v": "Volume"
        }, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], unit = "ms") # convert time stamps to a date

        df.set_index("Date", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        print(df)

        print(f"Data gathered for {ticker}")


    except Exception as e:
        print(f"Error getting data for {ticker}: {e}")
        return

def main():
  tick_filename = "tickers.txt"
  tickers = get_tickers(tick_filename)
  for ticker in tickers:
    get_polygon_data(ticker)

if __name__ == "__main__":
  main()