import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

tick_filename = "tickers.txt" # list of tickers to train on 
output_dir = "stock_data" # folder to save ticker data to

def get_tickers(filename): # Returns a list of tickers from a file
  try:
    with open(filename, "r") as file:
      output = [stripped for line in file.readlines() if (stripped := line.strip()) and not stripped.startswith("#")]
      return output
  except FileNotFoundError:
    print(f"Error reading tickers from {filename}")
    return []
def get_polygon_data(ticker, output_folder): # single ticker info to new file in output folder
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

        #print(df)
        #print(f"Data gathered for {ticker}")

        output_file = os.path.join(output_folder, f"{ticker}.csv")
        df.to_csv(output_file)
        print(f"Data saved to {output_file}")
        time.sleep(12.5) # pause for 12.5 seconds to keep under 5 req/min

    except Exception as e:
        print(f"Error getting data for {ticker}: {e}")
        return
def load_stock_data(ticker, output_folder):
  #create a output folder thats clear
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")
  else:
    for file in os.listdir(output_folder):
      file_path = os.path.join(output_folder, file)
      if os.path.isfile(file_path):
        os.remove(file_path)
    print(f"Cleared output folder: {output_folder}")

  #add cleaned data to the output folder
  tickers = get_tickers(tick_filename)
  if tickers:
    for ticker in tickers:
      get_polygon_data(ticker, output_folder)
  else:
    print(f"No tickers found in {tick_filename}")
    return

def process_data(file_path):
  filename = os.path.basename(file_path)
  try:
    df = pd.read_csv(file_path)
    data_col = next((col for col in df.columns if col.strip().lower() == "date"), None)
    
    if data_col:
      #convert to datetime with strict parsing from ambiguous formats
      df[data_col] = pd.to_datetime(df[data_col], errors="coerce", dayfirst=False)
      df[data_col] = df[data_col].dt.strftime("%Y-%m-%d")

      df.to_csv(file_path, index=False)
      return f"Cleaned {filename}"
    else:
      print(f"No date column found in {filename}, skipping...")

  except Exception as e:
    return f"Error cleaning {filename}: {e}"
def format_all_data(data_folder):
  csv_files = [
    os.path.join(data_folder, file)
    for file in os.listdir(data_folder) if file.endswith(".csv")
  ]

  output = []
  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_data, file): file for file in csv_files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
      output.append(future.result())

    
  for result in output:
    print(result)
    

def main():
  load_stock_data(tick_filename, output_dir)
  print("Stock data loaded successfully")
  format_all_data(output_dir)
  print("Data formatted successfully")



if __name__ == "__main__":
  main()