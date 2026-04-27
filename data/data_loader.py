# import yfinance as yf
# import pandas as pd

# def fetch_and_save_data():

#     stocks = [
#         "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TCS.NS", "INFY.NS",
#         "RELIANCE.NS", "ONGC.NS", "M&M.NS", "MARUTI.NS", "HINDUNILVR.NS",
#         "ITC.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "LT.NS", "GRASIM.NS"
#     ]

#     all_data = []

#     for stock in stocks:
#         print(f"Downloading {stock}...")
#         try:
#             df = yf.Ticker(stock).history(start="2016-01-01", end="2025-12-31")

#             if df.empty:
#                 continue

#             df.reset_index(inplace=True)
#             df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
#             df["Stock"] = stock

#             all_data.append(df)

#         except Exception as e:
#             print(f"Error with {stock}: {e}")

#     final_df = pd.concat(all_data, ignore_index=True)

#     final_df = final_df[["Date", "Open", "High", "Low", "Close", "Volume", "Stock"]]
#     final_df.dropna(inplace=True)
#     final_df.sort_values(by=["Stock", "Date"], inplace=True)

#     final_df["Daily Return"] = final_df.groupby("Stock")["Close"].pct_change()

#     final_df.to_csv("data_files/new_stock_data.csv", index=False)

#     print("✅ Data saved successfully")

# def load_data():
#     df = pd.read_csv("data_files/new_stock_data.csv")
#     df["Date"] = pd.to_datetime(df["Date"])
#     df = df.sort_values(by=["Stock", "Date"])
#     print("data_loader loaded")
#     return df

import yfinance as yf
import pandas as pd

def load_data():

    stocks = [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TCS.NS", "INFY.NS",
        "RELIANCE.NS", "ONGC.NS", "M&M.NS", "MARUTI.NS", "HINDUNILVR.NS",
        "ITC.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "LT.NS", "GRASIM.NS"
    ]

    all_data = []

    for stock in stocks:
        df = yf.Ticker(stock).history(start="2016-01-01", end="2025-12-31")

        if df.empty:
            continue

        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Stock"] = stock

        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    final_df = final_df[["Date", "Open", "High", "Low", "Close", "Volume", "Stock"]]
    final_df.dropna(inplace=True)
    final_df.sort_values(by=["Stock", "Date"], inplace=True)

    final_df["Daily Return"] = final_df.groupby("Stock")["Close"].pct_change()

    return final_df