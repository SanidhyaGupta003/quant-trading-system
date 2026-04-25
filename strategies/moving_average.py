import pandas as pd

def moving_average_strategy(df, short_window=10, long_window=30):

    df = df.copy()

    # Moving averages
    df["short_ma"] = df["Close"].rolling(window=short_window).mean()
    df["long_ma"] = df["Close"].rolling(window=long_window).mean()

    # Signal: 1 = buy, 0 = no position
    df["signal"] = 0
    df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1

    # Positions: when signal changes
    # df["position"] = df["signal"].diff()
    df["position"] = df["signal"].diff().fillna(0)

    return df