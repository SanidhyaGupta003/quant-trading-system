import pandas as pd

def mean_reversion_strategy(df, window=20, threshold=0.02):

    df = df.copy()

    # Moving average
    df["ma"] = df["Close"].rolling(window=window).mean()

    # Deviation from mean
    df["deviation"] = (df["Close"] - df["ma"]) / df["ma"]

    # Signal
    df["signal"] = 0

    # Buy when price is below mean
    df.loc[df["deviation"] < -threshold, "signal"] = 1

    # Sell when price is above mean
    df.loc[df["deviation"] > threshold, "signal"] = 0

    # Position
    df["position"] = df["signal"].diff().fillna(0)

    return df