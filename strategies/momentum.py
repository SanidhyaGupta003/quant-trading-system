import pandas as pd

def momentum_strategy(df, window=20):

    df = df.copy()

    # Calculate momentum (percentage change over window)
    df["momentum"] = df["Close"].pct_change(periods=window)

    # Signal: 1 = buy, 0 = no position
    df["signal"] = 0
    df.loc[df["momentum"] > 0, "signal"] = 1

    # Position (entry/exit)
    df["position"] = df["signal"].diff().fillna(0)

    return df