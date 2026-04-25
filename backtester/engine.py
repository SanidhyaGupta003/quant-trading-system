import pandas as pd

def backtest_stock(df, initial_cash=10000, transaction_cost=0.001):

    cash = initial_cash
    position = 0
    portfolio_values = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        signal = df["position"].iloc[i]

        # BUY
        if signal == 1 and cash > 0:
            cost = cash * transaction_cost
            position = (cash - cost) / price
            cash = 0

        # SELL
        elif signal == -1 and position > 0:
            proceeds = position * price
            cost = proceeds * transaction_cost
            cash = proceeds - cost
            position = 0

        total_value = cash + (position * price)
        portfolio_values.append(total_value)

    df["portfolio_value"] = portfolio_values

    return df