import pandas as pd

def run_strategy(data, strategy_func, backtest_func):
    
    results = []

    for stock in data["Stock"].unique():
        df = data[data["Stock"] == stock].copy()
        
        df = strategy_func(df)
        df = backtest_func(df)
        
        results.append(df)

    return pd.concat(results, ignore_index=True)