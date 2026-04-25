import pandas as pd

def optimize_ma(data, short_range, long_range, backtest_func, metric_func):

    results = []

    for short in short_range:
        for long in long_range:
            if short >= long:
                continue

            all_results = []

            for stock in data["Stock"].unique():
                df = data[data["Stock"] == stock].copy()
                # Apply MA with custom params
                df["short_ma"] = df["Close"].rolling(short).mean()
                df["long_ma"] = df["Close"].rolling(long).mean()
                df["signal"] = 0
                df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1
                df["position"] = df["signal"].diff().fillna(0)
                df = backtest_func(df)
                all_results.append(df)

            combined = pd.concat(all_results)

            portfolio = (
                combined.groupby("Date")["portfolio_value"]
                .sum()
                .reset_index()
            )

            metrics = metric_func(portfolio)

            results.append({
                "short": short,
                "long": long,
                "sharpe": metrics["Sharpe Ratio"],
                "return": metrics["Total Return"]
            })

    return pd.DataFrame(results)