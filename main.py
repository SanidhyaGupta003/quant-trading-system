from data.data_loader import load_data
from strategies.moving_average import moving_average_strategy
from backtester.engine import backtest_stock
import pandas as pd
import matplotlib.pyplot as plt
from backtester.metrics import calculate_metrics
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from backtester.optimizer import optimize_ma
import numpy as np



# ----------------------------------------------------------------
# run these two lines only for the first time
# after first time run line 1 instead of these
# from data.data_loader import fetch_and_save_data
# fetch_and_save_data()
# ----------------------------------------------------------------


def main():
    data = load_data()
    print(data.head())
    print("Total rows:", len(data))
    print("Unique stocks:", data["Stock"].nunique())

    # Apply strategy per stock

    # moving average    
    strategy_results = []
    for stock in data["Stock"].unique():        
        stock_df = data[data["Stock"] == stock].copy()        
        # Apply strategy
        stock_df = moving_average_strategy(stock_df)        
        # Append back
        strategy_results.append(stock_df)

    # Combine all stocks
    strategy_data = pd.concat(strategy_results, ignore_index=True)
    print(strategy_data .head())
    print(strategy_data .tail())
    print(strategy_data ["position"].value_counts())
    print(strategy_data.columns)

    # Apply backtest per stock
    backtest_results = []
    for stock in strategy_data["Stock"].unique():        
        stock_df = strategy_data[strategy_data["Stock"] == stock].copy()        
        stock_df = backtest_stock(stock_df)        
        backtest_results.append(stock_df)

    backtest_data = pd.concat(backtest_results, ignore_index=True)
    print(backtest_data.head())
    print(backtest_data.tail())
    print(backtest_data[backtest_data["Stock"] == "RELIANCE.NS"][["Date","Close","position","portfolio_value"]].tail(20))


    # ---------------------------------------
    # Strategy-Level Portfolio Optimization
    # ---------------------------------------

    print("---------------Strategy-Level Portfolio Optimization-------------------")

    # Compute Strategy Returns per Stock
    strategy_returns = []
    for stock in backtest_data["Stock"].unique():        
        df = backtest_data[backtest_data["Stock"] == stock].copy()
        # Compute returns from portfolio value
        df["strategy_return"] = df["portfolio_value"].pct_change()
        strategy_returns.append(df[["Date", "Stock", "strategy_return"]])

    strategy_returns_df = pd.concat(strategy_returns)

    returns_matrix = (
        strategy_returns_df
        .pivot(index="Date", columns="Stock", values="strategy_return")
        .dropna()
    )

    mean_returns = returns_matrix.mean() * 252
    cov_matrix = returns_matrix.cov() * 252

    num_portfolios = 5000
    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol
        results.append((port_return, port_vol, sharpe, weights))
    
    best_strategy = max(results, key=lambda x: x[2])
    best_return_strategy, best_vol_strategy, best_sharpe_strategy, best_weights_strategy = best_strategy
    # -----------------------------------
    # STRATEGY PORTFOLIO TIME SERIES
    # -----------------------------------
    strategy_portfolio_returns = returns_matrix.dot(best_weights_strategy)
    strategy_portfolio = pd.DataFrame({
        "Date": returns_matrix.index,
        "portfolio_return": strategy_portfolio_returns.values
    })
    initial_value = 100000
    strategy_portfolio["portfolio_value"] = initial_value * (1 + strategy_portfolio["portfolio_return"]).cumprod()
    strategy_opt_metrics = {
        "Total Return": best_return_strategy,
        "Volatility": best_vol_strategy,
        "Sharpe Ratio": best_sharpe_strategy
    }
    strategy_opt_metrics = calculate_metrics(strategy_portfolio)
    # best = max(results, key=lambda x: x[2])
    # best_return, best_vol, best_sharpe, best_weights = best
    # strategy_opt_metrics = {
    #     "Total Return": best_return_strategy,
    #     "Volatility": best_vol_strategy,
    #     "Sharpe Ratio": best_sharpe_strategy
    # }
    print("Strategy Portfolio Sharpe:", best_sharpe_strategy)
    print("Return:", best_return_strategy)
    print("Volatility:", best_vol_strategy)

    print("-----------------------------------------")


    # -----------------------------------
    # PORTFOLIO LEVEL AGGREGATION
    # -----------------------------------

    # Sum portfolio across stocks per date
    portfolio = (
        backtest_data.groupby("Date")["portfolio_value"]
        .sum()
        .reset_index()
    )

    print(portfolio.head())

    # plot portfolio
    plt.figure(figsize=(10,5))
    plt.plot(portfolio["Date"], portfolio["portfolio_value"])
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()

    # total return
    initial = portfolio["portfolio_value"].iloc[0]
    final = portfolio["portfolio_value"].iloc[-1]

    total_return = (final / initial) - 1
    print("Total Portfolio Return:", total_return)

    # ----------------------------------------------
    
    metrics = calculate_metrics(portfolio)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")



    # ---------------------------------------
    # MOMENTUM STRATEGY
    # ---------------------------------------

    momentum_results = []
    for stock in data["Stock"].unique():        
        stock_df = data[data["Stock"] == stock].copy()        
        stock_df = momentum_strategy(stock_df)        
        stock_df = backtest_stock(stock_df)        
        momentum_results.append(stock_df)

    momentum_data = pd.concat(momentum_results, ignore_index=True)

    momentum_portfolio = (
        momentum_data.groupby("Date")["portfolio_value"]
        .sum()
        .reset_index()
    )

    # Compare with MA Strategy

    plt.figure(figsize=(10,5))
    plt.plot(portfolio["Date"], portfolio["portfolio_value"], label="MA Strategy")
    plt.plot(momentum_portfolio["Date"], momentum_portfolio["portfolio_value"], label="Momentum Strategy")
    plt.legend()
    plt.title("Strategy Comparison")
    plt.show()

    momentum_metrics = calculate_metrics(momentum_portfolio)

    print("\nMomentum Strategy Metrics:")
    for k, v in momentum_metrics.items():
        print(f"{k}: {v:.4f}")

    # ---------------------------------------
    # MEAN REVERSION STRATEGY
    # ---------------------------------------

    mr_results = []

    for stock in data["Stock"].unique():        
        stock_df = data[data["Stock"] == stock].copy()        
        stock_df = mean_reversion_strategy(stock_df)        
        stock_df = backtest_stock(stock_df)        
        mr_results.append(stock_df)

    mr_data = pd.concat(mr_results, ignore_index=True)

    mr_portfolio = (
        mr_data.groupby("Date")["portfolio_value"]
        .sum()
        .reset_index()
    )

    mr_metrics = calculate_metrics(mr_portfolio)

    print("\nMean Reversion Metrics:")
    for k, v in mr_metrics.items():
        print(f"{k}: {v:.4f}")

    # ---------------------------------------
    # FINAL STRATEGY COMPARISON
    # ---------------------------------------

    print("\n=== FINAL STRATEGY COMPARISON ===")

    print("\nMoving Average:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nMomentum:")
    for k, v in momentum_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nMean Reversion:")
    for k, v in mr_metrics.items():
        print(f"{k}: {v:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(portfolio["Date"], portfolio["portfolio_value"], label="MA")
    plt.plot(momentum_portfolio["Date"], momentum_portfolio["portfolio_value"], label="Momentum")
    plt.plot(mr_portfolio["Date"], mr_portfolio["portfolio_value"], label="Mean Reversion")
    plt.legend()
    plt.title("Strategy Comparison")
    plt.show()

    # ------------------------------------------------------------------------

    opt_results = optimize_ma(
        data,
        short_range=range(5, 30, 5),
        long_range=range(30, 100, 10),
        backtest_func=backtest_stock,
        metric_func=calculate_metrics
    )

    print(opt_results.sort_values(by="sharpe", ascending=False).head())

    # --------------------------------------------
    # Efficient Frontier (Portfolio Optimization)
    # --------------------------------------------

    returns_df = (
        data.pivot(index="Date", columns="Stock", values="Close")
        .pct_change()
        .dropna()
    )

    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    num_portfolios = 5000

    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol
        results.append((port_return, port_vol, sharpe, weights))

    best = max(results, key=lambda x: x[2])
    best_return, best_vol, best_sharpe, best_weights = best
    # -----------------------------------
    # RAW ASSET PORTFOLIO TIME SERIES
    # -----------------------------------
    # Daily portfolio returns
    raw_portfolio_returns = returns_df.dot(best_weights)
    # Convert to DataFrame
    raw_portfolio = pd.DataFrame({
        "Date": raw_portfolio_returns.index,
        "portfolio_return": raw_portfolio_returns.values
    })
    # Create portfolio value
    initial_value = 100000
    raw_portfolio["portfolio_value"] = initial_value * (1 + raw_portfolio["portfolio_return"]).cumprod()
    raw_metrics = {
        "Total Return": best_return,
        "Volatility": best_vol,
        "Sharpe Ratio": best_sharpe
    }
    raw_metrics = calculate_metrics(raw_portfolio)
    print("Best Sharpe:", best_sharpe)
    print("Return:", best_return)
    print("Volatility:", best_vol)

    # Efficient Frontier Plot

    returns = [r[0] for r in results]
    vols = [r[1] for r in results]
    sharpes = [r[2] for r in results]

    plt.figure(figsize=(10,6))
    scatter = plt.scatter(vols, returns, c=sharpes, cmap="viridis")
    plt.colorbar(scatter, label="Sharpe Ratio")
    # Highlight best portfolio
    plt.scatter(best_vol, best_return, color="red", s=100, label="Max Sharpe")
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.show()
    
   
    # ------------------------------------------------
    # Final Comparison Table
    # ------------------------------------------------

    print("\n=== FINAL COMPARISON ===")

    print("\nEqual Weight Strategy:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nRaw Asset Optimized:")
    print(f"Sharpe: {best_sharpe:.4f}")

    print("\nStrategy-Level Optimized:")
    print(f"Sharpe: {best_sharpe_strategy:.4f}")


    print("\n========== FINAL PORTFOLIO COMPARISON ==========\n")

    print("1️⃣ Equal Weight Strategy Portfolio:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n2️⃣ Raw Asset Optimized Portfolio:")
    for k, v in raw_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n3️⃣ Strategy-Level Optimized Portfolio:")
    for k, v in strategy_opt_metrics.items():
        print(f"{k}: {v:.4f}")


    comparison_df = pd.DataFrame({
        "Equal Weight Strategy": metrics,
        "Raw Asset Optimized": raw_metrics,
        "Strategy Optimized": strategy_opt_metrics
    })
    print("\n=== FINAL COMPARISON TABLE ===\n")
    print(comparison_df)

    
    comparison_df = pd.DataFrame({
        "Equal Weight MA": metrics,
        "Momentum": momentum_metrics,
        "Mean Reversion": mr_metrics,
        "Raw Asset Optimized": raw_metrics,
        "Strategy Optimized": strategy_opt_metrics
    })
    print(comparison_df)

    returns_df = returns_df.dropna()
    returns_matrix = returns_matrix.dropna()
    plt.figure(figsize=(10,5))
    plt.plot(raw_portfolio["Date"], raw_portfolio["portfolio_value"], label="Raw Optimized")
    plt.plot(strategy_portfolio["Date"], strategy_portfolio["portfolio_value"], label="Strategy Optimized")
    plt.legend()
    plt.title("Optimized Portfolio Comparison")
    plt.show()

if __name__ == "__main__":
    main()