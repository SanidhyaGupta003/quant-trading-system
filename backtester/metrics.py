import numpy as np

def calculate_metrics(portfolio_df):

    portfolio_df = portfolio_df.copy()

    # Daily returns
    portfolio_df["returns"] = portfolio_df["portfolio_value"].pct_change()

    returns = portfolio_df["returns"].dropna()

    # Total Return
    total_return = (portfolio_df["portfolio_value"].iloc[-1] /
                    portfolio_df["portfolio_value"].iloc[0]) - 1

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)

    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    # Max Drawdown
    cumulative_max = portfolio_df["portfolio_value"].cummax()
    drawdown = (portfolio_df["portfolio_value"] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }