import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_data
from strategies.moving_average import moving_average_strategy
from strategies.momentum import momentum_strategy
from strategies.mean_reversion import mean_reversion_strategy
from backtester.engine import backtest_stock
from backtester.metrics import calculate_metrics


# =========================================================
# CACHE DATA
# =========================================================
@st.cache_data
def get_data():
    return load_data()


data = get_data()

# =========================================================
# TITLE
# =========================================================
st.title("📊 Quant Trading Dashboard")

st.caption("Interactive system for strategy backtesting, portfolio optimization, and risk analysis.")

st.markdown("### Dataset Overview")
st.write(data.head())
st.write("Total rows:", len(data))
st.write("Unique stocks:", data["Stock"].nunique())

st.markdown("---")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    ["Moving Average", "Momentum", "Mean Reversion"]
)

short_window = 10
long_window = 30

if strategy_name == "Moving Average":
    short_window = st.sidebar.slider("Short Window", 5, 30, 10)
    long_window = st.sidebar.slider("Long Window", 20, 100, 30)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Strategy Dashboard",
    "📈 Efficient Frontier",
    "🚀 Strategy Optimization"
])


# =========================================================
# TAB 1 — STRATEGY DASHBOARD
# =========================================================
with tab1:

    st.markdown("## 📊 Strategy Performance")

    with st.spinner("Running strategy backtest..."):
        results = []

        for stock in data["Stock"].unique():
            df = data[data["Stock"] == stock].copy()

            if strategy_name == "Moving Average":
                df = moving_average_strategy(df, short_window, long_window)
            elif strategy_name == "Momentum":
                df = momentum_strategy(df)
            else:
                df = mean_reversion_strategy(df)

            df = backtest_stock(df)
            results.append(df)

    final_df = pd.concat(results, ignore_index=True)

    portfolio = (
        final_df.groupby("Date")["portfolio_value"]
        .sum()
        .reset_index()
    )

    # Plot
    fig, ax = plt.subplots()
    ax.plot(portfolio["Date"], portfolio["portfolio_value"])
    ax.set_title("Cumulative Portfolio Value")
    st.pyplot(fig)

    st.markdown("---")

    # Download
    st.download_button(
        "📥 Download Portfolio Data",
        portfolio.to_csv(index=False).encode("utf-8"),
        "portfolio.csv"
    )

    # Metrics
    metrics = calculate_metrics(portfolio)

    st.markdown("## 📊 Performance Metrics")

    col1, col2 = st.columns(2)

    for i, (k, v) in enumerate(metrics.items()):
        if i % 2 == 0:
            col1.metric(k, f"{float(v):.4f}")
        else:
            col2.metric(k, f"{float(v):.4f}")

    st.markdown("---")

    # Comparison
    compare = st.checkbox("Compare All Strategies")

    if compare:

        def run_strategy(strategy_func):
            results = []
            for stock in data["Stock"].unique():
                df = data[data["Stock"] == stock].copy()
                df = strategy_func(df)
                df = backtest_stock(df)
                results.append(df)

            final = pd.concat(results)
            return final.groupby("Date")["portfolio_value"].sum().reset_index()

        ma_port = run_strategy(lambda df: moving_average_strategy(df, 10, 30))
        mom_port = run_strategy(momentum_strategy)
        mr_port = run_strategy(mean_reversion_strategy)

        fig2, ax2 = plt.subplots()
        ax2.plot(ma_port["Date"], ma_port["portfolio_value"], label="MA")
        ax2.plot(mom_port["Date"], mom_port["portfolio_value"], label="Momentum")
        ax2.plot(mr_port["Date"], mr_port["portfolio_value"], label="Mean Reversion")
        ax2.legend()
        ax2.set_title("Strategy Comparison")

        st.pyplot(fig2)


# =========================================================
# TAB 2 — EFFICIENT FRONTIER
# =========================================================
with tab2:

    st.markdown("## 📈 Efficient Frontier")

    returns_df = (
        data.pivot(index="Date", columns="Stock", values="Close")
        .pct_change()
        .dropna()
    )

    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    num_portfolios = st.slider("Number of Portfolios", 1000, 10000, 3000)

    with st.spinner("Simulating portfolios..."):
        results = []

        for _ in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)

            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = port_return / port_vol

            results.append((port_return, port_vol, sharpe))

    returns = [r[0] for r in results]
    vols = [r[1] for r in results]
    sharpes = [r[2] for r in results]

    best_idx = sharpes.index(max(sharpes))
    best_return = returns[best_idx]
    best_vol = vols[best_idx]
    best_sharpe = sharpes[best_idx]

    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(vols, returns, c=sharpes)
    plt.colorbar(scatter, ax=ax3)
    ax3.scatter(best_vol, best_return, color="red", s=100)
    ax3.set_xlabel("Volatility")
    ax3.set_ylabel("Return")

    st.pyplot(fig3)

    st.markdown("### 🏆 Best Portfolio")

    col1, col2, col3 = st.columns(3)
    col1.metric("Return", f"{best_return:.4f}")
    col2.metric("Volatility", f"{best_vol:.4f}")
    col3.metric("Sharpe", f"{best_sharpe:.4f}")

    st.download_button(
        "📥 Download Frontier Data",
        pd.DataFrame({
            "Return": returns,
            "Volatility": vols,
            "Sharpe": sharpes
        }).to_csv(index=False).encode("utf-8"),
        "efficient_frontier.csv"
    )


# =========================================================
# TAB 3 — STRATEGY OPTIMIZATION
# =========================================================
with tab3:

    st.markdown("## 🚀 Strategy-Level Optimization")

    strategy_results = []

    for stock in data["Stock"].unique():
        df = data[data["Stock"] == stock].copy()
        df = moving_average_strategy(df, 10, 30)
        df = backtest_stock(df)
        strategy_results.append(df)

    strategy_data = pd.concat(strategy_results, ignore_index=True)

    strategy_returns = []

    for stock in strategy_data["Stock"].unique():
        df = strategy_data[strategy_data["Stock"] == stock].copy()
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

    num_portfolios = st.slider("Strategy Portfolios", 1000, 10000, 3000)

    with st.spinner("Optimizing strategy portfolio..."):
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

    # Portfolio
    strategy_portfolio_returns = returns_matrix.dot(best_weights)

    strategy_portfolio = pd.DataFrame({
        "Date": returns_matrix.index,
        "portfolio_return": strategy_portfolio_returns.values
    })

    strategy_portfolio["portfolio_value"] = (
        100000 * (1 + strategy_portfolio["portfolio_return"]).cumprod()
    )

    fig4, ax4 = plt.subplots()
    ax4.plot(strategy_portfolio["Date"], strategy_portfolio["portfolio_value"])
    ax4.set_title("Optimized Portfolio Value")
    st.pyplot(fig4)

    opt_metrics = calculate_metrics(strategy_portfolio)

    st.markdown("### 📊 Optimized Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Return", f"{float(opt_metrics['Total Return']):.4f}")
    col2.metric("Volatility", f"{float(opt_metrics['Volatility']):.4f}")
    col3.metric("Sharpe", f"{float(opt_metrics['Sharpe Ratio']):.4f}")
    col4.metric("Max Drawdown", f"{float(opt_metrics['Max Drawdown']):.4f}")

    st.markdown("---")

    # Weights
    weights_df = pd.DataFrame({
        "Stock": returns_matrix.columns,
        "Weight": best_weights
    }).sort_values(by="Weight", ascending=False)

    st.markdown("### 🧾 Portfolio Weights")
    st.dataframe(weights_df)

    st.download_button(
        "📥 Download Portfolio Weights",
        weights_df.to_csv(index=False).encode("utf-8"),
        "weights.csv"
    )