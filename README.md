# Quantitative Trading System

## Overview
This project implements a multi-asset quantitative trading system using Python. It includes strategy development, backtesting, parameter optimization, and portfolio construction.

## Features
- Multi-asset data pipeline (15 NSE stocks)
- Strategy implementations:
  - Moving Average (Trend Following)
  - Momentum Strategy
  - Mean Reversion Strategy
- Backtesting engine with transaction costs
- Risk metrics:
  - Sharpe Ratio
  - Volatility
  - Max Drawdown
- Parameter optimization (grid search)
- Portfolio optimization:
  - Efficient Frontier
  - Strategy-level optimization

## Key Results
- Strategy-optimized portfolio achieved highest Sharpe ratio (~1.45)
- Moving Average strategy delivered highest total return
- Mean reversion underperformed in trending markets

## Project Structure
quant_project/
│
├── data/
├── strategies/
├── backtester/
├── portfolio/
├── utils/
├── main.py

## Installation
```bash
pip install -r requirements.txt

## Usage
python main.py

## Future Improvements
- Machine learning-based signals
- Regime detection
- Real-time data integration
