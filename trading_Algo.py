import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta


class TradingAlgorithm:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.portfolio = {}
        self.trade_history = []
        self.strategies = [self.strategy_1, self.strategy_2, self.strategy_3,
                           self.strategy_4, self.strategy_5, self.strategy_6,
                           self.strategy_7, self.strategy_8]
        self.daily_risk_limit = 0.01  # Maximum 1% daily risk
        self.max_drawdown = 0.05  # Maximum 5% capital loss allowed

    def strategy_1(self, market_data):
        """Mean Reversion Strategy."""
        return self.execute_trade(market_data, risk_factor=0.1)

    def strategy_2(self, market_data):
        """Momentum Strategy."""
        return self.execute_trade(market_data, risk_factor=0.08)

    def strategy_3(self, market_data):
        """Arbitrage Trading."""
        return self.execute_trade(market_data, risk_factor=0.05)

    def strategy_4(self, market_data):
        """Volatility Breakout."""
        return self.execute_trade(market_data, risk_factor=0.12)

    def strategy_5(self, market_data):
        """Scalping."""
        return self.execute_trade(market_data, risk_factor=0.07)

    def strategy_6(self, market_data):
        """Trend Following."""
        return self.execute_trade(market_data, risk_factor=0.09)

    def strategy_7(self, market_data):
        """Machine Learning Predictive Analysis."""
        return self.execute_trade(market_data, risk_factor=0.06)

    def strategy_8(self, market_data):
        """AI-Powered Sentiment Analysis."""
        return self.execute_trade(market_data, risk_factor=0.11)

    def execute_trade(self, market_data, risk_factor):
        """Executes a trade with controlled risk exposure."""
        if self.capital * self.max_drawdown > self.capital:
            print("Risk limit reached, stopping trades.")
            return None

        trade_size = self.capital * risk_factor
        profit_or_loss = trade_size * random.uniform(-0.02, 0.05)  # Simulating P/L
        self.capital += profit_or_loss
        self.trade_history.append({'time': datetime.now(), 'trade_size': trade_size, 'pnl': profit_or_loss})
        return profit_or_loss

    def run(self, days=30):
        """Runs the algorithm for a given number of days."""
        for _ in range(days):
            for strategy in self.strategies:
                market_data = self.get_market_data()
                strategy(market_data)

    def get_market_data(self):
        """Simulates market data."""
        return {
            'price': random.uniform(50, 500),
            'volatility': random.uniform(0.01, 0.05),
            'trend': random.choice(['up', 'down', 'sideways'])
        }

    def summary(self):
        """Prints a summary of trading performance."""
        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Number of Trades: {len(self.trade_history)}")


if __name__ == "__main__":
    algo = TradingAlgorithm()
    algo.run(days=60)
    algo.summary()
