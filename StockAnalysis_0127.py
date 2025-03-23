import asyncio
import aiohttp
from functools import lru_cache
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import spacy
from scipy.stats import norm, zscore
import quantstats as qs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")


@dataclass
class Config:
    alpha_vantage_api_key: str = 'TX0HJLIXT7XMCQ9B'
    news_api_key: str = '7665dc8b93f943ea88ad3ad53381aaec'
    risk_free_rate: float = 0.02
    max_position_size: float = 0.1  # Max 10% of portfolio per position
    backtest_window: int = 252  # 1 year of trading days


@dataclass
class MarketData:
    prices: pd.DataFrame
    indicators: Dict[str, pd.Series]
    volatility: float
    fundamental_metrics: Dict[str, float]


@dataclass
class RiskAnalysis:
    var_95: float
    sharpe_ratio: float
    max_drawdown: float
    position_size: float


@dataclass
class ForecastResults:
    prophet: List[float]
    monte_carlo: List[float]
    consensus: List[float]


class AdvancedStockAnalyzer:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.session = aiohttp.ClientSession()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.news_sources = {  # Source credibility weights
            'reuters.com': 0.9, 'bloomberg.com': 0.85,
            'wsj.com': 0.88, 'cnbc.com': 0.82
        }

    async def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Async fetch stock data with LRU caching"""
        try:
            data = await asyncio.to_thread(
                yf.download, ticker, period="5y", interval="1d"
            )
            return data.dropna()
        except Exception as e:
            logger.error(f"Data fetch failed: {str(e)}")
            return None

    async def fetch_market_trends(self) -> Dict[str, float]:
        """Async fetch multiple market indices"""
        indices = ['^GSPC', '^IXIC', '^DJI', '^N225']
        async with self.session as session:
            tasks = [self._fetch_index(session, idx) for idx in indices]
            results = await asyncio.gather(*tasks)
        return {idx: trend for idx, trend in results}

    async def _fetch_index(self, session: aiohttp.ClientSession, symbol: str) -> Tuple[str, float]:
        """Helper for index data fetching"""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        async with session.get(url) as response:
            data = await response.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            return symbol, price

    @lru_cache(maxsize=128)
    async def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Cached company info with fundamental metrics"""
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('trailingPE'),
            'market_cap': info.get('marketCap'),
            'beta': info.get('beta'),
            'sector': info.get('sector')
        }

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Enhanced technical analysis with multiple indicators"""
        closes = data['Close']

        # Bollinger Bands
        sma = closes.rolling(20).mean()
        std = closes.rolling(20).std()
        data['BB_upper'] = sma + 2 * std
        data['BB_lower'] = sma - 2 * std

        # MACD
        ema12 = closes.ewm(span=12).mean()
        ema26 = closes.ewm(span=26).mean()
        data['MACD'] = ema12 - ema26
        data['Signal'] = data['MACD'].ewm(span=9).mean()

        # ATR
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        data['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        # Z-Score
        data['ZScore'] = zscore(closes)

        return data

    async def analyze_news(self, ticker: str) -> Dict[str, Any]:
        """Enhanced news analysis with entity recognition and source credibility"""
        url = (f"https://newsapi.org/v2/everything?q={ticker}"
               f"&apiKey={self.config.news_api_key}&pageSize=50")

        async with self.session.get(url) as response:
            articles = (await response.json()).get('articles', [])

        analyzed = []
        for article in articles:
            source = article['source']['id']
            weight = self.news_sources.get(source, 0.5)

            # Entity recognition
            doc = nlp(article['title'] + " " + article['description'])
            entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON']]

            sentiment = self.sentiment_analyzer.polarity_scores(article['title'])
            analyzed.append({
                'sentiment': sentiment['compound'] * weight,
                'entities': entities,
                'source_weight': weight,
                'content': article['title']
            })

        return {
            'average_sentiment': np.mean([a['sentiment'] for a in analyzed]),
            'entity_frequency': self._count_entities(analyzed),
            'top_articles': analyzed[:5]
        }

    def _count_entities(self, articles: List[Dict]) -> Dict[str, int]:
        """Count entity frequency across articles"""
        counts = {}
        for article in articles:
            for entity in article['entities']:
                counts[entity] = counts.get(entity, 0) + 1
        return counts

    async def forecast_prices(self, data: pd.DataFrame) -> ForecastResults:
        """Hybrid forecasting with Prophet and Monte Carlo"""
        # Prophet forecast
        prophet_data = data.reset_index()[['Date', 'Close']].rename(
            columns={'Date': 'ds', 'Close': 'y'}
        )
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=7)
        prophet_forecast = model.predict(future)['yhat'][-7:].tolist()

        # Monte Carlo simulation
        log_returns = np.log(1 + data['Close'].pct_change().dropna())
        volatility = log_returns.std()
        simulations = np.random.normal(
            loc=log_returns.mean(),
            scale=volatility,
            size=(100, 7)
        )
        price_paths = data['Close'].iloc[-1] * np.exp(np.cumsum(simulations, axis=1))
        monte_carlo = price_paths.mean(axis=0).tolist()

        return ForecastResults(
            prophet=prophet_forecast,
            monte_carlo=monte_carlo,
            consensus=[(p + m) / 2 for p, m in zip(prophet_forecast, monte_carlo)]
        )

    def calculate_risk_metrics(self, data: pd.DataFrame) -> RiskAnalysis:
        """Calculate advanced risk metrics"""
        returns = data['Close'].pct_change().dropna()

        # Value at Risk (95% confidence)
        var_95 = norm.ppf(0.05, returns.mean(), returns.std())

        # Sharpe Ratio
        sharpe = (returns.mean() - self.config.risk_free_rate / 252) / returns.std()

        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak - 1).min()

        # Position sizing based on ATR
        atr = data['ATR'].iloc[-1]
        position_size = min(self.config.max_position_size, 0.01 * (data['Close'].iloc[-1] / atr))

        return RiskAnalysis(
            var_95=var_95,
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            position_size=position_size
        )

    async def backtest_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Backtesting framework with performance metrics"""
        signals = self._generate_signals(data)
        returns = data['Close'].pct_change().shift(-1)
        strategy_returns = signals.shift(1) * returns
        return qs.reports.metrics(strategy_returns, benchmark=returns)

    def _generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on technical indicators"""
        signals = pd.Series(0, index=data.index)
        # Example strategy: Buy when MACD crosses above Signal line
        signals[data['MACD'] > data['Signal']] = 1
        signals[data['MACD'] < data['Signal']] = -1
        return signals

    async def full_analysis(self, ticker: str) -> Dict[str, Any]:
        """Complete analysis pipeline"""
        data = await self.fetch_stock_data(ticker)
        if data is None:
            return {}

        # Parallel processing
        market_trends, company_info, news = await asyncio.gather(
            self.fetch_market_trends(),
            self.get_company_info(ticker),
            self.analyze_news(ticker)
        )

        data = self.calculate_technical_indicators(data)
        forecast = await self.forecast_prices(data)
        risk = self.calculate_risk_metrics(data)
        backtest = await self.backtest_strategy(data)

        return {
            'ticker': ticker,
            'market_data': market_trends,
            'company_info': company_info,
            'news_analysis': news,
            'forecast': forecast,
            'risk_metrics': risk,
            'backtest': backtest,
            'latest_price': data['Close'].iloc[-1]
        }


async def main():
    analyzer = AdvancedStockAnalyzer()
    while True:
        ticker = input("Enter ticker (or 'exit'): ").strip().upper()
        if ticker.lower() == 'exit':
            break

        results = await analyzer.full_analysis(ticker)
        print(f"\nAnalysis for {ticker}:")
        print(f"Latest Price: {results.get('latest_price', 'N/A')}")
        print(f"PE Ratio: {results['company_info'].get('pe_ratio', 'N/A')}")
        print(f"Risk Metrics (VaR 95%: {results['risk_metrics'].var_95:.2%})")
        print(f"7-Day Forecast: {results['forecast'].consensus}")


if __name__ == "__main__":
    asyncio.run(main())