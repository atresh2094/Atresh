from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import ssl

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API keys from original code
ALPHA_VANTAGE_API_KEY = 'TX0HJLIXT7XMCQ9B'
NEWS_API_KEY = '7665dc8b93f943ea88ad3ad53381aaec'

@dataclass
class MarketTrend:
    sp500: str
    nasdaq: str

@dataclass
class NewsAnalysis:
    sentiment_summary: str
    avg_sentiment: float
    news_summary: str

@dataclass
class TradingRecommendation:
    recommendation: str
    suggestion: str
    trend_signal: str
    price_target: Optional[float]
    buy_date: Optional[str]
    sell_date: Optional[str]

class ConfigError(Exception):
    """Raised when configuration or API keys are invalid."""
    pass

class StockAnalyzer:
    def __init__(self):
        self.alpha_vantage_api_key = ALPHA_VANTAGE_API_KEY
        self.news_api_key = NEWS_API_KEY

        if not all([self.alpha_vantage_api_key, self.news_api_key]):
            raise ConfigError("Missing required API keys")

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with error handling and retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1y", interval="1d")

                if data.empty:
                    logger.warning(f"No data retrieved for {ticker}")
                    return None

                return data

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {ticker}: {str(e)}")
                if attempt == max_retries - 1:
                    return None

    def get_market_trend(self) -> MarketTrend:
        """Get broader market trend with parallel execution."""
        def fetch_index(symbol: str) -> Tuple[str, pd.DataFrame]:
            return symbol, yf.Ticker(symbol).history(period="1mo")

        try:
            with ThreadPoolExecutor() as executor:
                results = dict(executor.map(
                    lambda x: fetch_index(x),
                    ['^GSPC', '^IXIC']
                ))

            sp500_trend = "Uptrend" if results['^GSPC']['Close'].iloc[-1] > results['^GSPC']['Close'].mean() else "Downtrend"
            nasdaq_trend = "Uptrend" if results['^IXIC']['Close'].iloc[-1] > results['^IXIC']['Close'].mean() else "Downtrend"

            return MarketTrend(sp500_trend, nasdaq_trend)

        except Exception as e:
            logger.error(f"Failed to retrieve market trend: {str(e)}")
            return MarketTrend("Unknown", "Unknown")

    def get_company_name(self, ticker: str) -> str:
        """Get company name from ticker symbol."""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get("shortName", ticker)
        except Exception as e:
            logger.error(f"Failed to retrieve company name for {ticker}: {str(e)}")
            return ticker

    def calculate_technical_indicators(self, stock_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators in one pass."""
        indicators = {
            'jma': self._calculate_jma(stock_data),
            'rsi': self._calculate_rsi(stock_data)
        }
        return indicators

    def _calculate_jma(self, stock_data: pd.DataFrame, period: int = 14, phase: int = 0) -> pd.Series:
        """Calculate Jurik Moving Average with vectorized operations."""
        if 'Close' not in stock_data.columns:
            return pd.Series(index=stock_data.index)

        price = stock_data['Close'].values
        beta = 0.45 * (period - 1) / (0.45 + period - 1)
        alpha = np.exp(-np.sqrt(beta))

        # Vectorized calculation
        jma = np.zeros_like(price)
        jma[0] = price[0]
        mask = np.arange(1, len(price))
        jma[mask] = jma[mask - 1] + (price[mask] - jma[mask - 1]) * (1 + phase / 100) * (1 - alpha)

        return pd.Series(jma, index=stock_data.index)

    def _calculate_rsi(self, stock_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI using vectorized operations."""
        delta = stock_data['Close'].diff()
        gain = (delta.clip(lower=0)).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def fetch_news_sentiment(self, ticker: str) -> NewsAnalysis:
        """Fetch and analyze news with improved error handling."""
        try:
            company_name = self.get_company_name(ticker)
            today = datetime.now()
            last_week = today - timedelta(days=7)

            query = f"{ticker} {company_name}"
            url = (f"https://newsapi.org/v2/everything?q={query}"
                   f"&from={last_week.strftime('%Y-%m-%d')}"
                   f"&to={today.strftime('%Y-%m-%d')}"
                   f"&sortBy=publishedAt&language=en"
                   f"&apiKey={self.news_api_key}")

            response = requests.get(url, timeout=10, verify=False)
            response.raise_for_status()

            articles = response.json().get('articles', [])
            if not articles:
                return NewsAnalysis("Neutral", 0.0, "No recent news found")

            sentiments = [self.sentiment_analyzer.polarity_scores(article['title'])['compound']
                          for article in articles if article.get('title')]

            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_summary = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
            news_summary = "\n".join([f"- {article['title']}" for article in articles[:3] if article['title']])

            return NewsAnalysis(sentiment_summary, avg_sentiment, news_summary)

        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {str(e)}")
            return NewsAnalysis("Neutral", 0.0, "Error fetching news")

    def generate_price_forecast(self, stock_data: pd.DataFrame) -> List[float]:
        """Generate a 7-day price forecast using Exponential Smoothing."""
        if 'Close' not in stock_data.columns or stock_data.empty:
            return []

        # Using the last 60 days for forecasting
        recent_data = stock_data['Close'].tail(60)

        # Fit an Exponential Smoothing model to predict next 7 days
        model = ExponentialSmoothing(recent_data, trend='add', seasonal=None, initialization_method='estimated')
        model_fit = model.fit()
        forecast = model_fit.forecast(7)
        return forecast.tolist()

    def generate_recommendation(self, stock_data: pd.DataFrame, news_analysis: NewsAnalysis, market_trend: MarketTrend) -> TradingRecommendation:
        """Generate trading recommendation with improved logic and clearer explanations."""
        if stock_data.empty or 'Close' not in stock_data.columns:
            return TradingRecommendation("Hold", "Insufficient data", "Neutral", None, None, None)

        latest_close = stock_data['Close'].iloc[-1]
        indicators = self.calculate_technical_indicators(stock_data)

        # Initialize recommendation variables
        recommendation = "Hold"
        suggestion = []
        trend_signal = "Neutral"
        price_target = None
        buy_date = None
        sell_date = None

        # Technical Analysis
        jma_latest = indicators['jma'].iloc[-1]
        rsi_latest = indicators['rsi'].iloc[-1]

        # 1. News Analysis (Highest Priority)
        severe_news_keywords = ["plunge", "drop", "auditor", "loss", "resignation", "scrutiny",
                                "investigation", "fraud", "lawsuit", "default"]

        news_severity = sum(1 for keyword in severe_news_keywords if keyword in news_analysis.news_summary.lower())

        if news_severity > 0:
            recommendation = "Sell"
            suggestion.append("Severe news indicators detected. Recommendation is to SELL.")
        elif latest_close > jma_latest and rsi_latest < 70:
            recommendation = "Buy"
            trend_signal = "Bullish"
            suggestion.append("Price above JMA and RSI below 70. Recommendation is to BUY.")
            buy_date = datetime.now().strftime('%Y-%m-%d')
            price_target = latest_close * 1.05
            sell_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        elif latest_close < jma_latest and rsi_latest > 30:
            recommendation = "Sell"
            trend_signal = "Bearish"
            suggestion.append("Price below JMA and RSI above 30. Recommendation is to SELL.")

        return TradingRecommendation(
            recommendation=recommendation,
            suggestion="\n".join(suggestion),
            trend_signal=trend_signal,
            price_target=price_target,
            buy_date=buy_date,
            sell_date=sell_date
        )

    def send_alert_to_console(self, stock: str, recommendation: TradingRecommendation, current_price: float):
        """Send alert details to console instead of email."""
        print("\n================ ALERT =================")
        print(f"Stock: {stock}")
        print(f"Recommendation: {recommendation.recommendation}")
        print(f"Current Price: ${current_price:.2f}")
        if recommendation.price_target:
            print(f"Price Target: ${recommendation.price_target:.2f}")
        if recommendation.buy_date:
            print(f"Buy Date: {recommendation.buy_date}")
        if recommendation.sell_date:
            print(f"Sell Date: {recommendation.sell_date}")
        print("\nDetailed Analysis:")
        print(recommendation.suggestion)
        print("========================================\n")

    def run_analysis(self):
        """Run complete analysis for a list of stocks and send alerts to console."""
        # Fetching the tickers for major indices (NASDAQ, S&P 500, Dow Jones)
        tickers = self.get_all_tickers()
        for ticker in tickers:
            stock_data = self.fetch_stock_data(ticker)
            if stock_data is None:
                logger.error(f"Could not fetch stock data for {ticker}")
                continue

            market_trend = self.get_market_trend()
            news_analysis = self.fetch_news_sentiment(ticker)
            recommendation = self.generate_recommendation(stock_data, news_analysis, market_trend)
            current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else None

            if current_price:
                self.send_alert_to_console(ticker, recommendation, current_price)

    def get_all_tickers(self) -> List[str]:
        """Get all tickers from major indices."""
        try:
            # Get list of tickers for S&P 500, NASDAQ, and Dow Jones
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', verify=False)[0][
                'Symbol'].tolist()
            nasdaq = pd.read_csv('https://datahub.io/core/nasdaq-listings/r/nasdaq-listed-symbols.csv', verify=False)[
                'Symbol'].tolist()
            dow_jones = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', verify=False)[1][
                'Symbol'].tolist()

            # Combine all tickers and remove duplicates
            tickers = list(set(sp500 + nasdaq + dow_jones))
            return tickers

        except Exception as e:
            logger.error(f"Failed to fetch tickers from major indices: {str(e)}")
            return []

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run_analysis()
