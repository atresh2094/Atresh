from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Optional, Dict, List  # Corrected import for List and Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API keys (replace with your own)
ALPHA_VANTAGE_API_KEY = 'TX0HJLIXT7XMCQ9B'
NEWS_API_KEY = '7665dc8b93f943ea88ad3ad53381aaec'


@dataclass
class MarketTrend:
    sp500: str
    nasdaq: str
    sector_performance: str

@dataclass
class NewsAnalysis:
    sentiment_summary: str
    avg_sentiment: float
    significant_events: List[str]

@dataclass
class TradingRecommendation:
    recommendation: str
    detailed_analysis: str
    entry_price: Optional[float]
    exit_price: Optional[float]
    target_return: Optional[float]

class StockAnalyzer:
    def __init__(self):
        self.alpha_vantage_api_key = ALPHA_VANTAGE_API_KEY
        self.news_api_key = NEWS_API_KEY
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical stock data."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y", interval="1d")
            if data.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None

    def get_market_trend(self) -> MarketTrend:
        """Analyze broader market trends and sector performance."""
        try:
            sp500 = yf.Ticker('^GSPC').history(period="1mo")['Close']
            nasdaq = yf.Ticker('^IXIC').history(period="1mo")['Close']
            sp500_trend = "Uptrend" if sp500.iloc[-1] > sp500.mean() else "Downtrend"
            nasdaq_trend = "Uptrend" if nasdaq.iloc[-1] > nasdaq.mean() else "Downtrend"
            # Example sector analysis
            sector_performance = "Technology is leading"  # Placeholder
            return MarketTrend(sp500_trend, nasdaq_trend, sector_performance)
        except Exception as e:
            logger.error(f"Failed to fetch market trend: {e}")
            return MarketTrend("Unknown", "Unknown", "Unknown")

    def fetch_news_sentiment(self, ticker: str) -> NewsAnalysis:
        """Analyze news sentiment with event extraction."""
        try:
            today = datetime.now()
            last_week = today - timedelta(days=7)
            url = (
                f"https://newsapi.org/v2/everything?q={ticker}&from={last_week.strftime('%Y-%m-%d')}"
                f"&to={today.strftime('%Y-%m-%d')}&sortBy=publishedAt&language=en&apiKey={self.news_api_key}"
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            significant_events = [article['title'] for article in articles if 'earnings' in article.get('title', '').lower()]
            sentiments = [self.sentiment_analyzer.polarity_scores(article['title'])['compound']
                          for article in articles if article.get('title')]
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_summary = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
            return NewsAnalysis(sentiment_summary, avg_sentiment, significant_events)
        except Exception as e:
            logger.error(f"Failed to fetch news sentiment: {e}")
            return NewsAnalysis("Neutral", 0.0, [])

    def calculate_technical_indicators(self, stock_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD, RSI, and Bollinger Bands."""
        close = stock_data['Close']
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        rsi = self.calculate_rsi(stock_data)
        bollinger_mid = close.rolling(window=20).mean()
        bollinger_std = close.rolling(window=20).std()
        bollinger_upper = bollinger_mid + (2 * bollinger_std)
        bollinger_lower = bollinger_mid - (2 * bollinger_std)
        return {
            'macd': macd,
            'signal': signal,
            'rsi': rsi,
            'bollinger_upper': bollinger_upper,
            'bollinger_lower': bollinger_lower
        }

    def calculate_rsi(self, stock_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_recommendation(self, stock_data: pd.DataFrame, news_analysis: NewsAnalysis,
                                market_trend: MarketTrend) -> TradingRecommendation:
        """Provide hedge-fund-grade recommendations."""
        if stock_data.empty or 'Close' not in stock_data.columns:
            return TradingRecommendation("Hold", "Insufficient data for analysis.", None, None, None)

        indicators = self.calculate_technical_indicators(stock_data)
        close = stock_data['Close'].iloc[-1]
        macd = indicators['macd'].iloc[-1]
        signal = indicators['signal'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        bollinger_lower = indicators['bollinger_lower'].iloc[-1]
        bollinger_upper = indicators['bollinger_upper'].iloc[-1]

        logger.info(
            f"Latest MACD: {macd}, Signal: {signal}, RSI: {rsi}, Bollinger Lower: {bollinger_lower}, Bollinger Upper: {bollinger_upper}")

        analysis = []

        # Technical Analysis
        if macd > signal:
            analysis.append("MACD indicates a bullish trend.")
        elif macd < signal:
            analysis.append("MACD indicates a bearish trend.")

        if rsi < 30:
            analysis.append("RSI suggests the stock is oversold, indicating a buying opportunity.")
        elif rsi > 70:
            analysis.append("RSI suggests the stock is overbought, indicating a selling opportunity.")

        if close < bollinger_lower:
            analysis.append("Price is below the Bollinger Bands, indicating potential undervaluation.")
        elif close > bollinger_upper:
            analysis.append("Price is above the Bollinger Bands, indicating potential overvaluation.")

        # Market Trend Influence
        if market_trend.sp500 == "Downtrend" or market_trend.nasdaq == "Downtrend":
            analysis.append("Broader market trend is bearish, exercise caution.")
        elif market_trend.sp500 == "Uptrend" and market_trend.nasdaq == "Uptrend":
            analysis.append("Broader market trend is bullish, supporting a positive outlook.")

        recommendation = "Hold"
        if "bullish" in "".join(analysis).lower():
            recommendation = "Buy"
        elif "bearish" in "".join(analysis).lower():
            recommendation = "Sell"

        target_return = close * 1.1 if recommendation == "Buy" else None
        return TradingRecommendation(
            recommendation=recommendation,
            detailed_analysis="\n".join(analysis) if analysis else "No strong signals identified.",
            entry_price=close if recommendation == "Buy" else None,
            exit_price=target_return,
            target_return=target_return
        )

    def run_analysis(self):
        """Run analysis for a given stock ticker."""
        ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip()
        if not ticker:
            logger.error("No ticker provided.")
            return
        stock_data = self.fetch_stock_data(ticker)
        if stock_data is None:
            logger.error(f"Failed to fetch data for {ticker}")
            return
        market_trend = self.get_market_trend()
        news_analysis = self.fetch_news_sentiment(ticker)
        recommendation = self.generate_recommendation(stock_data, news_analysis, market_trend)
        print("\n=== Recommendation ===")
        print(f"Ticker: {ticker}")
        print(f"Recommendation: {recommendation.recommendation}")
        print(f"Analysis: {recommendation.detailed_analysis}")
        if recommendation.entry_price:
            print(f"Entry Price: ${recommendation.entry_price:.2f}")
        if recommendation.exit_price:
            print(f"Exit Price: ${recommendation.exit_price:.2f}")
        print("======================")

if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run_analysis()
