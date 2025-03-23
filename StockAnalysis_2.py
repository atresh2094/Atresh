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
    next_7_days_forecast: Optional[List[float]]
    rationale: Optional[List[str]] = None

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
                if ticker.endswith('.NS'):  # NSE stock
                    data = yf.download(ticker, period="1y", interval="1d")
                else:  # US stock
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
            'heikin_ashi': self._calculate_heikin_ashi(stock_data),
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

    def _calculate_heikin_ashi(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin Ashi with vectorized operations."""
        ha = pd.DataFrame(index=stock_data.index)

        ha['Close'] = (stock_data['Open'] + stock_data['High'] +
                       stock_data['Low'] + stock_data['Close']) / 4
        ha['Open'] = (stock_data['Open'].shift(1) + stock_data['Close'].shift(1)) / 2
        ha['High'] = stock_data[['High', 'Open', 'Close']].max(axis=1)
        ha['Low'] = stock_data[['Low', 'Open', 'Close']].min(axis=1)

        return ha.dropna()

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

            response = requests.get(url, timeout=10)
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
            return TradingRecommendation("Hold", "Insufficient data", "Neutral", None, None)

        latest_close = stock_data['Close'].iloc[-1]
        indicators = self.calculate_technical_indicators(stock_data)

        # Initialize recommendation variables
        recommendation = "Hold"
        suggestion = []
        trend_signal = "Neutral"
        price_target = None
        decision_factors = []  # Track reasoning behind decisions
        rationale = []  # Track detailed rationale behind recommendation

        # Technical Analysis
        jma_latest = indicators['jma'].iloc[-1]
        ha_latest = indicators['heikin_ashi']['Close'].iloc[-1]
        rsi_latest = indicators['rsi'].iloc[-1]

        # 1. News Analysis (Highest Priority)
        severe_news_keywords = ["plunge", "drop", "auditor", "loss", "resignation", "scrutiny",
                                "investigation", "fraud", "lawsuit", "default"]

        news_severity = sum(1 for keyword in severe_news_keywords if keyword in news_analysis.news_summary.lower())

        if news_severity > 0:
            recommendation = "Sell"
            decision_factors.append({
                'factor': 'Severe News',
                'impact': 'High Priority Override',
                'details': f"Detected {news_severity} severe news indicators - This takes precedence over technical signals for risk management"
            })
            rationale.append("Severe news detected, indicating high risk. Prioritizing SELL recommendation for risk mitigation.")

        # 2. Technical Analysis
        if latest_close < jma_latest:
            trend_signal = "Bearish"
            decision_factors.append({
                'factor': 'Price Trend',
                'impact': 'Primary Indicator',
                'details': "Price below JMA indicates downward trend"
            })
            rationale.append("Price below JMA suggests bearish sentiment. Considering HOLD or SELL based on risk factors.")

            if rsi_latest < 30:
                decision_factors.append({
                    'factor': 'RSI',
                    'impact': 'Secondary Indicator',
                    'details': "Oversold conditions detected (RSI < 30)"
                })
                if recommendation != "Sell":  # Don't override severe news
                    recommendation = "Hold"
                rationale.append("RSI below 30 indicates oversold conditions, suggesting potential for price rebound. HOLD recommended.")

        elif latest_close > jma_latest:
            trend_signal = "Bullish"
            decision_factors.append({
                'factor': 'Price Trend',
                'impact': 'Primary Indicator',
                'details': "Price above JMA indicates upward trend"
            })
            rationale.append("Price above JMA indicates bullish sentiment. BUY or HOLD may be considered based on other factors.")

            if rsi_latest > 70:
                decision_factors.append({
                    'factor': 'RSI',
                    'impact': 'Secondary Indicator',
                    'details': "Overbought conditions detected (RSI > 70)"
                })
                if recommendation != "Sell":
                    recommendation = "Sell"
                rationale.append("RSI above 70 indicates overbought conditions, suggesting potential for price correction. SELL recommended.")

        # 3. Market Context
        market_downtrend = (market_trend.sp500 == "Downtrend" and market_trend.nasdaq == "Downtrend")

        if market_downtrend:
            decision_factors.append({
                'factor': 'Market Context',
                'impact': 'Risk Modifier',
                'details': "Broader market downtrend increases risk"
            })
            rationale.append("Broader market is in a downtrend, increasing overall risk. Adjusting recommendation accordingly.")
            if recommendation == "Buy":
                recommendation = "Hold"
                rationale.append("Market downtrend suggests avoiding aggressive buying. HOLD recommended.")

        # 4. Final Decision Logic
        suggestion.append("\n=== Decision Analysis ===")
        for factor in decision_factors:
            suggestion.append(f"\n{factor['factor']} ({factor['impact']}):")
            suggestion.append(f"\u2022 {factor['details']}")

        suggestion.append("\n=== Final Recommendation Explanation ===")
        if recommendation == "Sell":
            if any(f['factor'] == 'Severe News' for f in decision_factors):
                suggestion.append("\u2022 SELL recommendation primarily due to severe news indicators")
                suggestion.append("\u2022 Technical indicators are secondary when severe risks are detected")
            else:
                suggestion.append("\u2022 SELL recommendation based on technical overbought conditions")
        elif recommendation == "Hold":
            suggestion.append("\u2022 HOLD recommendation due to mixed signals or risk factors")
        elif recommendation == "Buy":
            suggestion.append("\u2022 BUY recommendation based on positive technical indicators")
            suggestion.append("  and absence of significant risk factors")

        # 5. Price Target (if applicable)
        if recommendation in ["Buy", "Hold"]:
            price_target = latest_close * 1.05
            suggestion.append(f"\nPrice Target: ${price_target:.2f}")
            suggestion.append("(Based on 5% upside potential from current price)")

        # 6. Price Forecast
        next_7_days_forecast = self.generate_price_forecast(stock_data)

        return TradingRecommendation(
            recommendation=recommendation,
            suggestion="\n".join(suggestion),
            trend_signal=trend_signal,
            price_target=price_target,
            next_7_days_forecast=next_7_days_forecast,
            rationale=rationale
        )

    def run_analysis(self, ticker: str) -> Optional[Dict[str, Optional[object]]]:
        """Run complete analysis with enhanced output formatting."""
        try:
            stock_data = self.fetch_stock_data(ticker)
            if stock_data is None:
                logger.error(f"Could not fetch stock data for {ticker}")
                return None

            market_trend = self.get_market_trend()
            news_analysis = self.fetch_news_sentiment(ticker)
            recommendation = self.generate_recommendation(stock_data, news_analysis, market_trend)

            return {
                'company_name': self.get_company_name(ticker),
                'market_trend': market_trend,
                'news_analysis': news_analysis,
                'recommendation': recommendation,
                'last_price': stock_data['Close'].iloc[-1] if not stock_data.empty else None
            }

        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {str(e)}")
            return None


def main():
    """Main execution with improved output formatting."""
    try:
        analyzer = StockAnalyzer()

        while True:
            ticker = input("\nEnter stock ticker symbol (or 'quit' to exit): ").upper()
            if ticker.lower() == 'quit':
                break

            print(f"\nAnalyzing {ticker}...")
            results = analyzer.run_analysis(ticker)

            if results:
                print("\n════════════════════════════════════════")
                print(f"Analysis Results for {results['company_name']} ({ticker})")
                print("════════════════════════════════════════")

                print(f"\nCurrent Price: ${results['last_price']:.2f}")

                print("\n▓▓▓ Market Context ▓▓▓")
                print(f"S&P 500: {results['market_trend'].sp500}")
                print(f"NASDAQ: {results['market_trend'].nasdaq}")

                print("\n▓▓▓ News Analysis ▓▓▓")
                print(f"Sentiment: {results['news_analysis'].sentiment_summary}")
                print(f"Sentiment Score: {results['news_analysis'].avg_sentiment:.2f}")
                print("\nRecent Headlines:")
                print(results['news_analysis'].news_summary)

                print("\n▓▓▓ Technical Analysis & Recommendation ▓▓▓")
                print(f"Signal: {results['recommendation'].trend_signal}")
                print(f"Recommendation: {results['recommendation'].recommendation}")
                print("\nDetailed Analysis:")
                print(results['recommendation'].suggestion)

                if results['recommendation'].rationale:
                    print("\nRationale:")
                    for reason in results['recommendation'].rationale:
                        print(f"- {reason}")

                if results['recommendation'].next_7_days_forecast:
                    print("\nNext 7-Days Price Forecast:")
                    for i, price in enumerate(results['recommendation'].next_7_days_forecast, start=1):
                        print(f"Day {i}: ${price:.2f}")

                print("\n════════════════════════════════════════\n")
            else:
                print(f"Unable to complete analysis for {ticker}")

    except KeyboardInterrupt:
        print("\nAnalysis terminated by user")
    except Exception as e:
        logger.error(f"Program error: {str(e)}")
        print("\nAn error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()
