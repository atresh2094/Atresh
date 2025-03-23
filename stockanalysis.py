import os
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import colorama
from colorama import Fore, Style
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Configure logging
logging.basicConfig(
    level=logging.ERROR,  # Set to ERROR to suppress INFO logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded API keys (as per your request)
ALPHA_VANTAGE_API_KEY = 'TX0HJLIXT7XMCQ9B'
NEWS_API_KEY = '7665dc8b93f943ea88ad3ad53381aaec'

colorama.init(autoreset=True)

@dataclass
class MarketTrend:
    sp500: str
    nasdaq: str
    nse: Optional[str] = None  # Add NSE trend

@dataclass
class NewsAnalysis:
    sentiment_summary: str
    avg_sentiment: float
    news_summary: str
    industry_sentiment_summary: Optional[str] = None
    industry_avg_sentiment: Optional[float] = None
    industry_news_summary: Optional[str] = None

@dataclass
class TradingRecommendation:
    recommendation: str
    suggestion: str
    trend_signal: str
    price_target: Optional[float]
    next_7_days_forecast: Optional[List[float]]
    rationale: Optional[List[str]] = None
    risk_score: Optional[float] = None
    var_95: Optional[float] = None  # Value at Risk at 95% confidence
    stop_loss: Optional[float] = None

    def calculate_risk_score(self, market_trend: MarketTrend, news_analysis: NewsAnalysis, indicators: Dict[str, pd.Series]):
        risk = 0
        if market_trend.sp500 == "Downtrend" or market_trend.nasdaq == "Downtrend" or (market_trend.nse == "Downtrend" if market_trend.nse else False):
            risk += 2
        if news_analysis.sentiment_summary == "Negative" or (news_analysis.industry_sentiment_summary and news_analysis.industry_sentiment_summary == "Negative"):
            risk += 3
        if indicators['rsi'].iloc[-1] > 70:
            risk += 2
        self.risk_score = min(10, max(1, 10 - risk))

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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="2y", interval="1d")

                if data.empty:
                    logger.warning(f"No data retrieved for {ticker}")
                    return None

                # Fetch additional fundamental data
                stock_info = stock.info
                data['Revenue'] = stock_info.get('totalRevenue', None)
                data['NetIncome'] = stock_info.get('netIncomeToCommon', None)
                data['DebtToEquity'] = stock_info.get('debtToEquity', None)
                data['PE_Ratio'] = stock_info.get('trailingPE', None)
                data['PB_Ratio'] = stock_info.get('priceToBook', None)
                data['DividendYield'] = stock_info.get('dividendYield', None)
                data['EarningsGrowth'] = stock_info.get('earningsGrowth', None)
                data['ProfitMargin'] = stock_info.get('profitMargins', None)
                data['OperatingMargin'] = stock_info.get('operatingMargins', None)

                return data

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {ticker}: {str(e)}")
                if attempt == max_retries - 1:
                    return None

    def get_market_trend(self, is_nse: bool = False) -> MarketTrend:
        def fetch_index(symbol: str) -> Tuple[str, pd.DataFrame]:
            return symbol, yf.Ticker(symbol).history(period="1mo")

        try:
            indices = ['^GSPC', '^IXIC']
            if is_nse:
                indices.append('^NSEI')  # NSE Nifty 50 Index symbol

            with ThreadPoolExecutor() as executor:
                results = dict(executor.map(
                    lambda x: fetch_index(x),
                    indices
                ))

            sp500_trend = "Uptrend" if results['^GSPC']['Close'].iloc[-1] > results['^GSPC']['Close'].mean() else "Downtrend"
            nasdaq_trend = "Uptrend" if results['^IXIC']['Close'].iloc[-1] > results['^IXIC']['Close'].mean() else "Downtrend"
            nse_trend = None

            if is_nse:
                nse_trend = "Uptrend" if results['^NSEI']['Close'].iloc[-1] > results['^NSEI']['Close'].mean() else "Downtrend"

            return MarketTrend(sp500_trend, nasdaq_trend, nse_trend)

        except Exception as e:
            logger.error(f"Failed to retrieve market trend: {str(e)}")
            return MarketTrend("Unknown", "Unknown", "Unknown")

    def get_company_name(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get("shortName", ticker)
        except Exception as e:
            logger.error(f"Failed to retrieve company name for {ticker}: {str(e)}")
            return ticker

    def calculate_technical_indicators(self, stock_data: pd.DataFrame) -> Dict[str, pd.Series]:
        indicators = {
            'jma': self._calculate_jma(stock_data),
            'heikin_ashi': self._calculate_heikin_ashi(stock_data),
            'rsi': self._calculate_rsi(stock_data),
            'reflected_ema_difference': self._calculate_reflected_ema_difference(stock_data),
            'trend_tide_oscillator': self._calculate_trend_tide_oscillator(stock_data),
            'bollinger_bands': self._calculate_bollinger_bands(stock_data),
            'macd': self._calculate_macd(stock_data),
            'obv': self._calculate_obv(stock_data)
        }
        return indicators

    def _calculate_jma(self, stock_data: pd.DataFrame, period: int = 14, phase: int = 0) -> pd.Series:
        if 'Close' not in stock_data.columns:
            return pd.Series(index=stock_data.index)

        price = stock_data['Close'].values
        beta = 0.45 * (period - 1) / (0.45 + period - 1)
        alpha = np.exp(-np.sqrt(beta))
        jma = np.zeros_like(price)
        jma[0] = price[0]
        for i in range(1, len(price)):
            jma[i] = (1 - alpha) * price[i] + alpha * jma[i - 1]
        return pd.Series(jma, index=stock_data.index)

    def _calculate_heikin_ashi(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        ha = pd.DataFrame(index=stock_data.index)

        ha['Close'] = (stock_data['Open'] + stock_data['High'] +
                       stock_data['Low'] + stock_data['Close']) / 4
        ha['Open'] = (stock_data['Open'].shift(1) + stock_data['Close'].shift(1)) / 2
        ha['High'] = stock_data[['High', 'Open', 'Close']].max(axis=1)
        ha['Low'] = stock_data[['Low', 'Open', 'Close']].min(axis=1)

        return ha.dropna()

    def _calculate_rsi(self, stock_data: pd.DataFrame, window: int = 14) -> pd.Series:
        delta = stock_data['Close'].diff()
        gain = (delta.clip(lower=0)).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_reflected_ema_difference(self, stock_data: pd.DataFrame, short_window: int = 12, long_window: int = 26) -> pd.Series:
        if 'Close' not in stock_data.columns:
            return pd.Series(index=stock_data.index)

        short_ema = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = stock_data['Close'].ewm(span=long_window, adjust=False).mean()
        return short_ema - long_ema

    def _calculate_trend_tide_oscillator(self, stock_data: pd.DataFrame, length: int = 14, smoothing: int = 3) -> pd.Series:
        if 'Close' not in stock_data.columns:
            return pd.Series(index=stock_data.index)

        oscillator = stock_data['Close'].diff(length).fillna(0).rolling(smoothing).mean()
        return oscillator

    def _calculate_bollinger_bands(self, stock_data: pd.DataFrame, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
        if 'Close' not in stock_data.columns:
            return pd.DataFrame(index=stock_data.index)

        rolling_mean = stock_data['Close'].rolling(window=window).mean()
        rolling_std = stock_data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)

        return pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band}, index=stock_data.index)

    def _calculate_macd(self, stock_data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
        if 'Close' not in stock_data.columns:
            return pd.DataFrame(index=stock_data.index)

        short_ema = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = stock_data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        histogram = macd - signal
        return pd.DataFrame({'MACD': macd, 'Signal': signal, 'Histogram': histogram}, index=stock_data.index)

    def _calculate_obv(self, stock_data: pd.DataFrame) -> pd.Series:
        if 'Close' not in stock_data.columns or 'Volume' not in stock_data.columns:
            return pd.Series(index=stock_data.index)

        obv = [0]
        for i in range(1, len(stock_data)):
            if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + stock_data['Volume'].iloc[i])
            elif stock_data['Close'].iloc[i] < stock_data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - stock_data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=stock_data.index)

    def fetch_news_sentiment(self, ticker: str, is_nse: bool = False) -> NewsAnalysis:
        try:
            company_name = self.get_company_name(ticker)
            today = datetime.now()
            last_week = today - timedelta(days=7)

            query = f"{ticker} {company_name}"
            if is_nse:
                query += " NSE"

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

            # Advanced Sentiment Analysis using TextBlob
            sentiments = [TextBlob(article['title']).sentiment.polarity
                          for article in articles if article.get('title')]

            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_summary = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
            news_summary = "\n".join([f"- {article['title']}" for article in articles[:3] if article['title']])

            # Fetch industry-level news sentiment
            industry_query = f"{company_name} industry"
            if is_nse:
                industry_query += " India"

            industry_url = (f"https://newsapi.org/v2/everything?q={industry_query}"
                            f"&from={last_week.strftime('%Y-%m-%d')}"
                            f"&to={today.strftime('%Y-%m-%d')}"
                            f"&sortBy=publishedAt&language=en"
                            f"&apiKey={self.news_api_key}")

            industry_response = requests.get(industry_url, timeout=10)
            industry_response.raise_for_status()

            industry_articles = industry_response.json().get('articles', [])
            industry_sentiments = [TextBlob(article['title']).sentiment.polarity
                                   for article in industry_articles if article.get('title')]

            industry_avg_sentiment = np.mean(industry_sentiments) if industry_sentiments else 0
            industry_sentiment_summary = "Positive" if industry_avg_sentiment > 0.05 else "Negative" if industry_avg_sentiment < -0.05 else "Neutral"
            industry_news_summary = "\n".join([f"- {article['title']}" for article in industry_articles[:3] if article['title']])

            return NewsAnalysis(sentiment_summary, avg_sentiment, news_summary,
                                industry_sentiment_summary, industry_avg_sentiment, industry_news_summary)

        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {str(e)}")
            return NewsAnalysis("Neutral", 0.0, "Error fetching news")

    def generate_price_forecast(self, stock_data: pd.DataFrame) -> List[float]:
        # Advanced Forecasting using ARIMA
        if 'Close' not in stock_data.columns or stock_data.empty:
            return []

        data = stock_data['Close'].dropna()
        try:
            # Check for stationarity
            result = adfuller(data)
            p_value = result[1]
            d = 0 if p_value < 0.05 else 1  # Differencing order

            model = ARIMA(data, order=(5, d, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)
            return forecast.tolist()
        except Exception as e:
            logger.error(f"ARIMA model failed: {str(e)}")
            return []

    def calculate_var(self, stock_data: pd.DataFrame, confidence_level=0.95) -> float:
        # Value at Risk calculation
        returns = stock_data['Close'].pct_change().dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def calculate_stop_loss(self, stock_data: pd.DataFrame) -> float:
        # Stop-loss suggestion based on recent volatility
        recent_volatility = stock_data['Close'].pct_change().rolling(window=14).std().iloc[-1]
        last_price = stock_data['Close'].iloc[-1]
        stop_loss = last_price - (2 * recent_volatility * last_price)
        return stop_loss

    def generate_recommendation(self, stock_data: pd.DataFrame, news_analysis: NewsAnalysis, market_trend: MarketTrend) -> TradingRecommendation:
        if stock_data.empty or 'Close' not in stock_data.columns:
            return TradingRecommendation("Hold", "Insufficient data", "Neutral", None, None)

        latest_close = stock_data['Close'].iloc[-1]
        indicators = self.calculate_technical_indicators(stock_data)

        recommendation = "Hold"
        suggestion = []
        trend_signal = "Neutral"
        price_target = None
        decision_factors = []
        rationale = []

        # Technical Indicators
        rsi_latest = indicators['rsi'].iloc[-1]
        macd = indicators['macd']
        macd_latest = macd['MACD'].iloc[-1]
        macd_signal_latest = macd['Signal'].iloc[-1]
        obv_latest = indicators['obv'].iloc[-1]
        obv_prev = indicators['obv'].iloc[-2]

        # Fundamental Factors
        pe_ratio = stock_data['PE_Ratio'].iloc[-1]
        profit_margin = stock_data['ProfitMargin'].iloc[-1]

        score = 0

        # RSI Analysis
        if rsi_latest < 30:
            score += 1  # Bullish
            rationale.append("RSI indicates oversold conditions.")
        elif rsi_latest > 70:
            score -= 1  # Bearish
            rationale.append("RSI indicates overbought conditions.")
        else:
            rationale.append("RSI is in neutral range.")

        # MACD Analysis
        if macd_latest > macd_signal_latest:
            score += 1  # Bullish
            rationale.append("MACD indicates bullish momentum.")
        else:
            score -= 1  # Bearish
            rationale.append("MACD indicates bearish momentum.")

        # OBV Analysis
        if obv_latest > obv_prev:
            score += 1  # Bullish
            rationale.append("OBV shows increasing buying pressure.")
        else:
            score -= 1  # Bearish
            rationale.append("OBV shows increasing selling pressure.")

        # Fundamental Analysis
        if pe_ratio and pe_ratio < 15:
            score += 1  # Undervalued
            rationale.append("P/E ratio suggests the stock is undervalued.")
        elif pe_ratio and pe_ratio > 25:
            score -= 1  # Overvalued
            rationale.append("P/E ratio suggests the stock is overvalued.")

        if profit_margin and profit_margin > 0.2:
            score += 1  # Strong profitability
            rationale.append("High profit margin indicates strong profitability.")

        # News Sentiment
        if news_analysis.sentiment_summary == "Positive":
            score += 1
            rationale.append("Positive news sentiment supports bullish outlook.")
        elif news_analysis.sentiment_summary == "Negative":
            score -= 1
            rationale.append("Negative news sentiment indicates caution.")

        # Market Trend
        market_trend_score = 0
        if market_trend.sp500 == "Uptrend":
            market_trend_score += 0.5
        elif market_trend.sp500 == "Downtrend":
            market_trend_score -= 0.5

        if market_trend.nasdaq == "Uptrend":
            market_trend_score += 0.5
        elif market_trend.nasdaq == "Downtrend":
            market_trend_score -= 0.5

        if market_trend.nse:
            if market_trend.nse == "Uptrend":
                market_trend_score += 1
                rationale.append("NSE market is in an uptrend.")
            elif market_trend.nse == "Downtrend":
                market_trend_score -= 1
                rationale.append("NSE market is in a downtrend.")

        score += market_trend_score

        # Determine recommendation based on score
        if score >= 2:
            recommendation = "Buy"
            trend_signal = "Bullish"
        elif score <= -2:
            recommendation = "Sell"
            trend_signal = "Bearish"
        else:
            recommendation = "Hold"
            trend_signal = "Neutral"

        # Risk Management
        var_95 = self.calculate_var(stock_data)
        stop_loss = self.calculate_stop_loss(stock_data)

        suggestion.append(f"Value at Risk (95% confidence): {var_95:.2%}")
        suggestion.append(f"Suggested Stop-Loss Price: {stop_loss:.2f}")

        # Price Target (Example using recent highs)
        recent_high = stock_data['High'][-20:].max()
        price_target = recent_high
        suggestion.append(f"Price Target: {price_target:.2f}")

        next_7_days_forecast = self.generate_price_forecast(stock_data)

        trading_rec = TradingRecommendation(
            recommendation=recommendation,
            suggestion="\n".join(suggestion),
            trend_signal=trend_signal,
            price_target=price_target,
            next_7_days_forecast=next_7_days_forecast,
            rationale=rationale,
            var_95=var_95,
            stop_loss=stop_loss
        )

        trading_rec.calculate_risk_score(market_trend, news_analysis, indicators)
        return trading_rec

    def run_analysis(self, ticker: str) -> Optional[Dict[str, Optional[object]]]:
        try:
            is_nse = ticker.endswith('.NS')
            stock_data = self.fetch_stock_data(ticker)
            if stock_data is None:
                logger.error(f"Could not fetch stock data for {ticker}")
                return None

            market_trend = self.get_market_trend(is_nse=is_nse)
            news_analysis = self.fetch_news_sentiment(ticker, is_nse=is_nse)
            recommendation = self.generate_recommendation(stock_data, news_analysis, market_trend)

            return {
                'company_name': self.get_company_name(ticker),
                'market_trend': market_trend,
                'news_analysis': news_analysis,
                'recommendation': recommendation,
                'last_price': stock_data['Close'].iloc[-1] if not stock_data.empty else None,
                'is_nse': is_nse
            }

        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {str(e)}")
            return None

def main():
    try:
        analyzer = StockAnalyzer()

        while True:
            ticker = input("\nEnter stock ticker symbol (or 'quit' to exit): ").upper()
            if ticker.lower() == 'quit':
                break

            print(f"\nAnalyzing {ticker}...\n")
            results = analyzer.run_analysis(ticker)

            if results:
                currency_symbol = "₹" if results['is_nse'] else "$"
                market_trend = results['market_trend']
                print("\n" + "=" * 80)
                print(f"Analysis Results for {results['company_name']} ({ticker})")
                print("=" * 80)
                print(f"Last Price: {currency_symbol}{results['last_price']:.2f}")
                if results['is_nse']:
                    print(f"Market Trend - NSE: {market_trend.nse}")
                else:
                    print(f"Market Trend - S&P 500: {market_trend.sp500}, NASDAQ: {market_trend.nasdaq}")
                print(f"News Sentiment: {results['news_analysis'].sentiment_summary}")
                print(f"Recommendation: {results['recommendation'].recommendation}")
                print(f"Trend Signal: {results['recommendation'].trend_signal}")
                print(f"Risk Score: {results['recommendation'].risk_score}/10")
                print(f"Value at Risk (95% confidence): {results['recommendation'].var_95:.2%}")
                print(f"Suggested Stop-Loss Price: {currency_symbol}{results['recommendation'].stop_loss:.2f}")
                print(f"Price Target: {currency_symbol}{results['recommendation'].price_target:.2f}")

                print("\nSuggestion Details:")
                print(results['recommendation'].suggestion)
                print("\nRationale:")
                for reason in results['recommendation'].rationale:
                    print(f"- {reason}")
                if results['recommendation'].next_7_days_forecast:
                    print("\n📊 Next 7-Days Price Forecast:")
                    for i, price in enumerate(results['recommendation'].next_7_days_forecast, start=1):
                        print(f"Day {i}: {currency_symbol}{price:.2f}")
                print("=" * 80)
            else:
                print(f"Failed to analyze {ticker}. Please check the ticker symbol and try again.")

    except ConfigError as ce:
        logger.error(f"Configuration Error: {str(ce)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
