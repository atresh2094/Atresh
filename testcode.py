import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests


# Define function to fetch stock data
def fetch_data(ticker):
    """
    Fetches historical stock data for the ticker using Yahoo Finance.
    """
    data = yf.download(ticker, period="1y", interval="1d")
    return data


# Define function to calculate technical indicators
def calculate_indicators(data):
    """
    Calculates RSI, MACD, SMA, and volume-related indicators for the stock data.
    """
    # Ensure 'Close' is a one-dimensional series
    close_prices = data['Close'].squeeze()

    # RSI and MACD indicators
    data['RSI'] = RSIIndicator(close_prices).rsi()
    macd = MACD(close_prices)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()

    # Moving Averages
    data['SMA_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
    data['SMA_200'] = SMAIndicator(close_prices, window=200).sma_indicator()

    # Volume and Price Spikes
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Avg_Volume'] = data['Volume'].rolling(window=20).mean().fillna(method='bfill')
    data['Volume_Spike'] = data['Volume'] > 1.5 * data['Avg_Volume']
    data['Price_Change'] = data['Close'].pct_change()
    return data


# Define function for sentiment analysis using News API
def get_news_sentiment(ticker):
    """
    Fetches recent news headlines for the ticker and calculates average sentiment score.
    """
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url).json()

    # Set up sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(article['title'])['compound'] for article in response['articles']]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment


# Define the main function for stock analysis
def main(ticker):
    """
    Main function to fetch data, calculate indicators, get sentiment, and make buy decision.
    """
    data = fetch_data(ticker)
    data = calculate_indicators(data)
    sentiment_score = get_news_sentiment(ticker)

    # Latest data point for decision-making
    latest = data.iloc[-1]

    # Define buy criteria
    buy_conditions = (
            latest['RSI'] < 30 and  # RSI indicates oversold
            latest['MACD'] > latest['MACD_Signal'] and  # MACD crossover
            sentiment_score > 0.2 and  # Positive sentiment score
            latest['Volume_Spike'] and  # Volume spike
            latest['SMA_50'] > latest['SMA_200']  # Bullish SMA crossover
    )

    # Decision output
    if buy_conditions:
        print(f"Buy signal for {ticker}: All criteria met.")
    else:
        print(f"No buy signal for {ticker}.")


# Example usage
ticker = 'AAPL'
main(ticker)
