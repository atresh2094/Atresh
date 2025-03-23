import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime

# Constants for analysis
SYMBOL = input('Enter the stock symbol: ').upper()  # Get stock symbol from user input
YOUR_AVG_PRICE = float(input('Enter your average purchase price for the stock: '))  # Get average price from user input

# Broader US Markets Indices
MARKET_SYMBOLS = ['^GSPC', '^DJI', '^IXIC']


# Function to get stock data from Yahoo Finance
def get_stock_data(symbol, period='6mo', interval='1d'):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)
    return data


# Function to get the current price of the stock
def get_current_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    return data['Close'].iloc[-1]


# Function to get broader market analysis
def broader_market_analysis(symbols):
    overall_sentiment = "neutral"
    for symbol in symbols:
        market_data = get_stock_data(symbol)
        market_return = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0]

        if market_return > 0.05:
            overall_sentiment = "bullish"
        elif market_return < -0.05:
            overall_sentiment = "bearish"
    return overall_sentiment


# Function to calculate Jurik Moving Average (JMA)
def calculate_jma(data, length=14, phase=0):
    # Simplified approximation of JMA using exponential weighted moving average
    data['JMA'] = data['Close'].ewm(span=length, adjust=False).mean()
    return data


# Function to calculate Heikin Ashi Candlesticks
def calculate_heikin_ashi(data):
    ha_data = pd.DataFrame(index=data.index, columns=['HA_Open', 'HA_High', 'HA_Low', 'HA_Close'])
    ha_data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    ha_data['HA_Open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    ha_data.loc[ha_data.index[0], 'HA_Open'] = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2
    ha_data['HA_High'] = ha_data[['HA_Open', 'HA_Close']].join(data[['High']]).max(axis=1)
    ha_data['HA_Low'] = ha_data[['HA_Open', 'HA_Close']].join(data[['Low']]).min(axis=1)
    return ha_data


# Function to get technical indicators for stock
def analyze_technical_indicators(data):
    # Calculating Moving Averages (SMA and EMA)
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = (100 - (100 / (1 + data['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / data[
        'Close'].diff().apply(lambda x: abs(x)).rolling(window=14).mean())))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate JMA
    data = calculate_jma(data)

    # Calculate Heikin Ashi
    ha_data = calculate_heikin_ashi(data)

    # Determine signals
    sma_signal = "bullish" if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1] else "bearish"
    rsi_signal = "oversold" if data['RSI'].iloc[-1] < 30 else "overbought" if data['RSI'].iloc[-1] > 70 else "neutral"
    macd_signal = "bullish" if data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1] else "bearish"
    jma_signal = "bullish" if data['JMA'].iloc[-1] > data['JMA'].iloc[-2] else "bearish"
    ha_signal = "bullish" if ha_data['HA_Close'].iloc[-1] > ha_data['HA_Open'].iloc[-1] else "bearish"

    return sma_signal, rsi_signal, macd_signal, jma_signal, ha_signal


# Function to get recent market news sentiment for stock
def get_market_news_sentiment(symbol):
    url = f'https://finance.yahoo.com/quote/{symbol}/news?p={symbol}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract news headlines
    headlines = [item.text for item in soup.find_all('h3')]

    positive, negative = 0, 0
    for headline in headlines:
        analysis = TextBlob(headline)
        if analysis.sentiment.polarity > 0:
            positive += 1
        elif analysis.sentiment.polarity < 0:
            negative += 1

    if positive > negative:
        return "positive"
    elif negative > positive:
        return "negative"
    else:
        return "neutral"


# Function to forecast stock price for the next 7 days (simple approach using average change)
def forecast_stock_price(data):
    data['Daily_Change'] = data['Close'].pct_change()
    avg_daily_change = data['Daily_Change'].mean()
    last_close = data['Close'].iloc[-1]
    forecast = []
    for i in range(1, 8):
        forecast_price = last_close * (1 + avg_daily_change) ** i
        forecast.append(
            (datetime.datetime.now() + datetime.timedelta(days=i)).strftime('%Y-%m-%d') + f": ${forecast_price:.2f}")
    return forecast


# Function to make final recommendation
def make_recommendation(avg_price, current_price, sma_signal, rsi_signal, macd_signal, jma_signal, ha_signal,
                        news_sentiment, market_sentiment):
    recommendation = ""
    analysis = ""

    if current_price >= 2 * avg_price:
        recommendation = "SELL"
        analysis = "You have doubled your investment. Consider booking profits."
    elif market_sentiment == "bearish":
        recommendation = "HOLD"
        analysis = "Broader market is bearish. Consider waiting for better conditions."
    elif sma_signal == "bullish" and rsi_signal == "oversold" and news_sentiment == "positive" and market_sentiment == "bullish" and jma_signal == "bullish" and ha_signal == "bullish":
        recommendation = "ACCUMULATE"
        analysis = "Strong indicators to buy more shares based on moving averages, RSI, JMA, Heikin Ashi, and positive news sentiment."
    elif sma_signal == "bearish" or rsi_signal == "overbought" or news_sentiment == "negative" or macd_signal == "bearish" or jma_signal == "bearish" or ha_signal == "bearish":
        recommendation = "HOLD/SELL"
        analysis = "Mixed or negative indicators suggest caution. Consider holding or selling."
    else:
        recommendation = "HOLD"
        analysis = "No clear indicators to make a decisive move."

    return recommendation, analysis


# Main logic
if __name__ == "__main__":
    # Get stock data
    stock_data = get_stock_data(SYMBOL)

    # Get current price
    CURRENT_PRICE = get_current_price(SYMBOL)

    # Analyze broader market
    market_sentiment = broader_market_analysis(MARKET_SYMBOLS)

    # Get technical indicators
    sma_signal, rsi_signal, macd_signal, jma_signal, ha_signal = analyze_technical_indicators(stock_data)

    # Get news sentiment
    news_sentiment = get_market_news_sentiment(SYMBOL)

    # Forecast stock price for next 7 days
    forecast = forecast_stock_price(stock_data)

    # Make recommendation
    recommendation, analysis = make_recommendation(YOUR_AVG_PRICE, CURRENT_PRICE, sma_signal, rsi_signal, macd_signal,
                                                   jma_signal, ha_signal, news_sentiment, market_sentiment)

    # Output the result
    print(f"\n--- Stock Analysis Report for {SYMBOL} ---\n")
    print(f"Current Price: ${CURRENT_PRICE:.2f}")
    print(f"Your Average Purchase Price: ${YOUR_AVG_PRICE:.2f}")
    print(f"\nRecommendation: {recommendation}")
    print(f"Analysis: {analysis}")
    print(f"\nMarket Sentiment: {market_sentiment}")
    print(f"News Sentiment: {news_sentiment}")
    print(f"\nTechnical Indicators:")
    print(f"- SMA Signal: {sma_signal}")
    print(f"- RSI Signal: {rsi_signal}")
    print(f"- MACD Signal: {macd_signal}")
    print(f"- JMA Signal: {jma_signal}")
    print(f"- Heikin Ashi Signal: {ha_signal}")
    print(f"\nStock Forecast for Next 7 Days:")
    for day in forecast:
        print(f"- {day}")
    print(f"\nFinal Analysis and Recommendation: {recommendation}")
