from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
import yfinance as yf
import streamlit as st

import config


@st.cache_data(ttl=60)
def fetch_market_dashboard_data(
        timeframe_label: str = "1 Day",
        ticker_symbol: str = "GC=F"
) -> Tuple[pd.DataFrame, float, List[Dict[str, Any]], float]:

    interval_code = config.AVAILABLE_TIMEFRAMES.get(timeframe_label, "1d")
    lookback = _get_data_lookback(interval_code)

    market_dataframe = yf.download(
        tickers=ticker_symbol,
        period=lookback,
        interval=interval_code,
        auto_adjust=False
    )

    if market_dataframe.empty:
        return pd.DataFrame(), 0.0, [], 0.0

    # Clean multi-index headers if present
    if isinstance(market_dataframe.columns, pd.MultiIndex):
        market_dataframe.columns = market_dataframe.columns.get_level_values(0)

    # Standardize and enrich data
    market_dataframe = market_dataframe.ffill().dropna()
    market_dataframe = market_dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]
    market_dataframe = _enrich_with_technical_indicators(market_dataframe)

    # Prepare return values
    current_market_price = float(market_dataframe['Close'].iloc[-1])
    market_news = _fetch_asset_news(ticker_symbol)
    year_to_date_price = _fetch_year_start_price(market_dataframe, ticker_symbol)

    return market_dataframe, current_market_price, market_news, year_to_date_price


def _get_data_lookback(interval_code: str) -> str:
    #Determines appropriate history period for yfinance based on interval
    if interval_code in ["15m", "1h"]:
        return "60d"
    if interval_code == "1wk":
        return "10y"
    return "max"


def _enrich_with_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical overlays and oscillators."""
    settings = config.IndicatorSettings
    close = dataframe['Close']

    # 1. Trend Indicators (Moving Averages)
    dataframe['MA20'] = close.rolling(window=settings.BOLLINGER_BANDS_PERIOD).mean()
    dataframe['MA50'] = close.rolling(window=settings.MOVING_AVERAGE_LONG_TERM).mean()

    # 2. Volatility (Bollinger Bands)
    std_dev = close.rolling(window=settings.BOLLINGER_BANDS_PERIOD).std()
    dataframe['BB_U'] = dataframe['MA20'] + (std_dev * settings.BOLLINGER_BANDS_STANDARD_DEVIATIONS)
    dataframe['BB_L'] = dataframe['MA20'] - (std_dev * settings.BOLLINGER_BANDS_STANDARD_DEVIATIONS)

    # 3. Momentum (Relative Strength Index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=settings.RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=settings.RSI_PERIOD).mean()

    rs = gain / (loss + 1e-10) # 1e-10 prevents the division by zero error
    dataframe['RSI'] = 100 - (100 / (1 + rs))

    # 4. Trend Strength (MACD)
    ema12 = close.ewm(span=12, adjust=False).mean()# Using EWM (Exponential) instead of SMA to give higher priority to recent price action for trend momentum
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    dataframe['MACD_Hist'] = macd_line - signal_line

    # 5. Strategic Signal Assignment
    dataframe['Buy_Signal'] = (dataframe['RSI'] < settings.RSI_BULLISH_THRESHOLD) & (dataframe['MACD_Hist'] > 0)
    dataframe['Sell_Signal'] = (dataframe['RSI'] > settings.RSI_BEARISH_THRESHOLD) & (dataframe['MACD_Hist'] < 0)

    return dataframe


def calculate_performance_metrics(
        current_price: float,
        history_df: pd.DataFrame,
        ytd_start_price: float
) -> Tuple[float, float, float, float]:

    def _get_return(lookback_days: int) -> float:
        if len(history_df) <= lookback_days:
            return 0.0
        prev_price = history_df['Close'].iloc[-lookback_days]
        return ((current_price - prev_price) / prev_price) * 100

    # Risk Calculation: Annualized Volatility
    daily_returns = history_df['Close'].pct_change().tail(30)
    annualized_volatility = daily_returns.std() * (252 ** 0.5) * 100 #(252 trading days annually on average

    # Return Calculations
    weekly_return = _get_return(5)
    monthly_return = _get_return(21)
    ytd_return = ((current_price - ytd_start_price) / ytd_start_price) * 100

    return weekly_return, monthly_return, ytd_return, annualized_volatility


def _fetch_year_start_price(fallback_df: pd.DataFrame, ticker: str) -> float:
    try:
        ytd_data = yf.download(ticker, start=f"{datetime.now().year}-01-01", progress=False)

        if isinstance(ytd_data.columns, pd.MultiIndex):
            ytd_data.columns = ytd_data.columns.get_level_values(0) #Flattens the DataFrame columns if yfinance returns multi-level headers (common in recent yfinance updates)

        return float(ytd_data['Close'].iloc[0])
    except (IndexError, KeyError, ValueError, Exception):
        return float(fallback_df['Close'].iloc[0])


def _fetch_asset_news(ticker: str) -> List[Dict[str, Any]]:
    try:
        return yf.Search(ticker, news_count=6).news
    except Exception:
        return []