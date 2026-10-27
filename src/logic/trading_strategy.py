import pandas as pd
import config


def evaluate_market_signal(df: pd.DataFrame) -> tuple[str, str]:

    if len(df) < 10:
        return "WAITING FOR DATA", "#808495"

    settings = config.IndicatorSettings

    #1. Define the 10-bar window
    window = df.tail(10)
    latest = window.iloc[-1]

    #2. Extract current values
    rsi_now = latest.get('RSI')
    macd_now = latest.get('MACD_Hist')
    close_now = latest.get('Close')
    ma20_now = latest.get('MA20')

    if rsi_now is None or macd_now is None:
        return "DATA INCOMPLETE", "#808495"

    #3. Sequence Logic: Check if conditions were met ANYWHERE in the last 10 bars
    was_recently_oversold = (window['RSI'] < settings.RSI_OVERSOLD_LEVEL).any()
    was_recently_overbought = (window['RSI'] > settings.RSI_OVERBOUGHT_LEVEL).any()

    #4. Current momentum must confirm the recent extreme
    is_momentum_confirmed_up = macd_now > 0
    is_momentum_confirmed_down = macd_now < 0
    is_above_trend = close_now > ma20_now

    if was_recently_oversold and is_momentum_confirmed_up:
        return "STRONG BUY", "#00FF41"

    if was_recently_overbought and is_momentum_confirmed_down:
        return "STRONG SELL", "#FF3131"

    if not is_above_trend and rsi_now < 50:
        return "CAUTION: WEAK TREND", "#FFD700"

    return "NEUTRAL / HOLD", "#808495"