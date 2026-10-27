"""
TEMA Strategy Engine.
Implements the Triple Exponential Moving Average (TEMA) for lag-reduced forecasting.
"""

import pandas as pd


def generate_tema_forecast(
        market_dataframe: pd.DataFrame,
        forecast_steps: int = 30
) -> pd.DataFrame:
    # Generates a forecast based on the Triple Exponential Moving Average trend.
    # TEMA = (3 * EMA1) - (3 * EMA2) + EMA3
    if market_dataframe.empty:
        return pd.DataFrame()

    close_prices = market_dataframe['Close']

    # 1. Triple EMA Calculation
    # TEMA reduces "lag" by subtracting the multiple EMAs from each other.
    # Formula: TEMA = (3 * EMA1) - (3 * EMA2) + EMA3
    ema1 = close_prices.ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    ema3 = ema2.ewm(span=20, adjust=False).mean()

    tema = (3 * ema1) - (3 * ema2) + ema3

    # 2. Trend Velocity
    # We measure how fast the TEMA is rising or falling over the last 5 periods.
    latest_tema = tema.iloc[-1]
    previous_tema = tema.iloc[-5]
    tema_velocity = (latest_tema - previous_tema) / 5

    # 3. Project Future Path
    # We project the current price forward based on the TEMA's current momentum.
    latest_price = float(close_prices.iloc[-1])
    forecast_path = [latest_price + (tema_velocity * step) for step in range(1, forecast_steps + 1)]

    # 4. Generate Timestamps
    future_dates = _calculate_future_indices(market_dataframe, forecast_steps)

    # 5. Assemble Result
    # We combine the last known price with the new predicted path.
    return pd.concat([
        pd.DataFrame({'Predicted': [latest_price]}, index=[market_dataframe.index[-1]]),
        pd.DataFrame({'Predicted': forecast_path}, index=future_dates)
    ])


def _calculate_future_indices(df: pd.DataFrame, steps: int):
    if len(df) < 2:
        return pd.date_range(start=df.index[-1], periods=steps + 1, freq='D')[1:]

    # Detect the time difference (delta) between the last two candles
    delta = df.index[-1] - df.index[-2]
    return [df.index[-1] + (i * delta) for i in range(1, steps + 1)]