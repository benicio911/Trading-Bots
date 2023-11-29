import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from logging_setup import logger, logging_lock
import json

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger_bands(data, window=20, no_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * no_of_std)
    lower_band = rolling_mean - (rolling_std * no_of_std)
    return upper_band, rolling_mean, lower_band

def preprocess_data(bar_data):
    try:
        with logging_lock:
            logger.info("Starting data preprocessing.")
        """Preprocesses the raw data and calculates technical indicators."""
        df = pd.DataFrame(bar_data)

        # Convert Unix timestamps to readable dates and set as index
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('date', inplace=True)
        df.drop('t', axis=1, inplace=True)
        
        # Adding technical indicators
        df['sma_50'] = df['c'].rolling(window=50).mean()
        df['rsi'] = calculate_rsi(df['c'])
        df['macd'], df['macdsignal'], df['macdhist'] = calculate_macd(df['c'])
        df['upper_band'], df['middle_band'], df['lower_band'] = calculate_bollinger_bands(df['c'])

        # Normalization using Min-Max Scaling
        df['norm_volume'] = (df['v'] - df['v'].min()) / (df['v'].max() - df['v'].min())

        df.bfill(inplace=True)
        
        with logging_lock:
            logger.info("Data preprocessing completed.")
            return df
    except Exception as e:
        with logging_lock:
            logger.error(f"Error in preprocess_data: {e}")
        raise
        