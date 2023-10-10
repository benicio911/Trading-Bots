import gym
from gym import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
import shap
import threading
import MetaTrader5 as mt5
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding, Reshape, \
    GlobalAveragePooling1D, Flatten, Input, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime as dt
from datetime import datetime, timedelta
import time
import os
import pickle
from tqdm import tqdm
import ta
import ta.trend as trend
import ta.momentum as momentum
import traceback
import pandas_ta as pta
from sklearn.ensemble import RandomForestRegressor
import asyncio
from tensorflow.keras.layers import LSTM


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set up logging
logging.basicConfig(level=logging.INFO)

transformer_model = None  # global variable to store trained model
X_data = None  # global variable to store the training data


class ScaledDotProductAttention(Layer):
    """
    Computes scaled dot product attention given a query, key, and value.
    """

    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, q, k, v, mask=None):
        """
        Computes the scaled dot product attention.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Mask tensor to be added to the scaled dot product attention logits.

        Returns:
            The output tensor and the attention weights.
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = ScaledDotProductAttention()(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)])

    def call(self, x, training=True, mask=None):
        attn_output, attention_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerEncoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training=True, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class HybridTransformerLSTM(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, lstm_units, rate=0.1, **kwargs):
        super(HybridTransformerLSTM, self).__init__(**kwargs)

        self.d_model = tf.cast(d_model, tf.float32)
        self.num_layers = num_layers

        self.input_emb = Dense(d_model)
        self.pos_encoding = PositionalEncoding(input_dim, self.d_model)
        self.transformer_blocks = TransformerEncoder(num_layers, d_model, num_heads, dff, rate)
        self.dropout1 = Dropout(0.2)
        self.lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.dropout2 = Dropout(0.2)
        self.flatten = GlobalAveragePooling1D()
        self.final_layer = Dense(1, activation='sigmoid')

    def call(self, x):
        x *= tf.math.sqrt(self.d_model)
        x = self.input_emb(x)
        x = self.pos_encoding(x)
        x = self.transformer_blocks(x)
        x = self.dropout1(x)
        x = self.lstm(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.final_layer(x)

        return x

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01):
        self.n_actions = n_actions
        self.q_table = {}  # Initialize as an empty dictionary
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def hash_state(self, state):
        # Convert all elements of state to strings and concatenate
        state_str = "_".join(map(str, state))
        # Return the string representation (no need to hash for dictionary keys)
        return state_str
    
    def get_q_values(self, state):
        # If state is not in q_table, initialize it with random values
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(low=0, high=1, size=self.n_actions)
        return self.q_table[state]
    
    def choose_action(self, state):
        state_hashed = self.hash_state(state)
        q_values = self.get_q_values(state_hashed)
        
        if np.random.uniform(0, 1) < self.epsilon:
            return int(np.random.choice(self.n_actions))
        else:
            return int(np.argmax(q_values))
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def learn(self, state, action, reward, next_state, last_action):
        state_hashed = self.hash_state(state)
        next_state_hashed = self.hash_state(next_state)
        
        current_q_value = self.get_q_values(state_hashed)[action]
        next_max_q_value = np.max(self.get_q_values(next_state_hashed))
        
        # Q-learning update rule
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * next_max_q_value - current_q_value)
        self.q_table[state_hashed][action] = new_q_value
        
        
class TradingEnvironment(gym.Env):
    def __init__(self, model, data, symbol, RL_agent, start_idx=0, end_idx=None):
        super(TradingEnvironment, self).__init__()
        self.model = model
        self.data = data
        if self.data is not None:
            self.start_idx = 0
            self.end_idx = len(self.data)
            self.data = self.data[self.start_idx:self.end_idx]
        self.current_step = 0
        self.symbol = symbol
        self.window_size = window_size
        self.RL_agent = RL_agent
        self.position_size = 0.01  # The position size for each trade
        self.initial_balance = mt5.account_info().balance
        self.highest_balance = self.initial_balance  # To track the highest recorded balance
        self.simulated_balance = self.initial_balance  # To simulate balance changes
        self.open_position = None  # To track open positions
        self.open_position_price = 0.0
        self.trade_duration = 0  # in the __init__ method of TradingEnvironment
        
        # 3 actions: Buy, Sell, Hold
        self.action_space = spaces.Discrete(4)
        
        # Use the shape of the data for the observation space
        if self.data is not None:
            self.observation_space = spaces.Box(low=0, high=1, shape=self.data.shape[1:], dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(num_features,), dtype=np.float32)  # Assuming default shape is (num_features,)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step]

    def step(self, action):
        TIME_PENALTY = 0.01  # Define this at the top of your script or make it configurable
        # Use make_prediction_alpha to get the next prediction
        # At the beginning of the step function
        # Update the initial_balance for the next step
        reward = 0
        profit_scaling = 0.01  # arbitrary scaling value to keep profit impact manageable
        max_profit = 100  # maximum allowable profit for reward calculation
        profit = None
        predicted_action = make_prediction_alpha(self.symbol, window_size, self.model, self.RL_agent)
        
            # Retrieve symbol properties from MT5
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            # Handle error
            print("Failed to retrieve symbol info")
        
    
        # Actual market movement: calculate based on the difference between the next close price and the current close price
        df = pd.DataFrame({'close': self.data.iloc[:, 3]})
        if df['close'].isnull().any():
            df['close'].fillna(method='ffill', inplace=True)  # forward-fill missing values
        #print(df['close'])
        current_datetime = pd.to_datetime(self.data.iloc[self.current_step, 0])
            # Extracting time information
        current_hour = current_datetime.hour / 24.0  # Normalizing hour
        current_weekday = current_datetime.weekday() / 6.0  # Normalizing weekday
        # Compute TA Indicators
        df['MA'] = trend.SMAIndicator(close=df['close'], window=10).sma_indicator()
        df['RSI'] = momentum.RSIIndicator(close=df['close']).rsi()
        macd_indicator = trend.MACD(close=df['close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_SIGNAL'] = macd_indicator.macd_signal()
        if self.current_step + 1 < len(self.data):
            actual_movement = np.sign(self.data.iloc[self.current_step + 1, 3] - self.data.iloc[self.current_step, 3])
            # Calculate magnitude directly on the array 
            magnitude = abs(self.data.iloc[self.current_step + 1, 3] - self.data.iloc[self.current_step, 3])
        else:
            # Handle the edge case when you're at the last index
            actual_movement = 0  # or any other suitable value/logic
            magnitude = 0  # No magnitude change at the boundary

    # Handle Buy action
        if action == 0:
            # Assuming we are buying at the closing price of the current step
            self.open_position_price = self.data.iloc[self.current_step, 3]
            self.open_position = "Buy"

        
        # Handle Sell action
        elif action == 1:
            # Assuming we are selling at the closing price of the current step
            self.open_position_price = self.data.iloc[self.current_step, 3]
            self.open_position = "Sell"

        
        # Handle Hold action
        elif action == 2:  # Handle Hold action
            # No change to open_position_price or open_position_type
            if self.open_position_price is not None:  # Check if there is an open position
                current_price = self.data.iloc[self.current_step, 3]  # Assuming the closing price is at index 3
                
                # Check for profit or loss
                if self.open_position == "Buy":
                    if current_price > self.open_position_price:
                        reward = 0.01  # Reward for holding a profitable position
                    else:
                        reward = -0.05  # Penalty for holding a losing position
                elif self.open_position == "Sell":
                    if current_price < self.open_position_price:
                        reward = 0.01  # Reward for holding a profitable position
                    else:
                        reward = -0.05  # Penalty for holding a losing position
            else:
                reward = 0  # Neutral reward for holding with no open position
        
        # Handle Close action
        elif action == 3:
            if self.open_position_price is not None:
                close_price = self.data.iloc[self.current_step, 3]
                if self.open_position== "Buy":
                    profit = ((close_price - self.open_position_price) * self.position_size)* 2
                elif self.open_position == "Sell":
                    profit = ((self.open_position_price - close_price) * self.position_size) * 2
                
                if profit == None:
                    profit = 0  
                else:
                    profit = profit
                    
                self.simulated_balance += profit
                reward += profit 
                self.open_position_price = None
                self.open_position = None
                print(f"Closing trade. Profit: {profit}, New Balance: {self.simulated_balance}")
            else:
                reward = 0  # No position to close
        
        # Update highest recorded balance
        self.highest_balance = max(self.highest_balance, self.simulated_balance)
        
        # Calculate drawdown and check if it breaches the limit
        drawdown = (self.highest_balance - self.simulated_balance) / self.highest_balance

        # Magnitude-based adjustment
        MAX_MAGNITUDE = 5  # or some other value that makes sense for your use case
        magnitude = min(magnitude, MAX_MAGNITUDE)
        reward *= magnitude
        MIN_REWARD = -100  # define a minimum possible reward
        reward = max(reward, MIN_REWARD)
        # Update the trade duration
        if action != HOLD:  # Assuming HOLD is the action for not trading
            self.trade_duration = 0
        else:
            self.trade_duration += 1

        """        # Compute the time penalty
        TIME_PENALTY_BASE = 0.001
        time_penalty = TIME_PENALTY_BASE * (1 + self.trade_duration // 15)  # Increase penalty every 10 minutes

        # In the step function of TradingEnvironment class
        reward -= time_penalty"""
        # Incrementing the Current Step
        self.current_step += 1
        done = self.current_step >= len(self.data)

        # Return the appropriate state based on the current step. 
        # Integrate the TA indicators into the state representation.
        # Include time information in the next_state
        if not done:
            next_state = (self.data.iloc[self.current_step, 3], 
                        df['MA'].iloc[self.current_step], 
                        df['RSI'].iloc[self.current_step], 
                        df['MACD'].iloc[self.current_step], 
                        df['MACD_SIGNAL'].iloc[self.current_step],
                        current_hour,  # Adding time info to state
                        current_weekday)  # Adding weekday info to state
        else:
            last_index = len(df) - 1
            next_state = (self.data.iloc[-1, 3], 
                        df['MA'].iloc[last_index], 
                        df['RSI'].iloc[last_index], 
                        df['MACD'].iloc[last_index], 
                        df['MACD_SIGNAL'].iloc[last_index],
                        current_hour,  # Adding time info to state
                        current_weekday)  # Adding weekday info to state
        

        print(f"Step: {self.current_step}, Action: {action}, Reward: {reward}, Profit: {profit}, Drawdown: {drawdown}, Magnitude: {magnitude}")
        return next_state, reward, done, {}
    
class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


# Update the data preparation function to include the new columns
def preprocess_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


def get_financial_data(symbol, timeframe, start_time, end_time):
    if not mt5.initialize():
        return None

    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)

    if rates is None:
        logging.error('Error fetching financial data')
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


# Update the function to create datasets with the new features
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 3])  # 3 is the index of 'close' price in the array
    return np.array(dataX), np.array(dataY)

# Update the function to train the model
def train_transformer_model(data, window_size, num_features, num_heads, rate):
    # Create rolling window data
    lstm_units = 30
    X, y = create_dataset(data, window_size)

    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    # Define the model using the HybridTransformerLSTM
    hybrid_model = HybridTransformerLSTM(num_layers, num_features, num_heads, dff, window_size, lstm_units, rate)
    model = hybrid_model

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Rolling Window Validation
    window_length = int(len(X) * 0.2)  # For example, using 20% of the data for testing in each iteration
    start = 0
    histories = []

    while start + window_length < len(X):
        X_train, X_test = X[start:start+window_length], X[start+window_length:start+2*window_length]
        y_train, y_test = y[start:start+window_length], y[start+window_length:start+2*window_length]

        history = model.fit(X_train, y_train, epochs=125, validation_data=(X_test, y_test), callbacks=[early_stop])
        histories.append(history)

        start += window_length
    
    return model, histories


def compute_rsi(data, window):
    """Compute the RSI for a given data series and window."""
    diff = data.diff(1)
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sma(data, window):
    """Compute the Simple Moving Average for a given data series and window."""
    return data.rolling(window=window).mean()

def compute_ema(data, window):
    """Compute the Exponential Moving Average for a given data series and window."""
    return data.ewm(span=window, adjust=False).mean()

# Updating the data preparation function to include the technical indicators
def get_financial_data_with_indicators(symbol, timeframe, start_time, end_time):
    # Fetching data as previously done
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    df = pd.DataFrame(rates)
        # Debug: Validate the DataFrame structure before processing
    
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Add RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # Add SMA
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
    
    # Add EMA
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    
    # Drop NA values (as some indicators might introduce them)
    df.dropna(inplace=True)
    # Debug: Validate the DataFrame structure after processing
    
    return df

# Let's assume you're using these values
symbol = "NAS100.mini"
timeframe = mt5.TIMEFRAME_M15  # hourly data
start_time = datetime.now() - timedelta(days=365)  # 1 year ago
end_time = datetime.now()
window_size = 8
num_features = 6 # OHLC and Volume
num_heads = 3
rate = 0.1
num_layers = 2  # or any appropriate value
dff = 512  # or any appropriate value
NUM_EPISODES = 4
SAVE_FREQUENCY = 2
SAVED_Q_TABLE_PATH = "q_table.pkl"  # Define the path to the saved Q-table
ACTION_CHANGE_PENALTY = 0.01
last_action = None
open_position_type = None
agent = None


def get_latest_data(symbol, window_size, timeframe=mt5.TIMEFRAME_M15):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=window_size)
    df = get_financial_data_with_indicators(symbol, timeframe, start_time, end_time)
    return df

def split_data(data, train_ratio=0.8):
    """
    Splits the data into training and testing sets based on the provided ratio.
    
    Args:
    - data (pd.DataFrame or np.array): The data to be split.
    - train_ratio (float): The proportion of data to be used for training. Must be between 0 and 1.

    Returns:
    - train_data, test_data: Split data.
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data

# Constants for RL actions
BUY = 0
SELL = 1
HOLD = 2
CLOSE = 3

def convert_prediction_to_state(y_pred, feature_vector):
    """
    Convert transformer's prediction and other market features into a state representation for the RL agent.
    """
    state = np.append(y_pred, feature_vector)
    return state

# Updated make_prediction_alpha function
def make_prediction_alpha(symbol, window_size, model, RL_agent):
    df = get_latest_data(symbol, window_size)
    y_data = df  # retrieve the original unscaled data
    X_data, _ = preprocess_data(df[['high', 'low', 'close', 'sma_50', 'ema_50', 'ema_200']])
    X_reshaped = X_data[-window_size:].reshape((1, window_size, num_features))

    # Make prediction
    y_pred = model.predict(X_reshaped)[0][0]
    print("y_pred: ", y_pred)
    
    # Initialize and fit the scaler
    y_scaler = StandardScaler()
    y_scaler.fit(y_data.values.reshape(-1, 1))
    # Invert scaling on prediction
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1))

    print(y_pred_orig)
    
    # Convert to state
    state = tuple(y_pred_orig)

    print("State: ", state)
    # Get action from RL agent
    action = RL_agent.choose_action(state)
    print("Action from Agent: ",action)
    
    # Combine transformer prediction with RL agent's action
    #y_pred < 0.40 and y_pred > 0.60 and 
    if action == SELL:
        alpha = 1
    elif action == BUY:
        alpha = 0
    elif action == HOLD:
        alpha = -1
    else:
        alpha = -2
        
    return alpha


def make_prediction_xi(symbol, window_size, model):
    df = get_latest_data(symbol, window_size)
    X_data, _ = preprocess_data(df[['high', 'low', 'close', 'sma_50', 'ema_50', 'ema_200']])
    X_reshaped = X_data[-window_size:].reshape((1, window_size, num_features))

    # Make prediction
    y_pred = model.predict(X_reshaped)[0][0]

    # Make Buy or Sell decision based on the prediction
    if y_pred > 0.5:
        return "Buy"
    else:
        return "Sell"

    return xi

def train_RL_agent(model, data):
    global agent
    env = TradingEnvironment(model, data, symbol, agent)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        last_action = None  # Initialize last_action

        # wrap the inner loop with tqdm for a progress bar
        for _ in tqdm(range(len(data)), desc=f"Episode {episode+1}/{NUM_EPISODES}"):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, last_action)
            state = next_state
            last_action = action  # Update last_action
            agent.update_epsilon()  # Decay epsilon

            if done:
                break  # exit the loop if the episode is done

            # Save the Q-table periodically after each episode
            if episode % SAVE_FREQUENCY == 0:
                with open(SAVED_Q_TABLE_PATH, 'wb') as f:
                    pickle.dump(agent.q_table, f)
                    logging.info(f"Q-table saved after episode {episode}.")
              
def backtest(model, agent, data):
    agent.epsilon = agent.epsilon_min
    env = TradingEnvironment(model, data, symbol, agent)
    state = env.reset()
    done = False
    
    # Lists to store values for plotting
    pl_list = [env.simulated_balance]
    trades = []
    closing_prices = [data.iloc[0, 3]]  # Starting with the first closing price
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Append values for plotting
        # Append values for plotting
        pl_list.append(env.simulated_balance)
        if env.current_step < len(data):
            trades.append((env.current_step, data.iloc[env.current_step, 3], action))  # Use closing price instead of simulated_balance
        else:
            print(f"Warning: Attempt to access index {env.current_step}, which is out of bounds. Skipping.")
        
        state = next_state
    
    # Plotting
    plt.figure(figsize=(12,6))
    
    # Plot P/L
    plt.plot(pl_list, label='P/L', alpha=0.5)
    
    # Plot closing prices
    plt.plot(closing_prices, label='Close Price', alpha=0.5)
    
    # Plot trades
    for trade in trades:
        step, balance, action = trade
        if action not in [0, 1, 2, 3]:  # include 2 and 3 for hold and close
            print(f"Skipping trade at step {step} due to unrecognized action {action}")
            continue  # skip this iteration

        # Assign colors and markers based on action
        if action == 0:  # Buy
            color = 'g'
            marker = '^'
        elif action == 1:  # Sell
            color = 'r'
            marker = 'v'
        elif action == 2:  # Hold (optional: you can skip plotting hold actions if desired)
            color = 'b'
            marker = 's'
        elif action == 3:  # Close
            color = 'y'
            marker = 'o'

        plt.scatter(step, balance, color=color, marker=marker, alpha=1)
    
    plt.title('Backtest Results')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig('plot.png')

def feature_importance(data, target_column):
    X = data.drop(columns=[target_column])  # Excluding the target variable
    y = data[target_column]  # Only the target variable
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()  # Initialize model
    model.fit(X_train, y_train)  # Fit to the training data
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order and get the indices
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(10,6))
    
    # Create plot title
    plt.title("Feature Importance")
    
    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)
    
    # Show plot
    plt.show()
    
def thread_Alpha(kozeroshutdown, trading_data: dict, lock, alpha_done):
    with tf.device('/GPU:0'):
        global agent
        global alpha
        global xi
        global symbol
        placeholder_env = TradingEnvironment(None, None, None, None)
        
        # Create an RL agent
        agent = QLearningAgent(placeholder_env.action_space.n, placeholder_env.observation_space.shape[0])
        
                # Check if saved Q-table exists and load it
        q_table_exists = False
        if os.path.exists(SAVED_Q_TABLE_PATH):
            with open(SAVED_Q_TABLE_PATH, 'rb') as f:
                try:
                    loaded_q_table = pickle.load(f)
                    agent.q_table = loaded_q_table
                    logging.info("Successfully loaded Q-table.")
                    q_table_exists = True
                except EOFError:
                    logging.error("Error loading Q-table. The file may be corrupted.")
        
        alpha_done.clear()  # Reset alpha_done event
        symbol = trading_data['epsilon']
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        df = get_financial_data_with_indicators(symbol, timeframe, start_time, end_time)
        
        if df is not None:
            # Preprocess and Train Model
            df_scaled, _ = preprocess_data(df[['high', 'low', 'close', 'sma_50', 'ema_50', 'ema_200']].values)
            transformer_model, history = train_transformer_model(df_scaled, window_size, num_features, num_heads, rate)
            
            # Define Data Split Indices
            TRAIN_START = 0
            TRAIN_END = int(0.8 * len(df))
            TEST_START = TRAIN_END
            TEST_END = len(df)
            
            # Split Data
            df_train = df[TRAIN_START:TRAIN_END]
            df_test = df[TEST_START:TEST_END]
            
            # Debugging Information
            print(f"TRAIN: Start={TRAIN_START}, End={TRAIN_END}, Shape={df_train.shape}")
            print(f"TEST: Start={TEST_START}, End={TEST_END}, Shape={df_test.shape}")
            
            # Further data quality checks (examples)
            print(df_test.head())
            print(df_test.isna().sum())
            
                # Training Phase
            # Train the RL agent
            if not q_table_exists: 
                print("Beginning Training for RL Agent...") 
                try: 
                    print("wait...")
                    train_RL_agent(transformer_model, df_train)
                except Exception as e: 
                    logging.error(f"Error during training: {e}") 
                print("RL Agent Training Completed...")

                # Feature Importance - Might be moved elsewhere depending on exact usage
                try:
                    df_train = df_train.copy()
                    df_train['target'] = df_train['close'].shift(-1)
                    feature_importance(df_train.dropna(), 'target')  # Ensure no NaNs
                except Exception as e:
                    logging.error(f"Error during feature importance calculation: {e}") 
            
            print("Beginning Backtesting...")
            try:
                backtest(transformer_model, agent, df_test)  # Ensure df_test is unseen data
            except Exception as e:
                logging.error("Error during backtesting: {}".format(str(e)))
                logging.error("Exception type: {}".format(str(type(e))))
                logging.error("Traceback: {}".format(traceback.format_exc())) 
            print("Backtesting Completed...")
            counter = 0 
            while not kozeroshutdown.is_set():
                try:
                    """              if counter % 20 == 0:
                    print("Beginning Retraining for RL Agent...")
                    try:
                        train_RL_agent(transformer_model, df_test)
                    except Exception as e:
                        logging.error(f"Error during training: {e}")
                    print("RL Agent Training Completed...")
                    counter = 0"""
                    current_time = dt.datetime.now().time()
                    if current_time.minute % 15 == 0:  
                        agent.epsilon = agent.epsilon_min
                                    # Backtest the agent
                        alpha = make_prediction_alpha(symbol, window_size, transformer_model, agent)
                        xi = make_prediction_xi(symbol, window_size, transformer_model)
                        
                        # Update trading data
                        lock.acquire()
                        trading_data['alpha'] = alpha
                        trading_data['xi'] = xi
                        lock.release()
                        
                        alpha_done.set()  # Signal that alpha is done
                        """                        if counter % SAVE_FREQUENCY == 0:
                            with open(SAVED_Q_TABLE_PATH, 'wb') as f:
                                pickle.dump(agent.q_table, f)
                                logging.info("Q.TABLE Saved...")"""
                        
                        counter += 1
                        time.sleep(60)
                except Exception as e:
                    logging.error(f"Error during live prediction: {e}")
                    
                                # Save Q-table periodically
                