import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import json
from logging_setup import logger, logging_lock

def create_dataset(data, time_step=30):
    """ Creates a dataset for training the LSTM model. """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # Include all features
        y.append(data[i + time_step, 0])      # Predicting the next closing price
    return np.array(X), np.array(y)

def train_lstm_model(df):
    try:
        with open('config.json') as config_file:
            config = json.load(config_file)

        model_file_path = config["model_file_path"]
        features = config["data_features"]
        target = config["target_feature"]
        time_step = config["time_step"]

        with logging_lock:
            logger.info("Starting LSTM model training.")
        
        # Scaling data
        input_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_input_features = input_feature_scaler.fit_transform(df[features].values)

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_target = target_scaler.fit_transform(df[[target]].values)

        # Create dataset
        X, y = create_dataset(scaled_input_features, time_step)
        X = X.reshape(X.shape[0], X.shape[1], len(features))  # Reshape for the number of features

        # Splitting data into train and test sets
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

        # Check if the model exists
        if os.path.exists(model_file_path):
            print("Loading the existing model...")
            model = load_model(model_file_path)
        else:
            print("No existing model found. Creating and training a new model...")
            # Model definition
            model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, len(features))),
                tf.keras.layers.LSTM(100, return_sequences=False),
                tf.keras.layers.Dense(50),
                tf.keras.layers.Dense(1)
            ])

            # Compile and train
            optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=200, callbacks=[early_stopping])

            # Save the model
            model.save(model_file_path)
        with logging_lock:
            logger.info("LSTM model training completed.")
    except Exception as e:
        with logging_lock:
            logger.error(f"Error in train_lstm_model: {e}")
        raise

    return model, target_scaler, input_feature_scaler