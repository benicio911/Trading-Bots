import threading
import queue
import json
from data_retrieval import get_jwt_token, get_historical_data
from data_processing import preprocess_data
from model_training import train_lstm_model
from prediction_evaluation import make_predictions, evaluate_model
from logging_setup import setup_logging
import time
from PPO import TradingEnv, train_ppo

# Load configuration and set up logging
with open('config.json') as config_file:
    config = json.load(config_file)

logger = setup_logging()

# Shared resources and synchronization primitives
data_queue = queue.Queue()
processed_data_event = threading.Event()
model_queue = queue.Queue()
prediction_queue = queue.Queue()

def data_retrieval_thread():
    try:
        access_token = get_jwt_token(config["api_credentials"]["email"],
                                     config["api_credentials"]["password"],
                                     config["api_credentials"]["server"])

        tradable_instrument_id = 309  
        route_id = 452
        start_date = "2022-01-01"  # Define the start and end dates for historical data
        end_date = "2023-01-01"
        resolution = "30m"  # Assuming 30 minutes resolution
        acc_num = "1"
        while True:
            historical_data = get_historical_data(access_token, acc_num, tradable_instrument_id, route_id, start_date, end_date, resolution)
            data_queue.put(historical_data)
            logger.info("Data retrieved and added to the queue")
            time.sleep(30*60)  # Interval between data retrievals
    except Exception as e:
        logger.error(f"Data Retrieval Error: {e}")

def data_processing_thread():
    while True:
        try:
            # Retrieve data from the queue
            historical_data = data_queue.get()
            if not historical_data:
                continue
            
            # Extracting the relevant data
            bar_data = historical_data.get('d', {}).get('barDetails', [])
            if not bar_data:
                logger.error("No bar data found in historical data.")
                continue

            # Log a sample of bar_data for debugging
            logger.info(f"Sample bar data: {bar_data[:5]}")  # Log first 5 bars

            # Call preprocess_data
            processed_data = preprocess_data(bar_data)
            if processed_data is not None:
                # Further processing or model training
                pass
            else:
                logger.error("Processed data is None.")
        except Exception as e:
            logger.error(f"Error in data processing thread: {e}")

def model_training_thread():
    try:
        while True:
            processed_data_event.wait()
            df = model_queue.get()
            processed_data_event.clear()
            model = train_lstm_model(df)
            prediction_queue.put(model)
            logger.info("Model trained and available for predictions")
    except Exception as e:
        logger.error(f"Model Training Error: {e}")

def ppo_training_thread():
    try:
        while True:
            processed_data_event.wait()
            market_data = model_queue.get()  # Assuming this queue now contains market data for PPO
            processed_data_event.clear()

            # Initialize Trading Environment
            initial_balance = 10000  # Example initial balance
            env = TradingEnv(initial_balance, market_data)

            # Train PPO Model
            train_ppo(env, total_episodes=100)

            logger.info("PPO model trained and ready for action decisions")
    except Exception as e:
        logger.error(f"PPO Training Error: {e}")
        
def prediction_thread():
    try:
        while True:
            if not prediction_queue.empty():
                model = prediction_queue.get()
                # Logic to determine X_test goes here
                # predictions = make_predictions(model, X_test, ...)
                logger.info("Predictions made using the model")
    except Exception as e:
        logger.error(f"Prediction Error: {e}")

def main():
    threads = [
        threading.Thread(target=data_retrieval_thread),
        threading.Thread(target=data_processing_thread),
        threading.Thread(target=model_training_thread),
        threading.Thread(target=ppo_training_thread),
        threading.Thread(target=prediction_thread)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()