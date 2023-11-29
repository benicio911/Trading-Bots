import requests
import pandas as pd
import time
global access_token
from datetime import datetime

def get_jwt_token(email, password, server):
    url = "https://live.tradelocker.com/backend-api/auth/jwt/token"
    body = {
        "email": email,
        "password": password,
        "server": server
    }
    response = requests.post(url, json=body)
    if response.status_code == 201:
        return response.json()['accessToken']
    else:
        raise Exception(f"Failed to get JWT token: {response.text}")

# Example usage
email = "beniciomorales2@gmail.com"
password = "qCj#8jQ%"
server = "OSP-LIVE"

try:
    access_token = get_jwt_token(email, password, server)
    print("Access Token:", access_token)
except Exception as e:
    print(e)


def get_instruments_with_route_ids(access_token, acc_num, account_id):
    base_url = "https://live.tradelocker.com/backend-api"
    endpoint = f"/trade/accounts/{account_id}/instruments"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "accNum": acc_num
    }
    
    response = requests.get(base_url + endpoint, headers=headers)
    print("Response Data:", response.json())  # Add this line for debugging
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve instruments with route IDs: {response.text}")
    

def find_spx500_trade_route_id(instruments_info):
    for instrument in instruments_info['d']['instruments']:
        if instrument['name'] == 'SPX500':
            for route in instrument['routes']:
                if route['type'] == 'TRADE':
                    return route['id']
    raise Exception("SPX500 TRADE routeId not found")

# Test request with hardcoded values
def test_historical_data_request():
    # Use known good values for testing
    test_tradable_instrument_id = 307  # Example ID, replace with the correct one
    test_route_id = 452  # Example route ID, replace with the correct one
    response = get_historical_data(access_token, acc_num, test_tradable_instrument_id, test_route_id, start_date, end_date, resolution)
    print("Test Response:", response)
    
# Your existing function for getting historical data
def get_historical_data(access_token, acc_num, tradable_instrument_id, route_id, start_date, end_date, resolution):
    base_url = "https://live.tradelocker.com/backend-api"
    endpoint = "/trade/history"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "accNum": acc_num
    }
    params = {
        "tradableInstrumentId": tradable_instrument_id,
        "routeId": route_id,
        "from": get_unix_timestamp(start_date),
        "to": get_unix_timestamp(end_date),
        "resolution": resolution
    }
    response = requests.get(base_url + endpoint, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve historical data: {response.text}")

# Function to convert a date string to a Unix timestamp in milliseconds
def get_unix_timestamp(date_str):
    """ Convert a date string to a Unix timestamp in milliseconds. Handles both date only and datetime strings. """
    try:
        # Try parsing as datetime
        return int(time.mktime(time.strptime(date_str, "%Y-%m-%d %H:%M:%S")) * 1000)
    except ValueError:
        # Fallback to date only
        return int(time.mktime(time.strptime(date_str, "%Y-%m-%d")) * 1000)
    
# Placeholder values for access token and account details
# These should be replaced with actual values for real use
access_token = access_token
acc_num = "1"
tradable_instrument_id = 307  # SPX500 tradableInstrumentId
route_id = 452  # INFO routeId for SPX500
trade_route_id = 900
start_date = "2022-01-01"  # Example start date
 # Use the current datetime as the end date
end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
resolution = "4H"  # 1-hour resolution
 
 
def process_historical_data(historical_data):
    # Create DataFrame from historical data
    # Print the historical data for debugging
    print("Historical Data:", historical_data)
    
    df = pd.DataFrame(historical_data['d']['barDetails'])

    # Rename columns to more descriptive names
    df.rename(columns={'t': 'time', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    return df   
    
def place_order(access_token, acc_num, instrument_id, side, vol, stop_loss, route_id, order_type='market', validity='DAY', request_id=151515):
    account_id = "99207"
    url = "https://live.tradelocker.com/backend-api/trade/accounts/{99207}/orders"  # Replace {account_id} with actual account ID
    headers = {
        'Authorization': f"Bearer {access_token}",
        'accNum': acc_num,
        'Content-Type': 'application/json'
    }
    payload = {
        "qty": vol,
        "routeId": route_id,
        "side": side,
        "stopLoss": stop_loss,
        "stopLossType": "absolute",
        "tradableInstrumentId": instrument_id,
        "type": order_type,
        "validity": validity,
        "requestId": request_id  # Ensure this is unique for each request
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return response.json()
    else:
        print(f"Failed to place order. Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(f"Failed to place order: {response.text}")

def adjust_stop_loss(access_token, acc_num, order_id, new_stop_loss):
    # API endpoint to modify an order
    base_url = "https://live.tradelocker.com/backend-api"
    endpoint = f"/trade/orders/{order_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "accNum": acc_num
    }
    body = {
        "stopLoss": new_stop_loss  # New stop-loss value
    }

    response = requests.patch(base_url + endpoint, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to adjust stop loss: {response.text}")


def calculate_emas(df):
    print("Calculating EMAs...")
    # Calculate EMAs
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
        # Print the last row of EMAs for verification
            # Print the last row of close price and EMAs for verification
    print("Latest Close:", df['close'].iloc[-1])
    print("Current EMA_10:", df['EMA_10'].iloc[-1])
    print("Current EMA_50:", df['EMA_50'].iloc[-1])
    return df

def detect_signals(df, route_id):
    vol = 0.01  # Define volume

    # Check if DataFrame has at least two rows
    if len(df) < 2:
        print("Not enough data for signal detection.")
        return

    # Get the last two rows for the current and previous data points
    current_row = df.iloc[-1]
    previous_row = df.iloc[-2]

    # Round stop loss to the nearest tenth
    stop_loss = round(current_row['EMA_50'], 1)

    # Buy signal criteria
    if current_row['EMA_10'] > current_row['EMA_50'] and previous_row['EMA_10'] <= previous_row['EMA_50']:
        if current_row['close'] > current_row['EMA_10'] and current_row['close'] > current_row['EMA_50']:
            print("Buy Signal Detected")
            place_order(access_token, acc_num, tradable_instrument_id, "buy", vol, stop_loss, route_id)

    # Sell signal criteria
    elif current_row['EMA_10'] < current_row['EMA_50'] and previous_row['EMA_10'] >= previous_row['EMA_50']:
        if current_row['close'] < current_row['EMA_10'] and current_row['close'] < current_row['EMA_50']:
            print("Sell Signal Detected")
            place_order(access_token, acc_num, tradable_instrument_id, "sell", vol, stop_loss, route_id)
            
def main():
    while True:
        account_id = "99207"
        print("Checking for signals...")
        access_token = get_jwt_token(email, password, server)
        instruments_info = get_instruments_with_route_ids(access_token, acc_num, account_id)
        # Fetch historical data and calculate EMAs
        historical_data = get_historical_data(access_token, acc_num, tradable_instrument_id, 452, start_date, end_date, resolution)
        
        df = process_historical_data(historical_data)
        df = calculate_emas(df)
        detect_signals(df, trade_route_id)  # Pass route_id here
        print("Check Done. Waiting for next candle...")
        # Wait for the next candle
        
        time.sleep(3600)  # Sleep for one hour

if __name__ == "__main__":
    main()