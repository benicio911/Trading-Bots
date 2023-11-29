import requests
import json
from datetime import datetime

# Load API credentials from JSON file
with open('config.json', 'r') as file:
    config = json.load(file)

API_BASE_URL = "https://demo.tradelocker.com/backend-api"

def get_jwt_token():
    """Authenticate and get JWT token."""
    url = f"{API_BASE_URL}/auth/jwt/token"
    body = {
        "email": config["api_credentials"]["email"],
        "password": config["api_credentials"]["password"],
        "server": config["api_credentials"]["server"]
    }
    response = requests.post(url, json=body)
    if response.status_code == 201:
        return response.json()['accessToken']
    raise Exception(f"Failed to get JWT token: {response.text}")

def get_market_data(endpoint, params):
    """Generic function to fetch market data."""
    url = f"{API_BASE_URL}/{endpoint}"
    headers = {"Authorization": f"Bearer {get_jwt_token()}"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    raise Exception(f"Failed to retrieve data: {response.text}")

# Example usage: get_market_data('trade/dailyBar', {'tradableInstrumentId': 309, 'routeId': 452})

def place_order(order_details):
    """Place an order."""
    url = f"{API_BASE_URL}/trade/accounts/{account_id}/orders"
    headers = {"Authorization": f"Bearer {get_jwt_token()}"}
    response = requests.post(url, headers=headers, json=order_details)
    if response.status_code in [200, 201]:
        return response.json()
    raise Exception(f"Failed to place order: {response.text}")