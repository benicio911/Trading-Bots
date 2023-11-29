import requests
import time
import requests
import json
import logging
from logging_setup import logger, logging_lock

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
    
def get_all_accounts(access_token):
    base_url = "https://live.tradelocker.com/backend-api"
    endpoint = "/auth/jwt/all-accounts"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(base_url + endpoint, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve account numbers: {response.text}")
print("Access Token: ", access_token)
# Example usage
access_token = access_token

try:
    accounts_info = get_all_accounts(access_token)
    print("Accounts Information:", accounts_info)
except Exception as e:
    print(e)
     
def get_instruments_with_route_ids(access_token, acc_num, account_id):
    base_url = "https://live.tradelocker.com/backend-api"
    endpoint = f"/trade/accounts/{account_id}/instruments"  # Using accountId here
    headers = {
        "Authorization": f"Bearer {access_token}",
        "accNum": acc_num  # accNum is still used in the header
    }
    
    response = requests.get(base_url + endpoint, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve instruments with route IDs: {response.text}")

# Example usage
access_token = access_token
acc_num = "1"  # Replace with your actual accNum
account_id = "4219"  # Replace with your actual accountId

try:
    instruments_info = get_instruments_with_route_ids(access_token, acc_num, account_id)
    print("Instruments with Route IDs:", instruments_info)
except Exception as e:
    print(e)
    
def get_unix_timestamp(date_str):
    """ Convert a date string to a Unix timestamp in milliseconds. """
    return int(time.mktime(time.strptime(date_str, "%Y-%m-%d")) * 1000)

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
    print(f"Request URL: {response.request.url}")  # Log the final request URL
    print(f"Response Status Code: {response.status_code}")  # Log status code
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve historical data: {response.text}")


# Example usage
access_token = access_token
acc_num = "1"
tradable_instrument_id = 309  # Replace with the actual tradableInstrumentId for US30.MINI
route_id = 452  # Replace with the actual 'INFO' routeId for US30.MINI
start_date = "2022-01-01"
end_date = "2023-01-01"
resolution = "30m"  # Daily resolution

#try:
    #historical_data = get_historical_data(access_token, acc_num, tradable_instrument_id, route_id, start_date, end_date, resolution)
    #print("Historical Data:", historical_data)
#except Exception as e:
    #print(e)

# Example historical data (replace this with the actual call to get_historical_data)
#historical_data = get_historical_data(access_token, acc_num, tradable_instrument_id, route_id, start_date, end_date, resolution) 