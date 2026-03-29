import os
import requests

API_KEY = os.getenv("API_KEY_ODDS")

def get_odds():
    url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"

    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print("❌ Error ODDS:", response.status_code)
            return []

        return response.json()

    except Exception as e:
        print("Error odds:", e)
        return []
