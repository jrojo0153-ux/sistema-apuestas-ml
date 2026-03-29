import os
import requests

API_KEY = os.getenv("API_FOOTBALL_KEY")

def get_matches():
    url = "https://v3.football.api-sports.io/fixtures?next=20"

    headers = {
        "x-apisports-key": API_KEY
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("❌ Error API FOOTBALL:", response.status_code)
            return []

        data = response.json()
        return data.get("response", [])

    except Exception as e:
        print("Error:", e)
        return []
