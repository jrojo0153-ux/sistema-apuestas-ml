import requests
import os

def obtener_cuotas():
    API_KEY = os.getenv("API_KEY_ODDS")

    if not API_KEY:
        print("❌ Falta API_KEY_ODDS")
        return []

    url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"

    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h"
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"❌ Error API Odds: {response.status_code}")
            return []

        data = response.json()

        partidos = []

        for match in data:
            try:
                bookmakers = match.get("bookmakers", [])
                if not bookmakers:
                    continue

                markets = bookmakers[0].get("markets", [])
                if not markets:
                    continue

                outcomes = markets[0].get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                cuotas = [o["price"] for o in outcomes]

                partidos.append({
                    "home": match.get("home_team"),
                    "away": match.get("away_team"),
                    "cuotas": cuotas
                })

            except:
                continue

        return partidos

    except Exception as e:
        print("❌ Error:", e)
        return []
