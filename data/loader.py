# data/loader.py

import requests
import os


def obtener_cuotas():
    """
    Obtiene cuotas desde Odds API
    """

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
                if "bookmakers" not in match or len(match["bookmakers"]) == 0:
                    continue

                bookmaker = match["bookmakers"][0]
                market = bookmaker["markets"][0]
                outcomes = market["outcomes"]

                if len(outcomes) < 2:
                    continue

                cuotas = [o["price"] for o in outcomes]

                partidos.append({
                    "home": match["home_team"],
                    "away": match["away_team"],
                    "cuotas": cuotas
                })

            except Exception:
                continue

        return partidos

    except Exception as e:
        print(f"❌ Error obteniendo cuotas: {e}")
        return []
