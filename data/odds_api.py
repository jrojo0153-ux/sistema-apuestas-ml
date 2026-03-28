import requests
import os

API_KEY = os.getenv("API_KEY_ODDS")

def obtener_cuotas():
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds"

    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Error Odds API: {response.status_code}")
        return []

    data = response.json()

    cuotas = []

    for match in data:
        try:
            home = match["home_team"]
            away = match["away_team"]

            bookmakers = match.get("bookmakers", [])
            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])
            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])

            odds_dict = {o["name"]: o["price"] for o in outcomes}

            cuotas.append({
                "home": home,
                "away": away,
                "odds_home": odds_dict.get(home),
                "odds_draw": odds_dict.get("Draw"),
                "odds_away": odds_dict.get(away)
            })

        except Exception as e:
            continue

    return cuotas
