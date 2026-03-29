import requests
import os

API_KEY = os.getenv("API_KEY_ODDS")

def obtener_cuotas(partidos):
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={API_KEY}&regions=eu&markets=h2h"

    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Error cuotas")
        return []

    data = response.json()

    cuotas = []

    for match in data:
        cuotas.append({
            "home": match["home_team"],
            "away": match["away_team"],
            "cuota_home": match["bookmakers"][0]["markets"][0]["outcomes"][0]["price"],
            "cuota_away": match["bookmakers"][0]["markets"][0]["outcomes"][1]["price"]
        })

    return cuotas
