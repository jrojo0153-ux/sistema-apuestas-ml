import requests
import os

API_KEY = os.getenv("API_KEY_ODDS")

def obtener_partidos_hoy():
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={API_KEY}&regions=eu&markets=h2h"

    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Error obteniendo partidos")
        return []

    data = response.json()

    partidos = []

    for match in data:
        partidos.append({
            "home": match["home_team"],
            "away": match["away_team"]
        })

    return partidos
