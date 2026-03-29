import requests
import os

def obtener_partidos():
    api_key = os.getenv("API_KEY_ODDS")

    if not api_key:
        print("❌ API KEY no encontrada")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?apiKey={api_key}&regions=eu"

    try:
        response = requests.get(url)
        data = response.json()

        partidos = []

        for match in data:
            partidos.append({
                "home": match.get("home_team", "NA"),
                "away": match.get("away_team", "NA"),
                "cuota": 2.0  # placeholder seguro
            })

        return partidos

    except Exception as e:
        print(f"❌ Error API: {e}")
        return []
