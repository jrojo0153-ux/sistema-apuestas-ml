import requests
import os


API_KEY = os.getenv("API_KEY_ODDS")  # tu secret
URL = "https://api.the-odds-api.com/v4/sports/soccer/odds"


def obtener_partidos_hoy():
    try:
        params = {
            "apiKey": API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }

        response = requests.get(URL, params=params)

        # 🔧 CORRECCIÓN DE INDENTACIÓN
        if response.status_code != 200:
            print("❌ Error API:", response.status_code)
            return []

        data = response.json()

        partidos = []

        for match in data:
            home = match.get("home_team")
            away = match.get("away_team")

            partidos.append({
                "home": home,
                "away": away
            })

        return partidos

    except Exception as e:
        print("❌ Error obteniendo partidos:", e)
        return []
