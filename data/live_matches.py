import requests
from config import API_FOOTBALL_KEY

def obtener_partidos_hoy():
    url = "https://v3.football.api-sports.io/fixtures?live=all"

    headers = {
        "x-apisports-key": API_FOOTBALL_KEY
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("❌ Error API partidos:", response.status_code)
            return []

        data = response.json()

        partidos = []

        for match in data.get("response", []):
            partidos.append({
                "id": match["fixture"]["id"],
                "home": match["teams"]["home"]["name"],
                "away": match["teams"]["away"]["name"]
            })

        return partidos

    except Exception as e:
        print("❌ Error obteniendo partidos:", e)
        return []
