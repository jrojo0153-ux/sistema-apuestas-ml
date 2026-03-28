import requests
import os


def obtener_partidos_hoy():

    API_KEY = os.getenv("API_KEY_FOOTBALL")

    if not API_KEY:
        print("❌ API_KEY_FOOTBALL no definida")
        return []

    url = "https://api.football-data.org/v4/matches"

    headers = {
        "X-Auth-Token": API_KEY
    }

    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Error API Football: {response.status_code}")
            return []

        data = response.json()

        partidos = []

        for match in data.get("matches", []):

            partidos.append({
                "home": match["homeTeam"]["name"],
                "away": match["awayTeam"]["name"],
                "date": match["utcDate"]
            })

        return partidos

    except Exception as e:
        print(f"❌ Error obteniendo partidos: {e}")
        return []
