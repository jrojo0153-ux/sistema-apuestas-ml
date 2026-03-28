import requests
import pandas as pd
import os

API_KEY = os.getenv("API_KEY_ODDS")

def cargar_datos():
    url = "https://api.the-odds-api.com/v4/sports/soccer/odds"

    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": "h2h"
    }

    res = requests.get(url, params=params)

    if res.status_code != 200:
        raise Exception("Error en Odds API")

    data = res.json()

    partidos = []

    for game in data:
        try:
            home = game["home_team"]
            away = game["away_team"]

            book = game["bookmakers"][0]
            outcomes = book["markets"][0]["outcomes"]

            if len(outcomes) == 3:
                cuota_local = outcomes[0]["price"]
                cuota_empate = outcomes[1]["price"]
                cuota_visitante = outcomes[2]["price"]
            else:
                cuota_local = outcomes[0]["price"]
                cuota_empate = 3.0
                cuota_visitante = outcomes[1]["price"]

            partidos.append({
                "equipo_local": home,
                "equipo_visitante": away,
                "cuota_local": cuota_local,
                "cuota_empate": cuota_empate,
                "cuota_visitante": cuota_visitante,

                # dummy para features
                "goles_local": 1,
                "goles_visitante": 1,
                "resultado": 0
            })

        except:
            continue

    df = pd.DataFrame(partidos)

    print(f"🌍 Live partidos: {len(df)}")

    return df
