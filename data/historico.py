import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = os.getenv("API_KEY_FOOTBALL")


def cargar_partidos_hoy():
    """
    Obtiene partidos de HOY (plan gratuito compatible)
    """

    hoy = datetime.utcnow().date()

    url = f"https://api.football-data.org/v4/matches?dateFrom={hoy}&dateTo={hoy}"

    headers = {
        "X-Auth-Token": API_KEY
    }

    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        print(f"❌ Error API: {res.status_code}")
        print(res.text)
        return pd.DataFrame()

    data = res.json()

    matches = data.get("matches", [])

    if not matches:
        print("⚠️ No hay partidos hoy")
        return pd.DataFrame()

    partidos = []

    for m in matches:
        try:
            partidos.append({
                "equipo_local": m["homeTeam"]["name"],
                "equipo_visitante": m["awayTeam"]["name"],

                # placeholders (se combinarán con odds)
                "goles_local": 1,
                "goles_visitante": 1,
                "resultado": 0
            })

        except:
            continue

    df = pd.DataFrame(partidos)

    print(f"📅 Partidos hoy: {len(df)}")

    return df
