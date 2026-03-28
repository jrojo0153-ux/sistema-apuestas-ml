import requests
import pandas as pd
import os

API_KEY = os.getenv("API_KEY_FOOTBALL")

def cargar_historico():
    url = "https://api.football-data.org/v4/competitions/PL/matches"

    headers = {"X-Auth-Token": API_KEY}

    res = requests.get(url, headers=headers)
    data = res.json()

    partidos = []

    for m in data["matches"]:
        if m["status"] != "FINISHED":
            continue

        goles_local = m["score"]["fullTime"]["home"]
        goles_visitante = m["score"]["fullTime"]["away"]

        if goles_local > goles_visitante:
            resultado = 0
        elif goles_local == goles_visitante:
            resultado = 1
        else:
            resultado = 2

        partidos.append({
            "equipo_local": m["homeTeam"]["name"],
            "equipo_visitante": m["awayTeam"]["name"],
            "goles_local": goles_local,
            "goles_visitante": goles_visitante,
            "resultado": resultado
        })

    df = pd.DataFrame(partidos)

    print(f"📚 Histórico cargado: {len(df)}")

    return df
