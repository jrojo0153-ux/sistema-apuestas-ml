import requests
import pandas as pd
import os


API_KEY = os.getenv("API_KEY_FOOTBALL")


def cargar_historico():
    """
    Carga partidos históricos reales desde Football-Data
    Versión robusta con manejo de errores
    """

    url = "https://api.football-data.org/v4/competitions/PL/matches"

    headers = {
        "X-Auth-Token": API_KEY
    }

    res = requests.get(url, headers=headers)

    # -------------------------
    # 1. VALIDAR STATUS
    # -------------------------
    if res.status_code != 200:
        print(f"❌ Error API: {res.status_code}")
        print(res.text)
        return pd.DataFrame()

    data = res.json()

    # -------------------------
    # 2. VALIDAR RESPUESTA
    # -------------------------
    matches = data.get("matches", None)

    if matches is None:
        print("❌ ERROR: 'matches' no existe en respuesta")
        print("DEBUG:", data)
        return pd.DataFrame()

    if len(matches) == 0:
        print("⚠️ No hay partidos disponibles")
        return pd.DataFrame()

    # -------------------------
    # 3. PROCESAR DATOS
    # -------------------------
    partidos = []

    for m in matches:

        # Solo partidos terminados
        if m.get("status") != "FINISHED":
            continue

        try:
            goles_local = m["score"]["fullTime"]["home"]
            goles_visitante = m["score"]["fullTime"]["away"]

            # Validar datos
            if goles_local is None or goles_visitante is None:
                continue

            # Resultado
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

        except Exception as e:
            continue

    # -------------------------
    # 4. DATAFRAME FINAL
    # -------------------------
    df = pd.DataFrame(partidos)

    print(f"📚 Histórico válido: {len(df)} partidos")

    return df
