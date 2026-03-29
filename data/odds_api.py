import requests
import random

def obtener_cuotas(partidos):
    resultados = []

    for partido in partidos:
        try:
            # SIMULACIÓN ROBUSTA (puedes cambiar por API real luego)
            cuota_home = round(random.uniform(1.5, 3.0), 2)
            cuota_away = round(random.uniform(1.5, 3.0), 2)

            resultados.append({
                "partido": f"{partido['home']} vs {partido['away']}",
                "home_odds": cuota_home,
                "away_odds": cuota_away
            })

        except Exception as e:
            print("❌ Error cuotas:", e)

    return resultados
