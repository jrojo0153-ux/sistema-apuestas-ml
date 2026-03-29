import random

def obtener_cuotas(partidos):
    cuotas = []

    for p in partidos:
        cuotas.append({
            "partido": f"{p['home']} vs {p['away']}",
            "home_odds": round(random.uniform(1.5, 2.8), 2),
            "away_odds": round(random.uniform(1.8, 3.5), 2)
        })

    return cuotas
