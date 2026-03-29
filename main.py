from data.live_matches import obtener_partidos_hoy
from data.loader import obtener_cuotas
from models.model import (
    cargar_modelo,
    entrenar_modelo,
    predecir_probabilidades
)
from engine.ev_kelly import calcular_ev_y_kelly
from portfolio.bankroll import calcular_apuesta
from utils.telegram import enviar_alerta


def main():
    print("🚀 SISTEMA PRO INICIADO")

    partidos = obtener_partidos_hoy()
    print(f"📊 Partidos encontrados: {len(partidos)}")

    cuotas = obtener_cuotas(partidos)

    modelo = cargar_modelo()

    if modelo is None:
        print("⚠️ Modelo no encontrado, entrenando...")
        modelo = entrenar_modelo()
    else:
        print("✅ Modelo cargado")

    probs = predecir_probabilidades(modelo, cuotas)

    resultados = calcular_ev_y_kelly(probs, cuotas)

    for i, r in enumerate(resultados):
        if r["ev"] > 0.05:  # 🔥 FILTRO REAL

            apuesta = calcular_apuesta(1000, r["kelly"])

            mensaje = f"""
🔥 VALUE BET

{partidos[i]["home"]} vs {partidos[i]["away"]}

EV: {r["ev"]}
Kelly: {r["kelly"]}
Apuesta: ${apuesta}
"""

            print(mensaje)

            enviar_alerta(mensaje)

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
