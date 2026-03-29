from data.live_matches import obtener_partidos
from data.features import crear_features
from models.model import cargar_modelo, predecir
from engine.ev_kelly import calcular_ev_y_kelly
from portfolio.bankroll import calcular_apuesta
from alerts.telegram import enviar_alerta

def main():
    print("🚀 SISTEMA PRO INICIADO")

    # 1. Obtener partidos
    partidos = obtener_partidos()

    if not partidos:
        print("❌ No hay partidos disponibles")
        return

    print(f"📊 Partidos encontrados: {len(partidos)}")

    # 2. Features
    X = crear_features(partidos)

    if len(X) == 0:
        print("❌ No hay features")
        return

    # 3. Modelo
    modelo = cargar_modelo()

    probs = predecir(modelo, X)

    # 4. Cuotas
    cuotas = [p["cuota"] for p in partidos]

    # 5. EV + Kelly
    resultados = calcular_ev_y_kelly(probs, cuotas)

    if not resultados:
        print("❌ No hay resultados EV/Kelly")
        return

    # 6. Decisiones
    for i, r in enumerate(resultados):
        ev = r["ev"]
        kelly = r["kelly"]

        if ev > 0:
            apuesta = calcular_apuesta(kelly)

            mensaje = f"""
🔥 VALUE BET
{partidos[i]['home']} vs {partidos[i]['away']}
EV: {ev}
Kelly: {kelly}
Apuesta: ${apuesta}
"""

            print(mensaje)
            enviar_alerta(mensaje)

    print("✅ SISTEMA FINALIZADO")

if __name__ == "__main__":
    main()
