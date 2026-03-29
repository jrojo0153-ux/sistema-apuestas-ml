from data.live_matches import obtener_partidos_hoy
from data.odds_api import obtener_cuotas
from models.model import cargar_modelo, predecir
from engine.ev_kelly import calcular_ev_y_kelly
from portfolio.bankroll import calcular_apuesta
from alerts.telegram import enviar_alerta


def main():
    print("🚀 SISTEMA PRO IA INICIADO")

    bankroll = 1000

    partidos = obtener_partidos_hoy()

    print(f"📊 Partidos encontrados: {len(partidos)}")

    if not partidos:
        print("❌ No hay partidos")
        return

    cuotas = obtener_cuotas(partidos)

    modelo = cargar_modelo()

    probabilidades = predecir(modelo, len(cuotas))

    resultados = calcular_ev_y_kelly(probabilidades, [c["home_odds"] for c in cuotas])

    for i, r in enumerate(resultados):
        if r["ev"] > 0:
            apuesta = calcular_apuesta(bankroll, r["kelly"])

            mensaje = (
                f"🔥 VALUE BET\n"
                f"{cuotas[i]['partido']}\n"
                f"EV: {r['ev']}\n"
                f"Kelly: {r['kelly']}\n"
                f"Apuesta: ${apuesta}"
            )

            print(mensaje)
            enviar_alerta(mensaje)

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
