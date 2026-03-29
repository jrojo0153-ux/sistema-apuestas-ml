from data.live_matches import obtener_partidos
from ml.model import entrenar_modelo, cargar_modelo, predecir
from core.value import evaluar_apuesta
from portfolio.bankroll import calcular_apuesta, BANKROLL_INICIAL

def main():
    print("🚀 SISTEMA PRO IA INICIADO")

    partidos = obtener_partidos()
    print(f"📊 Partidos encontrados: {len(partidos)}")

    if not partidos:
        print("❌ No hay partidos disponibles")
        return

    modelo = cargar_modelo()

    if modelo is None:
        print("⚠️ No existe modelo, entrenando...")
        modelo = entrenar_modelo()

    bankroll = BANKROLL_INICIAL

    for partido in partidos:
        prob = predecir(modelo, partido)

        ev, kelly = evaluar_apuesta(prob, partido["home_odds"])

        if ev > 0:
            stake = calcular_apuesta(kelly, bankroll)

            print("\n🔥 VALUE BET")
            print(f"{partido['home_team']} vs {partido['away_team']}")
            print(f"Prob: {round(prob, 2)}")
            print(f"EV: {round(ev, 4)}")
            print(f"Kelly: {round(kelly, 4)}")
            print(f"Apuesta: ${stake}")

    print("\n✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
