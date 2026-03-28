from data.loader import cargar_datos
from data.historico import cargar_partidos_hoy
from features.features import crear_features
from models.ensemble import EnsemblePredictor
from engine.ev_kelly import MotorEV


def main():
    print("🚀 SISTEMA LIVE INICIADO")

    # ----------------------------------
    # 1. PARTIDOS HOY (Football-Data)
    # ----------------------------------
    df_teams = cargar_partidos_hoy()

    if df_teams.empty:
        print("❌ No hay partidos hoy → fin")
        return

    # ----------------------------------
    # 2. CUOTAS (Odds API)
    # ----------------------------------
    df_odds = cargar_datos()

    if df_odds.empty:
        print("❌ No hay cuotas disponibles")
        return

    # ----------------------------------
    # 3. USAMOS DIRECTO ODDS (simplificado)
    # ----------------------------------
    df = df_odds.copy()

    print(f"📊 Partidos con cuotas: {len(df)}")

    # ----------------------------------
    # 4. FEATURES
    # ----------------------------------
    X = crear_features(df)

    # ----------------------------------
    # 5. MODELO (FAKE por ahora)
    # ----------------------------------
    model = EnsemblePredictor()

    # ⚠️ NO entrenamos (no hay histórico real)
    probs = model.predict_proba(X)

    # ----------------------------------
    # 6. EV + KELLY
    # ----------------------------------
    motor = MotorEV()

    señales = motor.generar_senales(df, probs)

    print(f"💰 Señales generadas: {len(señales)}")

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
