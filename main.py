from data.historico import cargar_partidos
from data.loader import cargar_cuotas
from features.features import crear_features
from models.ensemble import EnsemblePredictor
from engine.ev_kelly import MotorEV


def main():
    print("🚀 SISTEMA LIVE PRO INICIADO")

    # ----------------------------------
    # 1. PARTIDOS (Football API)
    # ----------------------------------
    df_partidos = cargar_partidos()

    if df_partidos.empty:
        print("❌ No hay partidos disponibles")
    else:
        print(f"📅 Partidos próximos: {len(df_partidos)}")

    # ----------------------------------
    # 2. CUOTAS (Odds API)
    # ----------------------------------
    df_odds = cargar_cuotas()

    if df_odds.empty:
        print("❌ No hay cuotas")
        return

    print(f"🌍 Cuotas disponibles: {len(df_odds)}")

    # ----------------------------------
    # 3. DATA BASE (ODDS)
    # ----------------------------------
    df = df_odds.copy()

    # 🔥 FIX CRÍTICO → evitar errores de features
    if "goles_local" not in df.columns:
        df["goles_local"] = 1

    if "goles_visitante" not in df.columns:
        df["goles_visitante"] = 1

    if "resultado" not in df.columns:
        df["resultado"] = 0

    print(f"📊 Partidos analizados: {len(df)}")

    # ----------------------------------
    # 4. FEATURES
    # ----------------------------------
    X = crear_features(df)

    if X.empty:
        print("❌ Error generando features")
        return

    # ----------------------------------
    # 5. MODELO (DUMMY)
    # ----------------------------------
    model = EnsemblePredictor()

    try:
        probs = model.predict_proba(X)
    except Exception as e:
        print(f"❌ Error modelo: {e}")
        return

    # ----------------------------------
    # 6. EV + KELLY
    # ----------------------------------
    motor = MotorEV()

    try:
        señales = motor.generar_senales(df, probs)
    except Exception as e:
        print(f"❌ Error EV/Kelly: {e}")
        return

    print(f"💰 Señales generadas: {len(señales)}")

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
