from data.historico import cargar_partidos
from data.loader import cargar_cuotas
from features.features import crear_features
from models.ensemble import EnsemblePredictor
from engine.ev_kelly import MotorEV


def main():
    print("🚀 SISTEMA LIVE PRO INICIADO")

    # -------------------------
    # 1. PARTIDOS (Football)
    # -------------------------
    df_partidos = cargar_partidos()

    if df_partidos.empty:
        print("❌ No hay partidos disponibles")
        return

    # -------------------------
    # 2. CUOTAS (Odds)
    # -------------------------
    df_odds = cargar_cuotas()

    if df_odds.empty:
        print("❌ No hay cuotas")
        return

    # -------------------------
    # 3. USAMOS CUOTAS DIRECTO (simplificado)
    # -------------------------
    df = df_odds.copy()

    print(f"📊 Partidos analizados: {len(df)}")

    # -------------------------
    # 4. FEATURES
    # -------------------------
    X = crear_features(df)

    # -------------------------
    # 5. MODELO (dummy por ahora)
    # -------------------------
    model = EnsemblePredictor()
    probs = model.predict_proba(X)

    # -------------------------
    # 6. EV + KELLY
    # -------------------------
    motor = MotorEV()
    señales = motor.generar_senales(df, probs)

    print(f"💰 Señales generadas: {len(señales)}")

    print("✅ SISTEMA FINALIZADO")


if __name__ == "__main__":
    main()
