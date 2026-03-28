from data.loader import cargar_datos
from data.historico import cargar_historico
from features.features import crear_features
from models.ensemble import EnsemblePredictor
from engine.ev_kelly import MotorEV

def main():
    print("🚀 SISTEMA REAL INICIADO")

    # -------------------------
    # 1. ENTRENAR MODELO
    # -------------------------
    df_hist = cargar_historico()

    X_hist = crear_features(df_hist)
    y_hist = df_hist["resultado"]

    model = EnsemblePredictor()
    model.fit(X_hist, y_hist)

    print("✅ Modelo entrenado")

    # -------------------------
    # 2. DATOS LIVE
    # -------------------------
    df_live = cargar_datos()

    X_live = crear_features(df_live)

    # -------------------------
    # 3. PREDICCIÓN
    # -------------------------
    probs = model.predict_proba(X_live)

    # -------------------------
    # 4. EV + KELLY
    # -------------------------
    motor = MotorEV()
    señales = motor.generar_senales(df_live, probs)

    print(f"💰 Señales: {len(señales)}")

if __name__ == "__main__":
    main()
