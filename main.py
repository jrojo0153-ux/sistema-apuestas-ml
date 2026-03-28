"""
main.py — Orquestador principal del sistema de predicción
Versión simplificada para GitHub Actions
"""

import sys
from pathlib import Path

# Ajustar path para imports
sys.path.append(str(Path(__file__).parent))

# Imports del sistema
from data.loader import cargar_datos
from data.features import IngenieroFeatures
from models.ensemble import EnsemblePredictor
from engine.ev_kelly import MotorEV
from engine.backtester import Backtester
from portfolio.bankroll import GestorBankroll

def main():
    print("🚀 Iniciando sistema de apuestas...")

    # 1. Cargar datos
    print("📊 Cargando datos...")
    df = cargar_datos()

    # 2. Features
    print("⚙️ Generando features...")
    fe = IngenieroFeatures()
    X, y = fe.fit_transform(df)

    # 3. Modelo
    print("🤖 Entrenando modelo...")
    model = EnsemblePredictor()
    model.fit(X, y)

    # 4. Predicciones
    print("🔮 Generando predicciones...")
    probs = model.predict_proba(X)

    # 5. EV + Kelly
    print("💰 Calculando valor esperado...")
    motor = MotorEV()
    señales = motor.generar_senales(df, probs)

    # 6. Backtesting
    print("📉 Ejecutando backtesting...")
    bt = Backtester()
    resultado = bt.run(df, señales)

    # 7. Bankroll
    print("🏦 Gestionando bankroll...")
    gestor = GestorBankroll()
    for s in señales:
        gestor.registrar_apuesta(s)

    print("✅ Sistema finalizado correctamente")

if __name__ == "__main__":
    main()
