"""
config.py — Configuración central del sistema
"""

# ─────────────────────────────────────────────
# 💰 BANKROLL (GESTIÓN DE DINERO)
# ─────────────────────────────────────────────
BANKROLL_INICIAL = 1000.0
KELLY_FRACCION = 0.25
APUESTA_MINIMA = 1.0
APUESTA_MAXIMA_PCT = 0.10
STOP_LOSS_PCT = 0.30
MAX_APUESTAS_DIA = 10

# ─────────────────────────────────────────────
# 📈 FILTROS DE APUESTAS (EV+)
# ─────────────────────────────────────────────
EV_MINIMO = 0.03
PROB_MINIMA = 0.40
PROB_MAXIMA = 0.90
CUOTA_MINIMA = 1.30
CUOTA_MAXIMA = 8.00
EDGE_MINIMO = 0.02

# ─────────────────────────────────────────────
# 🤖 MODELO
# ─────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 📊 FEATURES
# ─────────────────────────────────────────────
FEATURES_BASE = []
TARGET = "resultado"

# Columnas de cuotas (ajustar si usas datos reales)
CUOTAS_COLS = ["cuota_local", "cuota_empate", "cuota_visitante"]

# ─────────────────────────────────────────────
# 📁 PATHS
# ─────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
LOG_PATH = "logs/historial.csv"
REPORT_PATH = "reports/"
