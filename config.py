from typing import Final
from pathlib import Path

# 💰 BANKROLL
BANKROLL_INICIAL: Final[float] = 1000.0
KELLY_FRACCION: Final[float] = 0.10
APUESTA_MINIMA: Final[float] = 1.0
APUESTA_MAXIMA_PCT: Final[float] = 0.03
STOP_LOSS_PCT: Final[float] = 0.30
MAX_APUESTAS_DIA: Final[int] = 5

# 📈 FILTROS
EV_MINIMO: Final[float] = 0.05
PROB_MINIMA: Final[float] = 0.40
PROB_MAXIMA: Final[float] = 0.80
CUOTA_MINIMA: Final[float] = 1.30
CUOTA_MAXIMA: Final[float] = 5.00
EDGE_MINIMO: Final[float] = 0.02

# 🤖 MODELO
MODEL_PATH: Final[Path] = Path("models/modelo.pkl")