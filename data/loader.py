"""
data/loader.py — Generador de datos de fútbol (simplificado)
"""

import pandas as pd
import numpy as np


def cargar_datos(n_partidos: int = 200):
    """
    Genera datos simulados de partidos de fútbol
    """

    np.random.seed(42)

    data = []

    for i in range(n_partidos):
        equipo_local = f"Equipo_{np.random.randint(1, 20)}"
        equipo_visitante = f"Equipo_{np.random.randint(1, 20)}"

        goles_local = np.random.poisson(1.5)
        goles_visitante = np.random.poisson(1.2)

        # Resultado
        if goles_local > goles_visitante:
            resultado = 0  # local
        elif goles_local == goles_visitante:
            resultado = 1  # empate
        else:
            resultado = 2  # visitante

        data.append({
            "equipo_local": equipo_local,
            "equipo_visitante": equipo_visitante,
            "goles_local": goles_local,
            "goles_visitante": goles_visitante,
            "resultado": resultado,

            # cuotas simuladas
            "cuota_local": round(np.random.uniform(1.5, 3.5), 2),
            "cuota_empate": round(np.random.uniform(2.5, 4.0), 2),
            "cuota_visitante": round(np.random.uniform(1.5, 3.5), 2),
        })

    df = pd.DataFrame(data)

    print(f"✅ Datos generados: {len(df)} partidos")

    return df
