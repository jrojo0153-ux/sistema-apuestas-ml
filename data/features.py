import numpy as np

def crear_features(partidos):
    features = []

    for p in partidos:
        features.append([
            len(p["home"]),
            len(p["away"]),
            1 if "FC" in p["home"] else 0
        ])

    return np.array(features)

    df = pd.DataFrame(partidos)

    # Validación mínima
    if "home" not in df.columns or "away" not in df.columns:
        print("❌ Datos inválidos en partidos")
        return pd.DataFrame()

    # Features básicas (placeholder)
    df["len_home"] = df["home"].apply(len)
    df["len_away"] = df["away"].apply(len)

    # Diferencia simple
    df["diff"] = df["len_home"] - df["len_away"]

    # Probabilidades dummy (para evitar errores)
    df["prob_local"] = 0.45
    df["prob_empate"] = 0.25
    df["prob_visitante"] = 0.30

    return df
