def crear_features(df):
    """
    Genera features básicas para el modelo
    """

    df = df.copy()

    # Diferencia de goles
    df["diff_goles"] = df["goles_local"] - df["goles_visitante"]

    # Cuotas como features
    df["cuota_local"] = df["cuota_local"]
    df["cuota_empate"] = df["cuota_empate"]
    df["cuota_visitante"] = df["cuota_visitante"]

    X = df[[
        "diff_goles",
        "cuota_local",
        "cuota_empate",
        "cuota_visitante"
    ]]

    return X
