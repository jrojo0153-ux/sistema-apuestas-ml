import pandas as pd


def crear_features(df):
    """
    Genera features robustas para el modelo
    Compatible con datos incompletos (Odds API)
    """

    if df is None or df.empty:
        print("⚠️ DataFrame vacío en features")
        return pd.DataFrame()

    df = df.copy()

    # -------------------------
    # 1. ASEGURAR COLUMNAS
    # -------------------------
    columnas_necesarias = [
        "goles_local",
        "goles_visitante",
        "cuota_local",
        "cuota_empate",
        "cuota_visitante"
    ]

    for col in columnas_necesarias:
        if col not in df.columns:
            print(f"⚠️ Columna faltante: {col} → usando default")

            if "cuota" in col:
                df[col] = 2.0
            else:
                df[col] = 1

    # -------------------------
    # 2. LIMPIEZA BÁSICA
    # -------------------------
    df = df.fillna({
        "goles_local": 1,
        "goles_visitante": 1,
        "cuota_local": 2.0,
        "cuota_empate": 3.0,
        "cuota_visitante": 2.0
    })

    # -------------------------
    # 3. FEATURES PRINCIPALES
    # -------------------------
    df["diff_goles"] = df["goles_local"] - df["goles_visitante"]

    df["ratio_cuotas"] = df["cuota_local"] / df["cuota_visitante"]

    df["prob_impl_local"] = 1 / df["cuota_local"]
    df["prob_impl_empate"] = 1 / df["cuota_empate"]
    df["prob_impl_visitante"] = 1 / df["cuota_visitante"]

    # -------------------------
    # 4. SELECCIÓN FINAL
    # -------------------------
    features = [
        "diff_goles",
        "cuota_local",
        "cuota_empate",
        "cuota_visitante",
        "ratio_cuotas",
        "prob_impl_local",
        "prob_impl_empate",
        "prob_impl_visitante"
    ]

    X = df[features]

    print(f"🧠 Features generadas: {X.shape}")

    return X
