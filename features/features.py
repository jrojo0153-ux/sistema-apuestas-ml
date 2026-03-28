import pandas as pd


def crear_features(df):

    if df is None or df.empty:
        print("⚠️ DataFrame vacío en features")
        return pd.DataFrame()

    df = df.copy()

    # -------------------------
    # ASEGURAR CUOTAS
    # -------------------------
    for col in ["cuota_local", "cuota_empate", "cuota_visitante"]:
        if col not in df.columns:
            print(f"⚠️ Falta {col}, usando default")
            df[col] = 2.0

    df = df.fillna({
        "cuota_local": 2.0,
        "cuota_empate": 3.0,
        "cuota_visitante": 2.0
    })

    # -------------------------
    # FEATURES CLAVE
    # -------------------------
    df["prob_impl_local"] = 1 / df["cuota_local"]
    df["prob_impl_empate"] = 1 / df["cuota_empate"]
    df["prob_impl_visitante"] = 1 / df["cuota_visitante"]

    df["ratio_cuotas"] = df["cuota_local"] / df["cuota_visitante"]

    # -------------------------
    # OUTPUT
    # -------------------------
    X = df[[
        "cuota_local",
        "cuota_empate",
        "cuota_visitante",
        "prob_impl_local",
        "prob_impl_empate",
        "prob_impl_visitante",
        "ratio_cuotas"
    ]]

    print(f"🧠 Features generadas: {X.shape}")

    return X
