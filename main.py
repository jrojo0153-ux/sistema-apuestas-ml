import pandas as pd

from data.live_matches import obtener_partidos_hoy
from data.odds_api import obtener_cuotas
from features.features import crear_features
from models.ensemble import EnsemblePredictor
from utils.kelly import calcular_kelly


def main():

    print("🚀 SISTEMA LIVE PRO INICIADO")

    # -------------------------
    # 1. PARTIDOS
    # -------------------------
    partidos = obtener_partidos_hoy()

    if partidos is None or len(partidos) == 0:
        print("❌ No hay partidos hoy → fin")
        return

    print(f"📅 Partidos próximos: {len(partidos)}")

    # -------------------------
    # 2. CUOTAS
    # -------------------------
    df_odds = obtener_cuotas(partidos)

    if df_odds is None or df_odds.empty:
        print("❌ No hay cuotas")
        return

    print(f"🌍 Cuotas disponibles: {len(df_odds)}")

    # -------------------------
    # 3. FEATURES
    # -------------------------
    X = crear_features(df_odds)

    if X is None or X.empty:
        print("❌ No se generaron features")
        return

    print(f"📊 Partidos analizados: {len(X)}")

    # -------------------------
    # 4. MODELO
    # -------------------------
    modelo = EnsemblePredictor()
    probs = modelo.predict_proba(X)

    if probs is None or len(probs) == 0:
        print("❌ No se pudieron generar probabilidades")
        return

    print(f"🧠 Probabilidades generadas: {len(probs)}")

    # -------------------------
    # 5. EV + KELLY
    # -------------------------
    resultados = []

    for i in range(len(X)):

        try:
            cuota_local = X.iloc[i]["cuota_local"]
            cuota_empate = X.iloc[i]["cuota_empate"]
            cuota_visitante = X.iloc[i]["cuota_visitante"]

            p_local, p_empate, p_visitante = probs[i]

            # EV
            ev_local = (p_local * cuota_local) - 1
            ev_empate = (p_empate * cuota_empate) - 1
            ev_visitante = (p_visitante * cuota_visitante) - 1

            # Kelly
            k_local = calcular_kelly(p_local, cuota_local)
            k_empate = calcular_kelly(p_empate, cuota_empate)
            k_visitante = calcular_kelly(p_visitante, cuota_visitante)

            resultados.append({
                "match": df_odds.iloc[i].get("match", "N/A"),
                "EV_local": round(ev_local, 3),
                "EV_empate": round(ev_empate, 3),
                "EV_visitante": round(ev_visitante, 3),
                "Kelly_local": round(k_local, 3),
                "Kelly_empate": round(k_empate, 3),
                "Kelly_visitante": round(k_visitante, 3),
            })

        except Exception as e:
            print(f"⚠️ Error en cálculo partido {i}: {e}")
            continue

    # -------------------------
    # 6. RESULTADOS
    # -------------------------
    if len(resultados) == 0:
        print("❌ No hay resultados válidos")
        return

    df_resultados = pd.DataFrame(resultados)

    print("\n💰 RESULTADOS:")
    print(df_resultados)

    print("\n✅ SISTEMA FINALIZADO")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    main()
