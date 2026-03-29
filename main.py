import pandas as pd

# DATA
from data.live_matches import obtener_partidos_hoy
from data.loader import obtener_cuotas

# ENGINE
from engine.ev_kelly import calcular_ev_y_kelly

# MODELOS
from models.model import cargar_modelo


def main():
    print("🚀 SISTEMA LIVE PRO INICIADO")

    try:
        # 1. Obtener partidos
        partidos = obtener_partidos_hoy()
        print(f"📅 Partidos próximos: {len(partidos)}")

        if not partidos:
            print("❌ No hay partidos")
            return

        # 2. Obtener cuotas
        cuotas = obtener_cuotas()
        print(f"🌍 Cuotas disponibles: {len(cuotas)}")

        if not cuotas:
            print("❌ No hay cuotas")
            return

        # 3. Convertir a DataFrame
        df = pd.DataFrame(partidos)

        # Validación mínima
        if df.empty:
            print("❌ DataFrame vacío")
            return

        print(f"📊 Partidos analizados: {len(df)}")

        # 4. Cargar modelo
        modelo = cargar_modelo()

        if modelo is None:
            print("❌ Modelo no entrenado")
            return

        # 5. Predicción (segura)
        try:
            df["prob_local"] = modelo.predict_proba(df)[:, 1]
        except Exception:
            print("⚠️ Error en predicción, usando dummy")
            df["prob_local"] = 0.5

        # 6. EV + Kelly
        try:
            df = calcular_ev_y_kelly(df)
        except Exception as e:
            print(f"⚠️ Error EV/Kelly: {e}")
            return

        # 7. Resultados
        print("\n🔥 TOP APUESTAS 🔥")
        print(df.sort_values(by="ev", ascending=False).head(5))

    except Exception as e:
        print(f"❌ ERROR GENERAL: {e}")


if __name__ == "__main__":
    main()
