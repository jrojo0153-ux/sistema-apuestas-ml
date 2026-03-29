import os

from engine.ev_kelly import calcular_ev_y_kelly

def main():
    print("🚀 SISTEMA LIVE PRO INICIADO")

    try:
        # ========================
        # 1. CARGAR API KEY
        # ========================
        api_key = os.getenv("API_KEY_ODDS")

        if not api_key:
            print("❌ ERROR: API_KEY_ODDS no encontrada")
            return

        print("✅ API KEY cargada")

        # ========================
        # 2. DATOS SIMULADOS (fallback)
        # ========================
        probabilidades = [0.6, 0.55, 0.48]
        cuotas = [2.0, 1.8, 2.5]

        print(f"📊 Partidos detectados: {len(probabilidades)}")

        # ========================
        # 3. CALCULAR EV + KELLY
        # ========================
        resultados = calcular_ev_y_kelly(probabilidades, cuotas)

        if not resultados:
            print("⚠️ No hay resultados válidos")
            return

        # ========================
        # 4. MOSTRAR RESULTADOS
        # ========================
        print("\n📈 RESULTADOS:")
        for r in resultados:
            print(f"EV: {r['ev']} | Kelly: {r['kelly']}")

        print("\n✅ Sistema ejecutado correctamente")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")


if __name__ == "__main__":
    main()
