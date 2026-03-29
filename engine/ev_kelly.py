def calcular_ev_y_kelly(probabilidades, cuotas):
    resultados = []

    if not probabilidades or not cuotas:
        return []

    if len(probabilidades) != len(cuotas):
        print("⚠️ Tamaño diferente entre probabilidades y cuotas")
        return []

    for p, cuota in zip(probabilidades, cuotas):
        try:
            if cuota <= 1:
                continue

            ev = (p * cuota) - 1
            kelly = ((p * (cuota - 1)) - (1 - p)) / (cuota - 1)

            resultados.append({
                "ev": round(ev, 4),
                "kelly": round(kelly, 4)
            })

        except Exception as e:
            print(f"⚠️ Error cálculo: {e}")
            continue

    return resultados
