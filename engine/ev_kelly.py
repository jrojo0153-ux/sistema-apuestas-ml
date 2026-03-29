def calcular_ev_y_kelly(probabilidades, cuotas):
    """
    Calcula el valor esperado (EV) y el porcentaje de Kelly para cada apuesta.

    Args:
        probabilidades (list[float]): probabilidades del modelo (0-1)
        cuotas (list[float]): cuotas decimales

    Returns:
        list[dict]: resultados con EV y Kelly
    """

    resultados = []

    # Validación básica
    if not probabilidades or not cuotas:
        return []

    if len(probabilidades) != len(cuotas):
        print("⚠️ Error: tamaños diferentes entre probabilidades y cuotas")
        return []

    for i in range(len(probabilidades)):
        try:
            p = probabilidades[i]
            cuota = cuotas[i]

            # Validaciones
            if p <= 0 or cuota <= 1:
                continue

            # EV
            ev = (p * cuota) - 1

            # Kelly
            kelly = ((p * (cuota - 1)) - (1 - p)) / (cuota - 1)

            resultados.append({
                "probabilidad": p,
                "cuota": cuota,
                "ev": round(ev, 4),
                "kelly": round(kelly, 4)
            })

        except Exception as e:
            print(f"⚠️ Error en cálculo: {e}")
            continue

    return resultados
