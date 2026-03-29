import requests


def obtener_partidos_hoy():
    """
    Obtiene partidos del día (mock o API real)
    """

    try:
        # 🔥 EJEMPLO SIMPLE (puedes conectar API luego)
        partidos = [
            {"home": "Eibar", "away": "Las Palmas"},
            {"home": "Werder Bremen", "away": "SGS Essen"},
            {"home": "Wolfsburg", "away": "Union Berlin"},
            {"home": "Cultural Leonesa", "away": "Andorra CF"},
            {"home": "Zaragoza", "away": "Racing Santander"},
            {"home": "Freiburg", "away": "Hoffenheim"},
            {"home": "Almería", "away": "Real Sociedad B"},
            {"home": "Athletico PR", "away": "Botafogo"}
        ]

        return partidos

    except Exception as e:
        print("❌ Error obteniendo partidos:", e)
        return []
