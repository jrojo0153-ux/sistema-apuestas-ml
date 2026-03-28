"""
models/model.py — Modelo base simple
"""

from sklearn.ensemble import RandomForestClassifier


class ModeloBase:
    """
    Modelo base usando Random Forest
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

    def fit(self, X, y):
        """
        Entrena el modelo
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicción de clases
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Probabilidades (clave para apuestas)
        """
        return self.model.predict_proba(X)
