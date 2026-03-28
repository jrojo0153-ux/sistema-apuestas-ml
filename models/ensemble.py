"""
models/ensemble.py — Modelo ensemble (combinación de modelos)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class EnsemblePredictor:
    """
    Combina múltiples modelos para mejorar predicción
    """

    def __init__(self):
        self.model1 = LogisticRegression(max_iter=1000)
        self.model2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.fitted_ = False

    def fit(self, X, y):
        """
        Entrena ambos modelos
        """
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        self.fitted_ = True

    def predict_proba(self, X):
        """
        Promedia probabilidades de ambos modelos
        """
        if not self.fitted_:
            raise RuntimeError("Modelo no entrenado")

        p1 = self.model1.predict_proba(X)
        p2 = self.model2.predict_proba(X)

        # Promedio simple (ensemble)
        probs = (p1 + p2) / 2

        return probs

    def predict(self, X):
        """
        Devuelve clase final
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
