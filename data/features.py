"""
data/features.py — Ingeniería de features (simplificada)
"""

import pandas as pd
import numpy as np


class IngenieroFeatures:
    """
    Convierte datos de partidos en variables útiles para ML
    """

    def __init__(self):
        self.feature_names_ = []
        self.fitted_ = False

    def fit_transform(self, df: pd.DataFrame):
        """
        Entrenamiento: crea features y devuelve X, y
        """
        df = self._crear_features(df)

        X = df[self.feature_names_].values
        y = df["resultado"]

        self.fitted_ = True

        return X, y

    def transform(self, df: pd.DataFrame):
        """
        Inferencia: usa mismas features
        """
        if not self.fitted_:
            raise RuntimeError("Primero debes ejecutar fit_transform()")

        df = self._crear_features(df)

        return df[self.feature_names_].values

    def _crear_features(self, df: pd.DataFrame):
        df = df.copy()

        # 🎯 Feature 1: diferencia de goles
        df["diff_goles"] = df["goles_local"] - df["goles_visitante"]

        # 🎯 Feature 2: total de goles
        df["total_goles"] = df["goles_local"] + df["goles_visitante"]

        # 🎯 Feature 3: ratio ofensivo
        df["ratio_goles"] = df["goles_local"] / (df["goles_visitante"] + 1)

        # Lista de features usadas
        self.feature_names_ = [
            "diff_goles",
            "total_goles",
            "ratio_goles",
        ]

        return df
