import numpy as np


class EnsemblePredictor:

    def predict_proba(self, X):

        if X is None or X.empty:
            print("❌ Features vacías")
            return np.array([])

        try:
            prob_local = X["prob_impl_local"].values
            prob_empate = X["prob_impl_empate"].values
            prob_visitante = X["prob_impl_visitante"].values

            total = prob_local + prob_empate + prob_visitante

            prob_local = prob_local / total
            prob_empate = prob_empate / total
            prob_visitante = prob_visitante / total

            probs = np.vstack([
                prob_local,
                prob_empate,
                prob_visitante
            ]).T

            return probs

        except Exception as e:
            print(f"❌ Error en modelo: {e}")
            return np.array([])
