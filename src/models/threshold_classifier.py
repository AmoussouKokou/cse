from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper pour un classificateur scikit-learn permettant d'ajuster le seuil de décision.
    """
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold

    def fit(self, X, y):
        """ Entraîne le modèle de base. """
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        """ Prédit en utilisant le seuil personnalisé. """
        probas = self.base_model.predict_proba(X)[:, 1]  # Probabilité d'appartenance à la classe positive
        return (probas >= self.threshold).astype(int)  # Applique le seuil
    
    def predict_proba(self, X):
        """ Retourne les probabilités, utile pour certaines métriques. """
        return self.base_model.predict_proba(X)

    def set_params(self, **params):
        """ Permet de modifier les paramètres du modèle et du seuil. """
        for param, value in params.items():
            if param == "threshold":
                self.threshold = value
            else:
                self.base_model.set_params(**{param: value})
        return self

    def get_params(self, deep=True):
        """ Retourne les paramètres pour GridSearch ou RandomizedSearch. """
        params = self.base_model.get_params(deep)
        params["threshold"] = self.threshold
        return params
