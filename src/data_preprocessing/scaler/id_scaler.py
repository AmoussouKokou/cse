from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class IdScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Identité Scaler : Ne modifie pas les données, agit comme un scaler neutre.
        """
        pass
    
    def fit(self, X, y=None):
        """
        Ne fait rien pendant l'apprentissage.
        :param X: Données d'entraînement
        :param y: Variable cible (optionnelle, non utilisée)
        """
        return self
    
    def transform(self, X):
        """
        Retourne les données inchangées.
        :param X: Données d'entrée
        :return: Données inchangées
        """
        return X
    
    def inverse_transform(self, X):
        """
        Retourne les données inchangées lors de l'inverse transformation.
        :param X: Données transformées
        :return: Données inchangées
        """
        return X

# Exemple d'utilisation
if __name__ == "__main__":
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, 20, 30, 40, 50]
    })
    
    scaler = IdScaler()
    scaler.fit(data)
    transformed_data = scaler.transform(data)
    print(transformed_data)
