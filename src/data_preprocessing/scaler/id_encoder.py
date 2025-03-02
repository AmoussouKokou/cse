from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class IdEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Identité Encoder : Ne modifie pas les données, agit comme un encodeur neutre.
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
        'category': ['A', 'B', 'C', 'A', 'B']
    })
    
    encoder = IdEncoder()
    encoder.fit(data)
    transformed_data = encoder.transform(data)
    print(transformed_data)