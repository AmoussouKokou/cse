import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropMvImputer(TransformerMixin, BaseEstimator):
    """
    Un transformateur qui supprime les lignes contenant des valeurs manquantes.
    Hérite de BaseEstimator et TransformerMixin pour être compatible avec scikit-learn.
    """
    
    def __init__(self, how: str = "any"):
        """
        Initialise l'imputer.
        
        :param how: "any" pour supprimer les lignes contenant au moins une valeur manquante,
                    "all" pour supprimer les lignes où toutes les valeurs sont manquantes.
        """
        self.how = how

    def fit(self, X, y=None):
        """
        Apprend les caractéristiques des données (pas nécessaire pour ce transformateur).
        
        :param X: DataFrame ou array-like, les données d'entrée
        :param y: Ignoré, pour compatibilité avec scikit-learn
        :return: self
        """
        return self

    def transform(self, X):
        """
        Transforme les données en supprimant les lignes contenant des valeurs manquantes.
        
        :param X: DataFrame ou array-like, les données d'entrée
        :return: DataFrame sans les valeurs manquantes
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        return X.dropna(how=self.how).reset_index(drop=True)

if __name__ == '__main__':
    import pandas as pd
    from sklearn.pipeline import Pipeline

    # Exemple de données
    data = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [5, np.nan, np.nan, 8],
        "C": [9, 10, 11, 12]
    })

    print("Données avant traitement:")
    print(data)

    # Initialisation du transformateur
    imputer = DropMvImputer(how="all")

    # Transformation des données
    data_cleaned = imputer.fit_transform(data)

    print("\nDonnées après suppression des valeurs manquantes:")
    print(data_cleaned)

    # Exemple d'intégration dans un pipeline sklearn
    pipeline = Pipeline([
        ("drop_mv", DropMvImputer(how="any"))
    ])

    data_pipeline_cleaned = pipeline.fit_transform(data)
    print("\nDonnées après passage dans le pipeline:")
    print(data_pipeline_cleaned)
