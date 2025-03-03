import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import sys
import shutil
import os

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
        self.row_with_mv_bool = X.isnull().any(axis=1) if self.how == "any" else X.isnull().all(axis=1)
        row_with_mv = self.row_with_mv_bool[self.row_with_mv_bool].index.tolist()
        row_without_mv = self.row_with_mv_bool[~self.row_with_mv_bool].index.tolist()
        
        if y is not None:
            # rows_in = any(item in y.index.tolist() for item in row_with_mv)
        
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            
            root_path_cor = root_path.replace("\\", "/")
            # print(y.index.tolist())
            if sorted(y.index.tolist()) != sorted(row_without_mv):
                print(
                    "!!! Il vous faut enlever des lignes de la cible 'y'\n"
                    "en raison de la suppression des lignes ayant des valeurs manquantes dans X.\n"
                    f"un fichier temporaire sera créé ici : {root_path_cor}/data/temp/liste.txt\n"
                    "Ce fichier contient les index des lignes à supprimer dans X\n"
                    "Exécuter le code suivant pour transformer votre y :"
                )
                
                # print(f"{row_with_mv}")
                # Exporter la liste dans un fichier texte
                # sys.path.append("../")
                if not os.path.exists(f"{root_path}/data/temp"):
                    os.mkdir(f"{root_path}/data/temp")
                with open(f'{root_path}/data/temp/liste.txt', 'w') as file:
                    for item in row_with_mv:
                        file.write(f"{item}\n")
                
                # print(root_path_cor)
                message = f"""
                import pandas as pd
                df = pd.read_csv('{root_path_cor}/data/temp/liste.txt', header=None)
                liste = df[0].tolist()
                y = y.loc[~y.index.isin(liste)]
                """
                print(message)
                
                sys.exit()
            else:
                if os.path.exists(f"{root_path}/data/temp/liste.txt"):
                    shutil.rmtree(f"{root_path}/data/temp")
            
        return self

    def transform(self, X):
        """
        Transforme les données en supprimant les lignes contenant des valeurs manquantes.
        
        :param X: DataFrame ou array-like, les données d'entrée
        :return: DataFrame sans les valeurs manquantes
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        return X.dropna(how=self.how) #.reset_index(drop=True)

if __name__ == '__main__':
    
    import pandas as pd
    import os
    os.chdir("d:/Projet/cse")
    # Lire seulement quelques lignes pour vérifier
    data = pd.read_csv("data/raw/cs-training.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    
    # import pandas as pd
    # df = pd.read_csv('d:/Projet/cse/data/temp/liste.txt', header=None)
    # liste = df[0].tolist()
    # y = y.loc[~y.index.isin(liste)]
    # print(y)
    
    imputer = DropMvImputer(how="any")
    data_cleaned = imputer.fit_transform(X, y)
    
    print(data_cleaned)
    # import pandas as pd
    # from sklearn.pipeline import Pipeline

    # # Exemple de données
    # data = pd.DataFrame({
    #     "A": [1, 2, np.nan, 4],
    #     "B": [5, np.nan, np.nan, 8],
    #     "C": [9, 10, 11, 12]
    # })

    # print("Données avant traitement:")
    # print(data)

    # # Initialisation du transformateur
    # imputer = DropMvImputer(how="any")

    # # Transformation des données
    # data_cleaned = imputer.fit_transform(data)

    # print("\nDonnées après suppression des valeurs manquantes:")
    # print(data_cleaned)
    # y = pd.Series([0, 1, 2, 3])
    # print(y.loc[~y.index.isin([1, 2])].reset_index(drop=True))

    # Exemple d'intégration dans un pipeline sklearn
    # pipeline = Pipeline([
    #     ("drop_mv", DropMvImputer(how="any"))
    # ])

    # data_pipeline_cleaned = pipeline.fit_transform(data)
    # print("\nDonnées après passage dans le pipeline:")
    # print(data_pipeline_cleaned)
