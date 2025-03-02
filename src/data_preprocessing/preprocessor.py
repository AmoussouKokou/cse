from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

import sys
sys.path.append("../")

from cse.src.data_preprocessing.drop_mv_imputer import DropMvImputer
from cse.src.data_preprocessing.id_scaler import IdScaler
from cse.src.data_preprocessing.id_encoder import IdEncoder

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, imputer=None, scaler=None, encoder=None):
        """
        Initialise la classe Preprocessor avec des transformateurs spécifiques.
        
        :param imputer: Transformateur pour gérer les valeurs manquantes (par défaut DropMvImputer)
        :param scaler: Transformateur pour normaliser les colonnes numériques (par défaut StandardScaler)
        :param encoder: Transformateur pour encoder les variables catégorielles (par défaut LabelEncoder)
        """
        self.imputer = imputer if imputer else DropMvImputer()
        self.scaler = scaler if scaler else IdScaler()
        self.encoder = encoder if encoder else IdEncoder()
        self.num_columns = []
        self.cat_columns = []
    
    def fit(self, X, y=None):
        """
        Apprend les paramètres de transformation sur les données d'entraînement.
        
        :param X: DataFrame d'entrée
        :param y: Variable cible (non utilisée ici)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un DataFrame pandas")
        
        self.num_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.cat_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Apprentissage des transformateurs
        self.imputer.fit(X)
        if self.num_columns:
            self.scaler.fit(X[self.num_columns])
        return self
    
    def transform(self, X):
        """
        Transforme les données selon les transformations apprises.
        
        :param X: DataFrame d'entrée
        :return: DataFrame transformé
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un DataFrame pandas")
        
        X_copy = X.copy()
        
        # Gestion des valeurs manquantes
        X_copy = self.imputer.transform(X_copy)
        
        # Encodage des variables catégorielles
        for col in self.cat_columns:
            X_copy[col] = self.encoder.fit_transform(X_copy[col].astype(str))
        
        # Normalisation des variables numériques
        if self.num_columns:
            X_copy[self.num_columns] = self.scaler.transform(X_copy[self.num_columns])
        
        # Application des transformations personnalisées
        X_copy = self.apply_custom_transformations(X_copy)
        
        return X_copy
    
    def apply_custom_transformations(self, X):
        """
        Applique des transformations spécifiques aux données, comme supprimer la première colonne.
        
        :param X: DataFrame d'entrée
        :return: DataFrame transformé
        """
        if not X.empty:
            X = X.iloc[:, 1:]  # Supprime la première colonne
            
        return X

# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    import os
    os.chdir("d:/Projet/cse")
    # Lire seulement quelques lignes pour vérifier
    data = pd.read_csv("data/raw/cs-training.csv")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    print(X.shape)
    print(y)
    
    preprocessor = Preprocessor()
    preprocessor.fit(X)
    transformed_data = preprocessor.transform(X)
    print(transformed_data.shape)
    print(transformed_data)
