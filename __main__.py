from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append("../")
from cse.src.data_preprocessing.missing_values.drop_mv_imputer import DropMvImputer
import pandas as pd
import numpy as np

import pandas as pd
import os
os.chdir("d:/Projet/cse")
# Lire seulement quelques lignes pour vérifier
data = pd.read_csv("data/raw/cs-training.csv")
X = data.iloc[:, 2:]
y = data.iloc[:, 1]

# # Définition de la pipeline
pipeline = Pipeline([
    ("missing_values", Preprocessor(imputer=DropMvImputer(how="any"))),  # Étape de preprocessing
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))  # Modèle de classification
])

# # Entraînement de la pipeline
# pipeline.fit(data, target)

# # Prédiction sur de nouvelles données
# new_data = pd.DataFrame({
#     "A": [6, None, 8],
#     "B": ["y", "z", None],
#     "C": [50.2, 60.5, 70.8]
# })

# predictions = pipeline.predict(new_data)
# print("Prédictions :", predictions)
