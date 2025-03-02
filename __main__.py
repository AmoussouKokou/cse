from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from cse.src.data_preprocessing.preprocessor import Preprocessor
from cse.src.data_preprocessing.drop_mv_imputer import DropMvImputer
import pandas as pd
import numpy as np

# Création d'un DataFrame d'exemple
data = pd.DataFrame({
    "A": [1, 2, None, 4, 5],
    "B": [None, "x", "y", "z", "w"],
    "C": [10.5, 20.1, 30.2, 40.8, None]
})

target = pd.Series([0, 1, 0, 1, 1])

# Définition de la pipeline
pipeline = Pipeline([
    ("preprocessor", Preprocessor(imputer=DropMvImputer(how="any"))),  # Étape de preprocessing
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))  # Modèle de classification
])

# Entraînement de la pipeline
pipeline.fit(data, target)

# Prédiction sur de nouvelles données
new_data = pd.DataFrame({
    "A": [6, None, 8],
    "B": ["y", "z", None],
    "C": [50.2, 60.5, 70.8]
})

predictions = pipeline.predict(new_data)
print("Prédictions :", predictions)
