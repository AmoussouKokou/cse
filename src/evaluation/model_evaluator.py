import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    roc_curve, auc, classification_report
)

class PipelineEvaluator(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        metrics = ["f1-score", "precision", "recall", "auc", "accuracy"],
        scoring="f1-score"
    ):
        self.metrics = metrics
        self.scoring = scoring

    def fit(self, X, y=None):
        return self

