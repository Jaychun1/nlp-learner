from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)

        # Initialize and train Logistic Regression
        self._model = LogisticRegression(solver='liblinear', random_state=42)
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Transform texts into feature matrix
        X = self.vectorizer.transform(texts)
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }
