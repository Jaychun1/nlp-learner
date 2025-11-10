from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    def __init__(self, vectorizer):
        """
        Khởi tạo TextClassifier với vectorizer (CountVectorizer hoặc TfidfVectorizer).
        """
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện mô hình Logistic Regression trên dữ liệu văn bản.
        """
        X = self.vectorizer.fit_transform(texts)
        self._model = LogisticRegression(solver='liblinear', random_state=42)
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho văn bản mới.
        """
        if self._model is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        X = self.vectorizer.transform(texts)
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính các metric đánh giá: accuracy, precision, recall, F1.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
