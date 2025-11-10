import math
from typing import List

class TfidfVectorizerCustom:
    def __init__(self, count_vectorizer):
        """
        Nhận vào 1 CountVectorizer (đã có tokenizer).
        """
        self.count_vectorizer = count_vectorizer
        self.idf_values = None
        self.vocabulary_ = None

    def fit(self, documents: List[str]):
        """
        Học vocabulary và tính IDF cho từng từ trong corpus.
        """
        self.count_vectorizer.fit(documents)
        self.vocabulary_ = self.count_vectorizer.vocabulary_
        N = len(documents)

        df = [0] * len(self.vocabulary_)
        count_vectors = self.count_vectorizer.transform(documents)
        for vec in count_vectors:
            for i, count in enumerate(vec):
                if count > 0:
                    df[i] += 1

        # IDF
        self.idf_values = [math.log((N + 1) / (df_i + 1)) + 1 for df_i in df]
        return self

    def transform(self, documents: List[str]):
        """
        Biến danh sách văn bản thành ma trận TF-IDF.
        """
        count_vectors = self.count_vectorizer.transform(documents)
        tfidf_vectors = []

        for vec in count_vectors:
            tfidf_vec = [tf * self.idf_values[i] if tf > 0 else 0 for i, tf in enumerate(vec)]
            tfidf_vectors.append(tfidf_vec)

        return tfidf_vectors

    def fit_transform(self, documents: List[str]):
        """
        Fit và transform đồng thời.
        """
        self.fit(documents)
        return self.transform(documents)
