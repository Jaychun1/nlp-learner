from typing import List, Dict
from src.core.interfaces import Vectorizer, Tokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, corpus: List[str]):
        """
        Học vocabulary từ toàn bộ corpus.
        """
        unique_tokens = set()

        for doc in corpus:
            tokens = self.tokenizer.tokenizer(doc)
            unique_tokens.update(tokens)

        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        return self

    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Biến danh sách văn bản thành ma trận đếm (count matrix).
        """
        if not self.vocabulary_:
            raise ValueError("Vectorizer chưa được fit. Hãy gọi fit() trước.")

        vectors = []
        vocab_size = len(self.vocabulary_)

        for doc in documents:
            vector = [0] * vocab_size
            tokens = self.tokenizer.tokenizer(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1
            vectors.append(vector)

        return vectors

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        self.fit(corpus)
        return self.transform(corpus)
