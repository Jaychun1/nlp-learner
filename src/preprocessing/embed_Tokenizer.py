
from src.core.interfaces import Tokenizer
import numpy as np

class WordEmbeddingTokenizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_document(self, document: str):
        tokens = self.tokenizer.tokenize(document)
        vectors = []
        for tok in tokens:
            if tok in self.model.key_to_index:
                vectors.append(self.model[tok])
        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)