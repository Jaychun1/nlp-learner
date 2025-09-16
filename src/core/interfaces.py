from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        """
        pass


class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]):
        """
        Tokenize a string into a list of tokens.
        """
        pass


    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms a list of documents into count vectors based on learned vocabulary.
        """
        pass

    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Convenience method: fits the corpus and returns the transformed count vectors.
        """
        pass
