from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        """
        pass