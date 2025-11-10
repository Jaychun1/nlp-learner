import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenizer(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words and basic punctuation.
        Example: "Hello, world!" -> ["hello", ",", "world", "!"]
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        return tokens
