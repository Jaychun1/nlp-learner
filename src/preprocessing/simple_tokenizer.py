import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words and basic punctuation.
        Example: "Hello, world!" -> ["hello", ",", "world", "!"]
        """
        text = text.lower()
        tokens = re.findall(r'\w+|[.,!?]', text)

        return tokens
