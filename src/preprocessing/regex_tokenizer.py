import re
from typing import List
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenizer(self, text: str) -> List[str]:
        """
        Tokenize text using a regex.
        \w+ matches words (letters, digits, underscore)
        [^\w\s] matches punctuation
        """
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens
