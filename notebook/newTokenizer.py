import re
from typing import List
from src.core.interfaces import Tokenizer

class newTokenizer(Tokenizer):
    def tokenizer(self, text: str) -> List[str]:
        """
        Clean and tokenize text using regex.
        """
        if text is None:
            return []
        # Clean text
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
        text = re.sub(r"<.*?>", "", text)           # remove HTML tags
        text = re.sub(r"[^a-z0-9\s]", "", text)     # remove special characters

        tokens = text.split()
        return tokens
