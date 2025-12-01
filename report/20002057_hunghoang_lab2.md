## Task 1: interface

Trong src/core/interfaces.py, ta định nghĩa một interface mới tên là Vectorizer, đóng vai trò là lớp cơ sở (abstract base class) cho mọi bộ vector hóa văn bản.

Interface này gồm 3 phương thức:

fit(corpus) Phân chia một chuỗi thành một danh sách các token
Thu thập tất cả các từ xuất hiện và tạo mapping từ index.

transform(documents)

Biến documents thành một vector đếm số lần xuất hiện của từng từ trong vocabulary đã học.
fit_transform(corpus) Tiện ích giúp chạy fit() và sau đó transform() chỉ bằng một câu lệnh.

```python
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

```

## Task 2

Mục tiêu của là biểu diễn các tài liệu văn bản dưới dạng vector số bằng mô hình Bag-of-Words (BoW).
Để làm được điều đó, ta xây dựng một CountVectorizer có khả năng:

Học vocabulary từ tập văn bản

Biến đổi mỗi tài liệu thành document-term matrix

Tái sử dụng Tokenizer từ Lab 1

Đây là bước quan trọng trước khi đưa văn bản vào các mô hình học máy.

CountVectorizer Implementation
File: src/representations/count_vectorizer.py

CountVectorizer kế thừa từ Vectorizer và nhận vào một Tokenizer (SimpleTokenizer hoặc RegexTokenizer).

``` python
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

```