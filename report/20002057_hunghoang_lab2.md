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

Lớp CountVectorizer kế thừa từ interface Vectorizer và nhận vào một Tokenizer.
Điều này giúp Vectorizer linh hoạt, có thể dùng với SimpleTokenizer hoặc RegexTokenizer.

### Thuộc tính chính

tokenizer: một đối tượng tokenizer dùng để tách token từ văn bản.

vocabulary_: dictionary ánh xạ từ token sang chỉ số trong vector, được tạo ra sau khi học corpus.
Ví dụ:

{"ai": 0, "i": 1, "learning": 2, "love": 3, "machine": 4}

### Phương thức fit()

fit(corpus) thực hiện các bước sau:

Khởi tạo một tập hợp (set) để lưu tất cả token duy nhất trong corpus.

Duyệt qua từng tài liệu, sử dụng tokenizer để tách token và thêm vào tập hợp.

Sắp xếp các token và gán mỗi token một chỉ số để tạo dictionary vocabulary_.

Ví dụ, với corpus:

["I love AI.", "Love machine learning."]


Sau khi fit, vocabulary_ có thể là:

{"ai": 0, "i": 1, "learning": 2, "love": 3, "machine": 4}

### Phương thức transform()

transform(documents) chuyển danh sách văn bản thành document-term matrix:

Với mỗi tài liệu, tạo một vector toàn số 0, có độ dài bằng số từ trong vocabulary_.

Tokenize tài liệu.

Với mỗi token xuất hiện trong vocabulary, tăng giá trị đếm tại vị trí tương ứng trong vector.

Trả về danh sách các vector cho toàn bộ tài liệu.

Ví dụ:
Document: "Love AI AI"
Vector: [2, 0, 0, 1, 0] (theo thứ tự token đã học ở ví dụ trên)

### Phương thức fit_transform()

Phương thức fit_transform(corpus) là một tiện ích kết hợp fit() và transform() để tiết kiệm thao tác.
Nó học vocabulary từ corpus rồi ngay lập tức trả về ma trận đếm.

File test kết quả nằm ở vị trí src/testing/lab2_test.oy

```python

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

def run_lab2_test():
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    dt_matrix = vectorizer.fit_transform(corpus)

    print("Vocabulary:", vectorizer.vocabulary_)
    print("Document-Term Matrix:")
    for row in dt_matrix:
        print(row)

if __name__ == "__main__":
    run_lab2_test()

```

### Đánh giá kết quả 

Trong quá trình thử nghiệm, chúng tôi sử dụng RegexTokenizer và một corpus mẫu:

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]


Kết quả:

Vocabulary (ví dụ):

{"ai":0, "i":1, "is":2, "love":3, "nlp":4, "programming":5, "subfield":6, "of":7, ".":8}


Document-term matrix (ma trận đếm):

[
 [0, 1, 0, 1, 1, 0, 0, 0, 1],   # "I love NLP."
 [0, 1, 0, 1, 0, 1, 0, 0, 1],   # "I love programming."
 [1, 0, 1, 0, 1, 0, 1, 1, 1]    # "NLP is a subfield of AI."
]


Kết quả cho thấy CountVectorizer hoạt động chính xác, chuyển văn bản thành vector đếm theo đúng cơ chế Bag-of-Words.

Kêt luận

Lab 2 đã hoàn thành các mục tiêu đề ra:

Xây dựng Vectorizer interface chuẩn hóa quá trình vector hóa văn bản.

Cài đặt CountVectorizer dựa trên tokenizer của Lab 1.

Biểu diễn văn bản thành document-term matrix và học vocabulary.

CountVectorizer là bước quan trọng để chuẩn bị dữ liệu cho các mô hình học máy truyền thống hoặc các bước xử lý nâng cao như TF-IDF.