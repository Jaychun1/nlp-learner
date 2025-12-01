## Task 1: Simple Tokenizer

Mục tiêu của bài

Mục tiêu của Task 1 là xây dựng một Simple Tokenizer – bộ tách từ cơ bản dùng trong xử lý ngôn ngữ tự nhiên (NLP). Tokenizer này sẽ chịu trách nhiệm biến đổi một chuỗi văn bản thô thành một danh sách các token (từ và dấu câu).

Thiết kế Interface: Tokenizer

Trong file src/core/interfaces.py, ta định nghĩa một abstract base class (ABC) tên là Tokenizer.
Tạo chuẩn chung để mọi tokenizer khác (SimpleTokenizer, WordPieceTokenizer, BPE tokenizer...) đều phải tuân theo cấu trúc giống nhau.
Giúp code dễ mở rộng và bảo trì.
Cho phép hoán đổi tokenizer mà không sửa logic nơi khác.

Nội dung Interface
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
@abstractmethod
def tokenizer(self, text: str) -> List[str]:
"""
Tokenize a string into a list of tokens.
"""
pass

Phương thức tokenize là abstract nên lớp con bắt buộc phải triển khai.

Cài đặt Simple Tokenizer

File: src/preprocessing/simple_tokenizer.py
Lớp SimpleTokenizer kế thừa từ Tokenizer và triển khai phương thức tokenize.

Yêu cầu của SimpleTokenizer

Chuyển văn bản thành chữ thường (lowercase)
Tách token bằng khoảng trắng và tách dấu câu ra khỏi từ

Ví dụ:
"Hello, world!" thành ["hello", ",", "world", "!"]

Cách thực hiện:
Tách các chuỗi ký tự chữ (\w+) hoặc các dấu câu [.,?!]

```python
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
```

## Task 2: Regex-based Tokenizer (Bonus)

Mục tiêu của bài Task 2 mở rộng hệ thống tokenizer bằng cách xây dựng một Regex-based Tokenizer — một tokenizer linh hoạt và mạnh hơn SimpleTokenizer.

Mục tiêu chính:
Sử dụng một biểu thức chính quy duy nhất (regex) để tách token.
Xử lý tốt hơn các trường hợp phức tạp như dấu câu liên tiếp (...), contraction (isn't, let's), và ký tự đặc biệt.

Cài đặt RegexTokenizer
File: src/preprocessing/regex_tokenizer.py
Ý tưởng chính
Thay vì chia tách bằng khoảng trắng hoặc tách dấu câu thủ công, ta dùng một regex tổng quát: \w+ | [^\w\s]

Giải thích:

\w+ bắt các nhóm ký tự chữ/số (word token) [^\w\s] bắt mọi ký tự không phải chữ, số hoặc whitespace, tức là dấu câu, ký tự đặc biệt
Điều này giúp tokenizer hoạt động chính xác hơn SimpleTokenizer. Interface được sử dụng qua interface Tokenizer đã được viết ở lab 1

Code python

```python

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
```

Hàm run test cho bài toán src/testing/test_tokenizers.py

```python
def test_tokenizers():
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    for sentence in sentences:
        print("Original:", sentence)
        print("SimpleTokenizer:", simple_tokenizer.tokenize(sentence))
        print("RegexTokenizer: ", regex_tokenizer.tokenize(sentence))
        print("-" * 50)

```

## Task 3: Mở rộng test for dataset en_ewt-ud-train.txt

```python
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

def test_tokenizers_on_ud_dataset():
    dataset_path = "/Users/mac/Desktop/NLP/nlp-learner/UD_English-EWT/en_ewt-ud-train.txt"

    raw_text = load_raw_text_data(dataset_path)

    sample_text = raw_text[:500]
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    regex_tokens = regex_tokenizer.tokenize(sample_text)

    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

if __name__ == "__main__":
    test_tokenizers_on_ud_dataset()
```

### Kết luận

RegexTokenizer chính xác hơn và linh hoạt hơn, đặc biệt với văn bản tự nhiên tiếng Anh. Ngoài ra Task 2 giúp mở rộng hệ thống NLP với một tokenizer mạnh hơn.
Sau task này, ta đã có:

Tokenizer interface
SimpleTokenizer
RegexTokenizer
các file test để kiểm thử

Tokenizer dựa trên regex là lựa chọn tốt cho các văn bản tiếng Anh tự nhiên có dấu câu phức tạp và ký tự đặc biệt
