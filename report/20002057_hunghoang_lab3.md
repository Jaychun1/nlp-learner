Giới thiệu bài toán

Trong bài lab này, chúng ta thực hành các kỹ thuật liên quan đến Word Embeddings, một trong những thành phần cốt lõi trong Xử lý Ngôn ngữ Tự nhiên (NLP). Word Embedding là cách biểu diễn mỗi từ trong không gian vector sao cho các quan hệ ngữ nghĩa và cú pháp của từ được bảo tồn.

Bài lab gồm hai phần chính:

Giảm chiều (Dimensionality Reduction): Sử dụng PCA và t-SNE để đưa các vector từ không gian có hàng chục hoặc hàng trăm chiều xuống còn 2 chiều.

Trực quan hóa (Visualization): Vẽ biểu đồ scatter plot để quan sát mối quan hệ giữa các từ.

### Task1 Tải Mô Hình Word Embedding

Trong bài này, chúng ta sử dụng mô hình GloVe 50 chiều (glove-wiki-gigaword-50) từ thư viện Gensim.

```python
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")
print("Số lượng từ trong vocab:", len(model.key_to_index))
print("Kích thước vector:", model.vector_size)

print(model["king"][:10])
```
Nhận xét: Mô hình chứa hơn 400.000 từ, mỗi từ được biểu diễn bằng vector 50 chiều. Embedding của "king" cho thấy một danh sách các giá trị số onehot — đây là cách mô hình mã hóa ngữ nghĩa của từ.

Số lượng từ trong vocab: 400000
Kích thước vector: 50
[ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012 ]

### Task2 Lớp WordEmbedder và Các Chức Năng Chính

Dưới đây là lớp hỗ trợ embedding tài liệu, lấy vector từ, độ tương đồng và top các từ giống nhau:

```python
class WordEmbedder:

    # load model 
    def __init__(self, model_name: str):
        try: 
            self.model: KeyedVectors = api.load(model_name)
            print('Model succesfully loaded')
        except ValueError:
            print('Cant load the model please checked')

        self.tokenizer = SimpleTokenizer()
        self.doc_embedder = WordEmbeddingTokenizer(self.model, self.tokenizer)

    def embed_document(self, document: str):
        return self.doc_embedder.embed_document(document)

    # Get embedding vector for a given word. 
    def getVector(self, word: str) -> Optional[List[float]]:
        if word in self.model.key_to_index:
            return self.model[word].tolist()
        else:
            return None

    # Return cosine similarity between two words.
    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        if word1 not in self.model.key_to_index or word2 not in self.model.key_to_index:
            print(f"One or both words are not in vocabulary: '{word1}', '{word2}'")
            return None
        return self.model.similarity(word1, word2)
    
    # Return top 10 similar words to the given word.
    def get_most_similar(self, word: str, top_n: int = 10) -> Optional[List[Tuple[str, float]]]:
        if word not in self.model.key_to_index:
            print(f" Word '{word}' not in vocabulary.")
            return None
        return self.model.most_similar(word, topn=top_n)
```

file src/representations/word_embedder.py chứa lớp WordEmbedder có nhiệm vụ tải mô hình và cung cấp các hàm thao tác vector. Constructor nhận tên mô hình và load bằng gensim.downloader.load(model_name).

Các chức năng chính: 
#### Lấy vector của từ

Phương thức: get_vector(self, word: str) Nếu từ tồn tại trong từ vựng của mô hình thì trả về vector embedding

Nếu từ OOV (không có trong vocab) → xử lý ngoại lệ và trả về None hoặc vector rỗng tùy bạn cài đặt

#### Tính độ tương đồng giữa hai từ

Phương thức: get_similarity(self, word1: str, word2: str)

Sử dụng cosine similarity giữa hai vector để tính độ tương đồng 
Nếu một trong hai từ là OOV trả về None
Nếu hợp lệ trả về điểm similarity theo thang 0–1

#### Tìm các từ giống nhất

Phương thức: get_most_similar(self, word: str, top_n: int = 10)

Dùng hàm model.most_similar() của gensim
Kết quả trả về danh sách các cặp (từ, độ tương đồng)

### Task 3 Document Embedding
Mục tiêu: Biểu diễn một đoạn văn bằng cách trung bình vector của các từ xuất hiện trong đoạn đó.

Các bước thực hiện
Tokenize văn bản

Sử dụng Tokenizer từ Lab 1 để tách văn bản thành danh sách token
#### Lấy embedding từng token

Với mỗi token lấy vector bằng get_vector(Nếu token OOV bỏ qua)

#### Tính vector biểu diễn tài liệu

Nếu tài liệu không chứa token hợp lệ trả về vector zero có kích thước đúng bằng dimension của mô hình
Ngược lại tính trung bình element-wise các vector từ phương thức:
```python

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
        
``` 

