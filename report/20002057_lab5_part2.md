
# Lab: Text Classification – HWU Dataset

## PHẦN 2: LAB THỰC HÀNH

### hiết lập Môi trường và Tải Dữ liệu

```bash
# Giải nén dữ liệu
!tar -xzvf data/hwu.tar.gz
```

```python
import pandas as pd

# Đọc dữ liệu train/val/test
import pandas as pd

# Đọc dữ liệu train/val/test
df_train = pd.read_csv('../hwu/train_10.csv', sep='\t', header=None, names=['text', 'intent'])
df_val = pd.read_csv('../hwu/val.csv', sep='\t', header=None, names=['text', 'intent'])
df_test = pd.read_csv('../hwu/test.csv', sep='\t', header=None, names=['text', 'intent'])

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)

df_train.head()
```

<!-- # Train shape: (641, 2)
Validation shape: (1077, 2)
Test shape: (1077, 2)
text	intent
0	text,"category"	NaN
1	remind me about my alarms today,"alarm_query"	NaN
2	list my different alarm,"alarm_query"	NaN
3	what alarms are set,"alarm_query"	NaN
4	list alarms,"alarm_query"	NaN -->


```python
from sklearn.preprocessing import LabelEncoder

# Chuyển intent sang dạng số
label_encoder = LabelEncoder()
label_encoder.fit(df_train['intent'])

y_train = label_encoder.transform(df_train['intent'])
y_val = label_encoder.transform(df_val['intent'])
y_test = label_encoder.transform(df_test['intent'])
```

---

## Task 1: Pipeline TF-IDF + Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 1. Tạo pipeline
tfidf_lr_pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    LogisticRegression(max_iter=1000)
)

# 2. Huấn luyện trên tập train
tfidf_lr_pipeline.fit(df_train['text'], y_train)

# 3. Đánh giá trên tập test
y_pred = tfidf_lr_pipeline.predict(df_test['text'])
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

---

## Task 2: Word2Vec (Trung bình) + Dense Layer

```python
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Huấn luyện Word2Vec
sentences = [text.split() for text in df_train['text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 2. Hàm chuyển câu thành vector trung bình
def sentence_to_avg_vector(text, model):
    words = text.split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

# 3. Chuyển dữ liệu sang dạng vector trung bình
X_train_avg = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_train['text']])
X_val_avg = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_val['text']])
X_test_avg = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_test['text']])

# 4. Xây dựng mô hình Dense
num_classes = len(label_encoder.classes_)
model_avg_dense = Sequential([
    Dense(128, activation='relu', input_shape=(w2v_model.vector_size,)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 5. Compile và huấn luyện
model_avg_dense.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_avg_dense.fit(X_train_avg, y_train, validation_data=(X_val_avg, y_val), epochs=10, batch_size=32)
```

---

## Task 3: Embedding Pre-trained + LSTM

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM

# 1. Tokenizer và padding
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(df_train['text'])

max_len = 50
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(df_train['text']), maxlen=max_len, padding='post')
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(df_val['text']), maxlen=max_len, padding='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(df_test['text']), maxlen=max_len, padding='post')

# 2. Tạo ma trận trọng số từ Word2Vec
embedding_dim = w2v_model.vector_size
embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i <= vocab_size and word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# 3. Xây dựng mô hình LSTM với Embedding pre-trained
lstm_model_pretrained = Sequential([
    Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])

# 4. Compile và huấn luyện
lstm_model_pretrained.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Task 4: Embedding học từ đầu + LSTM

```python
# 1. Xây dựng mô hình LSTM với Embedding học từ đầu
lstm_model_scratch = Sequential([
    Embedding(input_dim=vocab_size+1, output_dim=100, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])

# 2. Compile và huấn luyện
lstm_model_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## Nhiệm vụ 5: Đánh giá, So sánh và Phân tích

* Tạo bảng so sánh định lượng:

                         Pipeline  F1-score (Macro) Test Loss
0    TF-IDF + Logistic Regression          0.642381       N/A
1          Word2Vec (Avg) + Dense          0.022756  4.152387
2  Embedding (Pre-trained) + LSTM          0.004813  4.158883
3      Embedding (Scratch) + LSTM          0.000542  4.159305

* Phân tích định tính: Chọn các câu khó, đặc biệt câu có phủ định hoặc cấu trúc phức tạp, ví dụ:

  * "can you remind me to not call my mom" → reminder_create
  * "is it going to be sunny or rainy tomorrow" → weather_query
  * "find a flight from new york to london but not through paris" → flight_search

* So sánh dự đoán từ 4 mô hình, nhận xét ưu nhược điểm và khả năng xử lý chuỗi của LSTM.

