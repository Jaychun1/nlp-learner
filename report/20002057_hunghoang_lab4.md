### Task 1 — Data Preparation (with Scikit-learn)

Trong bài lab này, chúng ta xây dựng một hệ thống phân loại cảm xúc (sentiment classification) dựa trên văn bản. Nhiệm vụ của bài toán là xác định xem một câu đánh giá phim mang cảm xúc tích cực (1) hay tiêu cực (0).

Để làm rõ quy trình xử lý văn bản trong học máy, bài lab sử dụng một tập dữ liệu nhỏ do chính sinh viên xây dựng, gồm 6 câu mô tả cảm xúc về phim:

Các câu như “This movie is fantastic and I love it!” mang cảm xúc positive
Các câu như “I hate this film, it's terrible.” mang cảm xúc negative
Tuy dataset nhỏ, nhưng nó đủ để minh họa toàn bộ pipeline xử lý văn bản.

#### dataset

```python
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

```

#### Custom Tokenizer + Vectorizer

Ở đây chúng ta sử dụng lại các hàm đã được xây dưng từ các bài lab trước nằm trong đường dẫn
src.preprocessing.

```python
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.representations.tfidf_vectorizer import TfidfVectorizerCustom

tokenizer = SimpleTokenizer()
count_vectorizer = CountVectorizer(tokenizer=tokenizer)
tfidf_vectorizer = TfidfVectorizerCustom(count_vectorizer)
tfidfMatrix = tfidf_vectorizer.fit_transform(texts)

```

#### Result TF-IDF Matrix

```python
import pandas as pd

vocab = tfidf_vectorizer.count_vectorizer.vocabulary_
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
columns = [word for word, _ in sorted_vocab]
df = pd.DataFrame(tfidfMatrix, columns=columns)
print(df)

```

Tính toán ma trận tìm các giá trị thể hiện từ thuộc nhóm nào
a absolutely acting and bad boring could experience \
0 0.000000 0.000000 0.000000 2.252763 0.0 0.000000 0.0 0.000000  
1 0.000000 0.000000 0.000000 0.000000 0.0 0.000000 0.0 0.000000  
2 1.559616 0.000000 2.252763 0.000000 0.0 0.000000 0.0 2.252763  
3 1.559616 2.252763 0.000000 0.000000 0.0 2.252763 0.0 0.000000  
4 1.559616 0.000000 0.000000 0.000000 0.0 0.000000 0.0 0.000000

fantastic film ... superb terrible the this time \
0 2.252763 0.000000 ... 0.000000 0.000000 0.000000 1.559616 0.000000  
1 0.000000 2.252763 ... 0.000000 2.252763 0.000000 1.559616 0.000000  
2 0.000000 0.000000 ... 2.252763 0.000000 2.252763 0.000000 0.000000  
3 0.000000 0.000000 ... 0.000000 0.000000 0.000000 0.000000 2.252763  
4 0.000000 0.000000 ... 0.000000 0.000000 0.000000 1.559616 0.000000

      truly       was     waste  watching      what

0 0.000000 0.000000 0.000000 0.0 0.000000  
1 0.000000 0.000000 0.000000 0.0 0.000000  
2 2.252763 2.252763 0.000000 0.0 0.000000  
3 0.000000 0.000000 2.252763 0.0 2.252763  
4 0.000000 0.000000 0.000000 0.0 0.000000

#### Using scikit-learn TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(texts)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df)

```

absolutely acting bad boring experience fantastic film \
0 0.0 0.000000 0.00000 0.0 0.000000 0.57735 0.00000  
1 0.0 0.000000 0.00000 0.0 0.000000 0.00000 0.57735  
2 0.0 0.447214 0.00000 0.0 0.447214 0.00000 0.00000  
3 0.5 0.000000 0.00000 0.5 0.000000 0.00000 0.00000  
4 0.0 0.000000 0.00000 0.0 0.000000 0.00000 0.00000  
5 0.0 0.000000 0.57735 0.0 0.000000 0.00000 0.00000

    finish     great     hate  ...     love  masterpiece    movie  recommend  \

0 0.00000 0.000000 0.00000 ... 0.57735 0.00000 0.57735 0.00000  
1 0.00000 0.000000 0.57735 ... 0.00000 0.00000 0.00000 0.00000  
2 0.00000 0.447214 0.00000 ... 0.00000 0.00000 0.00000 0.00000  
3 0.00000 0.000000 0.00000 ... 0.00000 0.00000 0.00000 0.00000  
4 0.00000 0.000000 0.00000 ... 0.00000 0.57735 0.00000 0.57735  
5 0.57735 0.000000 0.00000 ... 0.00000 0.00000 0.00000 0.00000

     superb  terrible  time     truly  waste  watching

0 0.000000 0.00000 0.0 0.000000 0.0 0.00000  
1 0.000000 0.57735 0.0 0.000000 0.0 0.00000  
2 0.447214 0.00000 0.0 0.447214 0.0 0.00000  
3 0.000000 0.00000 0.5 0.000000 0.5 0.00000  
4 0.000000 0.00000 0.0 0.000000 0.0 0.00000  
5 0.000000 0.00000 0.0 0.000000 0.0 0.57735

Giá trị giữa tự build thông qua các hàm khác xa nhiều so với sklearn do sklearn đã chuẩn hoá còn tự build thì chưa nên có sự khác biệt trên với sklearn loại bỏ các từ ít ảnh hưởng như a, ... trong khi tự build thông qua chuẩn hoá bản thân không loại bỏ dẫn tới sự sai khác trên

### Task 2 — TextClassifier Implementation

#### Implementation file: src/core/text_classifier.py

```python
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    def __init__(self, vectorizer):
        """
        Khởi tạo TextClassifier với vectorizer (CountVectorizer hoặc TfidfVectorizer).
        """
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện mô hình Logistic Regression trên dữ liệu văn bản.
        """
        X = self.vectorizer.fit_transform(texts)
        self._model = LogisticRegression(solver='liblinear', random_state=42)
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho văn bản mới.
        """
        if self._model is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        X = self.vectorizer.transform(texts)
        return self._model.predict(X).tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính các metric đánh giá: accuracy, precision, recall, F1.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
```

#### Training & Prediction

```python
from src.core.text_classifier import TextClassifier

classifier = TextClassifier(vectorizer)
classifier.fit(train_texts, train_labels)
preds = classifier.predict(texts)
metrics = classifier.evaluate(labels, preds)

print("Predictions:", preds)
print("Metrics:", metrics)

```

Trong đó các hàm

Huấn luyện mô hình (fit)

Mục đích: học quan hệ giữa dữ liệu đầu vào (văn bản đã được vector hóa) và nhãn (labels).

Quá trình bao gồm:
Biến đổi văn bản thành ma trận đặc trưng số dựa trên vectorizer.
Xây dựng mô hình học máy (Logistic Regression) và tìm các trọng số tối ưu để phân loại nhãn dựa trên các đặc trưng đầu vào.
Kết quả: mô hình đã học xong và sẵn sàng để dự đoán dữ liệu mới.

Dự đoán nhãn (predict)
Mục đích: dự đoán nhãn của các văn bản chưa được biết nhãn dựa trên mô hình đã huấn luyện.
Quá trình:
Chuyển đổi văn bản mới sang ma trận đặc trưng bằng vectorizer đã fit trước đó.
Sử dụng mô hình Logistic Regression đã huấn luyện để đưa ra nhãn dự đoán.
Kết quả: danh sách nhãn dự đoán cho các văn bản mới.

Đánh giá mô hình (evaluate)
Mục đích: đo lường chất lượng phân loại của mô hình dựa trên các nhãn dự đoán so với nhãn thực tế.
Các chỉ số đánh giá thường bao gồm:
Accuracy: tỉ lệ dự đoán đúng trên tổng số mẫu.
Precision: tỉ lệ dự đoán đúng trên tổng số mẫu được dự đoán là positive.
Recall: tỉ lệ mẫu positive thật sự được dự đoán đúng.
F1-score: trung bình điều hòa giữa precision và recall, đánh giá cân bằng giữa độ chính xác và khả năng phát hiện.


### Task 3 — Evaluation

```python
Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
```
Chia bộ dữ liệu thành 80 - 20 để thực hiện dự đoán trên tập test

Using RegexTokenizer + Custom TF-IDF
```python
from src.preprocessing.regex_tokenizer import RegexTokenizer
tokenizer = RegexTokenizer()
count_vectorizer = CountVectorizer(tokenizer=tokenizer)
tfidf_vectorizer = TfidfVectorizerCustom(count_vectorizer)

Training
classifier = TextClassifier(vectorizer)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

Evaluation
from sklearn.metrics import classification_report, accuracy_score

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
```

Classification Report:
               precision    recall  f1-score   support

           0       0.50      1.00      0.67         1
           1       0.00      0.00      0.00         1

    accuracy                           0.50         2
   macro avg       0.25      0.50      0.33         2
weighted avg       0.25      0.50      0.33         2

### Advanced Example: Sentiment Analysis with PySpark

Trong bài này, chúng ta xây dựng hệ thống phân tích cảm xúc (Sentiment Analysis) với PySpark dựa trên mô hình Logistic Regression và sau đó mở rộng với nhiều mô hình nâng cao như Naive Bayes, Gradient Boosted Trees (GBT), Multilayer Perceptron (MLP), và Word2Vec.
PySpark cho phép xử lý dữ liệu lớn và xây dựng pipeline ML phân tán hiệu quả.
#### Chuẩn bị dữ liệu
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
data_path = "sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Convert -1/1 labels to 0/1: Normalize sentiment labels
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

# Drop rows with null sentiment
df = df.dropna(subset=["sentiment"])

```

Giải thích

Dataset gồm 2 cột: text (nội dung) và sentiment (-1: negative, 1: positive)

Chuyển label từ -1/1 → 0/1 để phù hợp với Spark ML

Loại bỏ dòng chứa giá trị null

#### Pipeline tiền xử lý cơ bản (Tokenizer → StopWords → TF → IDF)
```python
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
```

#### Xây dựng mô hình hồi quy logistic 
```python
rom pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    maxIter=10, 
    regParam=0.001, 
    featuresCol="features", 
    labelCol="label"
)

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

# train - test data 80 20
trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(trainingData)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
```
Accuracy: 0.7294860234445446


```python
# Improve More Model Performance
from typing import List
from notebook.newTokenizer import newTokenizer
from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf

## chuẩn hoá hàm tokenizer với thêm các điều kiện 
class newTokenizer:
    def tokenizer(self, text: str):
        if text is None:
            return []
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

tokenizer = newTokenizer()
tokenizer_udf = udf(tokenizer.tokenizer, ArrayType(StringType()))
df = df.withColumn("filtered_words", tokenizer_udf(col("text")))

stopwords_remover = StopWordsRemover(inputCol="filtered_words", outputCol="final_words")
df = stopwords_remover.transform(df)


trainData, testData = df.randomSplit([0.8, 0.2], seed=42)

# TF-IDF
hashingTF = HashingTF(inputCol="final_words", outputCol="raw_features", numFeatures=5000)
idf = IDF(inputCol="raw_features", outputCol="features_tfidf")

# Word2Vec
word2vec = Word2Vec(inputCol="final_words", outputCol="features_w2v", vectorSize=100)

feature_pipelines = {
    "TF-IDF": ["final_words", hashingTF, idf, "features_tfidf"],
    "Word2Vec": ["final_words", word2vec, None, "features_w2v"]  # IDF not needed for W2V
}

# Model run

models = {
    "LogisticRegression": LogisticRegression(maxIter=10, regParam=0.001),
    "NaiveBayes": NaiveBayes(),
    "GBT": GBTClassifier(maxIter=20),
    "MLP": MultilayerPerceptronClassifier(layers=[5000, 100, 2], maxIter=100)  # only TF-IDF input
}

results = {}
```
#### Task 4: Evaluating and Improving Model Performance
```python
for feat_name, (input_col, feat_stage1, feat_stage2, features_col) in feature_pipelines.items():
    for model_name, model in models.items():
        # Skip incompatible combinations
        if feat_name == "Word2Vec" and model_name in ["NaiveBayes", "GBT", "MLP"]:
            continue  

        model.setParams(featuresCol=features_col, labelCol="label")

        # Build pipeline
        stages = [feat_stage1]
        if feat_stage2:
            stages.append(feat_stage2)
        stages.append(model)

        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(trainData)
        predictions = pipeline_model.transform(testData)

        # Evaluate
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        acc = evaluator.evaluate(predictions)

        results[f"{feat_name} + {model_name}"] = acc

# Print results
for k, v in results.items():
    print(f"{k} Accuracy: {v:.4f}")

```
Result:
TF-IDF + LogisticRegression Accuracy: 0.7096
TF-IDF + NaiveBayes Accuracy: 0.6916
TF-IDF + GBT Accuracy: 0.7457
TF-IDF + MLP Accuracy: 0.7547
Word2Vec + LogisticRegression Accuracy: 0.6664
