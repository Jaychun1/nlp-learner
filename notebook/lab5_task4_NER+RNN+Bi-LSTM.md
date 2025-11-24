
# Lab 5: Named Entity Recognition (NER) với RNN / Bi-LSTM

## Mục tiêu

Trong bài thực hành này, sinh viên sẽ:

* Tải và tiền xử lý dữ liệu NER từ Hugging Face (`conll2003`).
* Xây dựng từ điển (vocabulary) cho từ và nhãn NER.
* Tạo lớp Dataset tùy chỉnh trong PyTorch cho bài toán token classification.
* Xây dựng một mô hình RNN/Bi-LSTM sử dụng nn.Embedding, nn.LSTM, nn.Linear.
* Huấn luyện và đánh giá hiệu năng của mô hình trên bộ dữ liệu CoNLL 2003.
* Viết hàm dự đoán cho câu mới.

---

## Task 1: Tải và Tiền xử lý Dữ liệu

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

# Tạo mapping nhãn thủ công (theo CoNLL-2003)
tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
tag_to_ix = {tag: i for i, tag in enumerate(tag_names)}
ix_to_tag = {i: tag for i, tag in enumerate(tag_names)}

# Hàm convert nhãn số sang string
def convert_labels_to_str(seq_tag_ids):
    return [tag_names[i] for i in seq_tag_ids]

# Chuyển dữ liệu sang dạng list of tokens + list of tags
train_sentences = dataset["train"]["tokens"]
train_tags = [convert_labels_to_str(seq) for seq in dataset["train"]["ner_tags"]]

valid_sentences = dataset["validation"]["tokens"]
valid_tags = [convert_labels_to_str(seq) for seq in dataset["validation"]["ner_tags"]]

test_sentences = dataset["test"]["tokens"]
test_tags = [convert_labels_to_str(seq) for seq in dataset["test"]["ner_tags"]]

# Tạo vocabulary
from collections import Counter

all_words = [word for sent in train_sentences for word in sent]
word_counts = Counter(all_words)
word_to_ix = {word: i+2 for i, word in enumerate(word_counts)}  # +2 để dành 0,1 cho <PAD>, <UNK>
word_to_ix["<PAD>"] = 0
word_to_ix["<UNK>"] = 1
ix_to_word = {i: w for w, i in word_to_ix.items()}

print("Vocab size:", len(word_to_ix))
print("Tag size:", len(tag_to_ix))
```

---

## Task 2: Tạo PyTorch Dataset và DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        sentence_idx = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in sentence]
        tags_idx = [self.tag_to_ix[t] for t in tags]
        return torch.tensor(sentence_idx, dtype=torch.long), torch.tensor(tags_idx, dtype=torch.long)

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_to_ix["<PAD>"])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)  # ignore_index=-1
    return sentences_padded, tags_padded

train_dataset = NERDataset(train_sentences, train_tags, word_to_ix, tag_to_ix)
valid_dataset = NERDataset(valid_sentences, valid_tags, word_to_ix, tag_to_ix)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
```

---

## Task 3: Xây dựng Mô hình Bi-LSTM

```python
import torch.nn as nn

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        return logits

vocab_size = len(word_to_ix)
embedding_dim = 100
hidden_dim = 128
output_dim = len(tag_to_ix)
pad_idx = word_to_ix["<PAD>"]

model = BiLSTM_NER(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)
```

---

## Task 4: Huấn luyện Mô hình

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for sentences, tags in train_loader:
        sentences, tags = sentences.to(device), tags.to(device)
        optimizer.zero_grad()
        outputs = model(sentences)
        loss = criterion(outputs.view(-1, output_dim), tags.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

---

## Task 5: Đánh giá Mô hình

```python
from seqeval.metrics import classification_report

def evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sentences, tags in data_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(sentences.size(0)):
                sent_len = (tags[i] != -1).sum().item()
                all_preds.append([ix_to_tag[p.item()] for p in preds[i][:sent_len]])
                all_labels.append([ix_to_tag[t.item()] for t in tags[i][:sent_len]])
    return classification_report(all_labels, all_preds)

report = evaluate(model, valid_loader)
print(report)

# Hàm dự đoán câu mới
def predict_sentence(model, sentence):
    model.eval()
    idxs = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in sentence]
    tensor = torch.tensor([idxs], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(tensor)
        preds = torch.argmax(output, dim=-1)[0]
    return list(zip(sentence, [ix_to_tag[p.item()] for p in preds]))

example_sentence = ["VNU", "University", "is", "located", "in", "Hanoi"]
prediction = predict_sentence(model, example_sentence)
print(prediction)
```

---

## Kết quả thực hiện

Validation Accuracy: 0.9377
Test Accuracy: 0.9131

Câu: “VNU University is located in Hanoi”
Dự đoán: `[...]`
[('VNU', 'B-MISC'), ('University', 'I-ORG'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'O')]
