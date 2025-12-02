# Lab: POS  với PyTorch – Universal Dependencies (UD_English-EWT)

## . Mô tả bài lab

Trong bài lab này, chúng ta sẽ thực hiện bài toán **POS tagging** trên dữ liệu **Universal Dependencies (UD_English-EWT)** bằng PyTorch. Dữ liệu có định dạng **CoNLL-U**, mỗi câu gồm các token với các cột thông tin. Chúng ta chỉ quan tâm đến hai cột:

* **Cột 2 (FORM)**: Từ gốc
* **Cột 4 (UPOS)**: Nhãn Part-of-Speech theo chuẩn Universal

Ví dụ một câu:

```
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0003
# text = From the AP comes this story:
1 From from ADP IN _ 3 case _ _
2 the the DET DT Definite=Def|PronType=Art 3 det _ _
3 AP AP PROPN NNP Number=Sing 4 nsubj _ _
4 comes come VERB VBZ Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin 0 root _ _
5 this this DET DT Number=Sing|PronType=Dem 6 det _ _
6 story story NOUN NN Number=Sing 4 obj _ _
7 : : PUNCT : _ 4 punct _ _
```

## Task 1: Tải và tiền xử lý dữ liệu

### 1.1. Đọc file `.conllu`

```python
def load_conllu(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            elif line.startswith("#"):
                continue
            else:
                parts = line.split('\t')
                if len(parts) > 4:
                    word = parts[1]
                    upos = parts[3]
                    sentence.append((word, upos))
        if sentence:
            sentences.append(sentence)
    return sentences

train_sentences = load_conllu("data/UD_English-EWT/en_ewt-ud-train.conllu")
dev_sentences = load_conllu("data/UD_English-EWT/en_ewt-ud-dev.conllu")

print(f"Số câu train: {len(train_sentences)}, số câu dev: {len(dev_sentences)}")
```

### 2.2. Xây dựng Vocabulary

```python
from collections import Counter

# Từ điển từ
word_counter = Counter(word for sent in train_sentences for word, _ in sent)
word_to_ix = {word: i for i, word in enumerate(word_counter.keys(), start=0)}
word_to_ix["<UNK>"] = len(word_to_ix)

# Từ điển nhãn
tags = set(tag for sent in train_sentences for _, tag in sent)
tag_to_ix = {tag: i for i, tag in enumerate(tags)}

print(f"Số từ trong từ điển: {len(word_to_ix)}, số nhãn POS: {len(tag_to_ix)}")
```

## Task 2: Tạo PyTorch Dataset và DataLoader

### 2.1. Dataset class

```python
import torch
from torch.utils.data import Dataset

class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words_idx = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w, _ in sentence]
        tags_idx = [self.tag_to_ix[t] for _, t in sentence]
        return torch.tensor(words_idx, dtype=torch.long), torch.tensor(tags_idx, dtype=torch.long)
```

### 2.2. Collate function cho padding

```python
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = -100  # ignore_index cho CrossEntropyLoss

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_IDX)
    return sentences_padded, tags_padded
```

### 2.3. DataLoader

```python
from torch.utils.data import DataLoader

train_dataset = POSDataset(train_sentences, word_to_ix, tag_to_ix)
dev_dataset = POSDataset(dev_sentences, word_to_ix, tag_to_ix)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
```

## Task 3: Xây dựng mô hình RNN

```python
import torch.nn as nn

class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)          # [batch, seq_len, embedding_dim]
        rnn_out, _ = self.rnn(embeds)      # [batch, seq_len, hidden_dim]
        logits = self.fc(rnn_out)          # [batch, seq_len, tagset_size]
        return logits
```

## Task 4: Huấn luyện mô hình

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleRNNForTokenClassification(len(word_to_ix), len(tag_to_ix)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for sentences, tags in loader:
        sentences, tags = sentences.to(device), tags.to(device)
        optimizer.zero_grad()
        outputs = model(sentences)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sentences, tags in loader:
            sentences, tags = sentences.to(device), tags.to(device)
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=-1)
            mask = tags != PAD_IDX
            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()
    return correct / total

num_epochs = 5
for epoch in range(1, num_epochs+1):
    train_loss = train_epoch(model, train_loader)
    train_acc = evaluate(model, train_loader)
    dev_acc = evaluate(model, dev_loader)
    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Dev Acc={dev_acc:.4f}")
```

## Task 5: Dự đoán câu mới

```python
def predict_sentence(model, sentence):
    model.eval()
    tokens = sentence.strip().split()
    indices = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in tokens]
    inputs = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=-1).squeeze(0)
    idx_to_tag = {i: t for t, i in tag_to_ix.items()}
    return list(zip(tokens, [idx_to_tag[i.item()] for i in preds]))

# Ví dụ
predict_sentence(model, "I love NLP")
```

## 7. Kết quả thực hiện

* Accurance
Epoch 1: Loss=1.1047, Train Acc=0.7851, Dev Acc=0.7541
Epoch 2: Loss=0.6036, Train Acc=0.8486, Dev Acc=0.8036
Epoch 3: Loss=0.4511, Train Acc=0.8850, Dev Acc=0.8296
Epoch 4: Loss=0.3547, Train Acc=0.9087, Dev Acc=0.8448
Epoch 5: Loss=0.2860, Train Acc=0.9272, Dev Acc=0.8517
***Ví dụ dự đoán câu mới:**
  * Câu: `"I love NLP"`
  * [('I', 'PRON'), ('love', 'VERB'), ('NLP', '_')]
