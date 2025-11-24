## Task1

```python
from transformers import pipeline

mask_filler = pipeline("fill-mask", model="bert-base-uncased", framework="pt")

input_sentence = "Hanoi is the [MASK] of Vietnam."
predictions = mask_filler(input_sentence, top_k=5)

print(f"Câu gốc: {input_sentence}")
for pred in predictions:
    print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
    print(f" -> Câu hoàn chỉnh: {pred['sequence']}")
```

Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']

- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
  Device set to use cpu
  Câu gốc: Hanoi is the [MASK] of Vietnam.
  Dự đoán: 'capital' với độ tin cậy: 0.9991
  -> Câu hoàn chỉnh: hanoi is the capital of vietnam.
  Dự đoán: 'center' với độ tin cậy: 0.0001
  -> Câu hoàn chỉnh: hanoi is the center of vietnam.
  Dự đoán: 'birthplace' với độ tin cậy: 0.0001
  -> Câu hoàn chỉnh: hanoi is the birthplace of vietnam.
  Dự đoán: 'headquarters' với độ tin cậy: 0.0001
  -> Câu hoàn chỉnh: hanoi is the headquarters of vietnam.
  Dự đoán: 'city' với độ tin cậy: 0.0001
  -> Câu hoàn chỉnh: hanoi is the city of vietnam.

Mô hình có dự đoán đúng từ capital không?

Có, BERT thường trả về 'capital' là dự đoán top 1, vì đây là từ phù hợp nhất với ngữ cảnh của câu "Hanoi is the [MASK] of Vietnam."

Tại sao BERT (encoder-only) phù hợp cho tác vụ này?

BERT được huấn luyện bằng Masked Language Modeling (MLM), nghĩa là học dự đoán từ bị che dựa trên toàn bộ ngữ cảnh hai bên.

Encoder-only cho phép mô hình xem cả trái lẫn phải của từ bị mask, giúp dự đoán chính xác hơn so với decoder-only.

## task2

```python

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", framework="pt")
prompt = "The best thing about learning NLP is"
generated_texts = generator(prompt, max_length=50, num_return_sequences=1)

print(f"Câu mồi: '{prompt}'")
for text in generated_texts:
    print("Văn bản được sinh ra:")
    print(text['generated_text'])

```

Device set to use cpu
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Câu mồi: 'The best thing about learning NLP is'
Văn bản được sinh ra:
The best thing about learning NLP is to have a real learning curve, don't you think?

In the book you give examples of how to write NLP in a few simple sentences. Here are some of the best examples.

First, I'm going to take a look at what NLP is and what I mean by it.

NLP is a language that is built from the ground up to be the most powerful language on the planet. It's a language that is built to be extremely powerful. It is built to be accessible to people everywhere. It is a language that is built to be the most powerful language on earth. It is built to be the most powerful currency on earth. It is built to be the most powerful language on earth. It is the language that is responsible for the creation of civilization. It is the language that is responsible for the evolution of civilization.

So, to be sure, there are different things about NLP that you can learn and apply to your own life. But there are also things that you can learn and apply to your own life.

NLP is the most powerful language on the planet. It's the language that is responsible for the evolution of civilization. It is the language that is responsible for the evolution of civilization.

Kết quả sinh ra có hợp lý không?

Thường hợp lý, ngữ pháp đúng, ý nghĩa mạch lạc.

Ví dụ: "The best thing about learning NLP is that you can build intelligent applications..."

Lưu ý: kết quả cụ thể phụ thuộc vào mô hình GPT được sử dụng (thường là gpt2).

Tại sao GPT (decoder-only) phù hợp cho tác vụ này?

GPT huấn luyện bằng causal language modeling, nghĩa là dự đoán token tiếp theo dựa trên các token trước đó.

Vì tác vụ "dự đoán token tiếp theo" chỉ cần ngữ cảnh bên trái, GPT là lựa chọn tự nhiên hơn so với BERT.

### task3

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = ["This is a sample sentence."]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
attention_mask = inputs['attention_mask']
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
sentence_embedding = sum_embeddings / sum_mask

print("Vector biểu diễn của câu:")
print(sentence_embedding)
print("\nKích thước của vector:", sentence_embedding.shape)

```

Vector biểu diễn của câu:
tensor([[-6.3875e-02, -4.2837e-01, -6.6779e-02, -3.8430e-01, -6.5785e-02,
         -2.1826e-01,  4.7636e-01,  4.8659e-01,  3.9276e-05, -7.4274e-02,
         -7.4740e-02, -4.7635e-01, -1.9773e-01,  2.4824e-01, -1.2162e-01,
          1.6679e-01,  2.1045e-01, -1.4576e-01,  1.2637e-01,  1.8636e-02,
          2.4640e-01,  5.7090e-01, -4.7014e-01,  1.3782e-01,  7.3650e-01,
         -3.3808e-01, -5.0328e-02, -1.6453e-01, -4.3517e-01, -1.2900e-01,
          1.6516e-01,  3.4004e-01, -1.4930e-01,  2.2422e-02, -1.0488e-01,
         -5.1916e-01,  3.2964e-01, -2.2162e-01, -3.4206e-01,  1.1994e-01,
         -7.0148e-01, -2.3126e-01,  1.1224e-01,  1.2550e-01, -2.5191e-01,
         -4.6374e-01, -2.7261e-02, -2.8416e-01, -9.9250e-02, -3.7020e-02,
         -8.9192e-01,  2.5005e-01,  1.5816e-01,  2.2701e-01, -2.8497e-01,
          4.5300e-01,  5.0901e-03, -7.9441e-01, -3.1007e-01, -1.7403e-01,
          4.3029e-01,  1.6816e-01,  1.0590e-01, -4.8987e-01,  3.1856e-01,
          3.2861e-01, -1.3403e-02,  1.8808e-01, -1.0905e+00,  2.1009e-01,
         -6.7579e-01, -5.7076e-01,  8.5945e-02,  1.9121e-01, -3.3818e-01,
          2.7744e-01, -4.0539e-01,  3.1305e-01, -4.1197e-01, -5.6820e-01,
         -3.9074e-01,  4.0747e-01,  9.9897e-02,  2.3719e-01,  1.0154e-01,
         -2.5670e-01, -2.0583e-01,  1.1763e-01, -5.1439e-01,  4.0979e-01,
          1.2149e-01,  1.9333e-02, -5.9030e-02, -2.0141e-01,  7.0860e-01,
         -6.4610e-02,  2.4781e-02, -9.0587e-03,  1.9667e-02,  3.0815e-01,
         -4.9832e-02, -1.0691e+00,  6.1072e-01, -4.9723e-02, -1.5156e-01,
         -6.7778e-02,  4.7811e-02,  5.2102e-01,  1.6951e-01,  1.0144e-02,
          5.3093e-01, -7.8190e-02,  6.5842e-02, -2.9383e-01, -4.6045e-01,
          4.2072e-01,  1.1822e-01,  2.3631e-01, -4.5379e-02, -1.3740e-01,
...
         -3.9554e-02, -5.4193e-01, -4.4191e-01,  2.4927e-01,  6.6517e-01,
         -1.7534e-01, -1.2388e-01,  3.1970e-01]])

Kích thước của vector: torch.Size([1, 768])

768 là hidden size của mô hình bert-base-uncased. Mỗi token được biểu diễn bằng vector 768 chiều.

Tại sao dùng attention_mask?

attention_mask giúp xác định các token thực và padding.

Khi tính trung bình (Mean Pooling), các token padding không được tính để vector biểu diễn phản ánh chính xác nội dung câu.

Nếu không dùng mask, các token padding (giá trị 0) sẽ làm vector trung bình bị méo, không phản ánh ý nghĩa thực sự của câu.
