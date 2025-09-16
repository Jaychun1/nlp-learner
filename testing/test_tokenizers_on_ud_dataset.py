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