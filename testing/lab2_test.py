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
