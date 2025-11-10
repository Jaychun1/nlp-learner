from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.core.text_classifier import TextClassifier
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations import tfidf_vectorizer

def test_text_classifier_evaluation():


    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
    ]

    labels = [1, 0, 1, 0, 1, 0]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
    )

    # Initialize the tokenizer and vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = tfidf_vectorizer(tokenizer=tokenizer)

    # Instantiate and train classifier
    classifier = TextClassifier(vectorizer)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Optional: include an assertion for automated testing
    assert acc >= 0.5, "Model accuracy should be at least 50%"