def load_raw_text_data(file_path: str) -> str:
    """
    Load raw text data from a file and return as a single string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
