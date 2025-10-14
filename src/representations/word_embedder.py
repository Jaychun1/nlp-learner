import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from src.preprocessing.embed_Tokenizer import WordEmbeddingTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from typing import List, Tuple, Optional


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
    
    