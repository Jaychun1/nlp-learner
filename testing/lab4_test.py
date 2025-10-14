from src.representations.word_embedder import WordEmbedder

we = WordEmbedder("glove-wiki-gigaword-50")

print(we.getVector("king"))

print("sim(king, queen):", we.get_similarity("king", "queen"))
print("sim(king, man):", we.get_similarity("king", "man"))

print(we.get_most_similar("computer", 10))

vec = we.embed_document("The queen rules the country.")
print(vec)
