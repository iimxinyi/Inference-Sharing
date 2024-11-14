from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Demo
sentences = [
    "A cat is sleeping peacefully on a sunlit window sill.",
    "A cat is playing with a ball of yarn in a cozy living room.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)