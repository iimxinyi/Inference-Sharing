from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "A cat is sleeping peacefully on a sunlit window sill.",
    "A cat is playing with a ball of yarn in a cozy living room.",
    "A cat is sitting on a bookshelf surrounded by books.",
    "A cat is grooming itself on a soft blanket.",
    "A cat is lying on a windowsill, eyes closed, in a quiet library.",
    "A dog sleeps peacefully on a cozy couch.",
    "A dog sniffs curiously at a flower garden.",
    "A dog pounces playfully at a butterfly in a meadow.",
    "A dog lounges lazily on a patio during a summer afternoon.",
    "A dog rests contentedly by a window with a view of the mountains.",
    "A tiger rests under the shade of a large tree in the jungle.",
    "A tiger slinks through the underbrush in the twilight.",
    "A tiger prowls the edge of a dense thicket, looking for prey.",
    "A tiger lounges in the shade, conserving energy for the hunt.",
    "A tiger rests under a waterfall, the cool water soothing its fur.",
    "A panda sleeps peacefully in a sunny spot in the forest.",
    "A panda munches on bamboo while sitting in a tree hollow.",
    "A panda walks along a path in a bamboo forest at dusk.",
    "A panda munches on bamboo while sitting on a log.",
    "A panda walks along a path in a bamboo forest at sunrise."
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)