import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load model
print("Loading model...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# Embed chunks (normalized, no prefix on documents)
chunk_texts = [c["topic"] + " " + c["summary"] for c in chunks]
chunk_embeddings = embedder.encode(chunk_texts, normalize_embeddings=True)

TASK = ("Given a citizen's question about Nepal government transport services, "
        "retrieve the relevant service instructions")


def make_query(question, use_prefix):
    if use_prefix:
        return f"Instruct: {TASK}\nQuery: {question}"
    return question


def search(question, use_prefix):
    q = embedder.encode([make_query(question, use_prefix)],
                        normalize_embeddings=True)[0]
    scores = chunk_embeddings @ q
    best = int(np.argmax(scores))
    return chunks[best]["chunk_id"], float(scores[best])


questions = [
    "license renew kasari garne?",
    "how do I add a new vehicle category?",
    "k ho football?",
    "नयाँ लाइसेन्स कसरी बनाउने?",
    "what is the weather today?",
]

print(f"\n{'Question':40} | {'No prefix':32} | {'With prefix':32}")
print("-" * 112)
for question in questions:
    a_id, a_score = search(question, use_prefix=False)
    b_id, b_score = search(question, use_prefix=True)
    print(f"{question:40} | {a_id:22} {a_score:.2f}   | {b_id:22} {b_score:.2f}")