import hashlib
import json
import os
import pickle

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from retrieval_config import ALIASES, ANCHORS

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
LLM_NAME = "gemma2:9b"
TOP_K = 3
CACHE_FILE = "embeddings_cache.pkl"

NOT_FOUND = ("माफ गर्नुहोस्, यो जानकारी मसँग छैन। कृपया dotm.gov.np मा जानुहोस्।\n\n"
             "Sorry, I don't have this information. Please visit dotm.gov.np")

SYSTEM_PROMPT = """You are a DoTM Nepal government services assistant.

STRICT RULES - you must follow these exactly:
1. ONLY use information from the Knowledge Base provided below
2. Several Knowledge Base sections may be given. Use only the ones that answer the question
3. Do NOT add any information that is not in the Knowledge Base
4. Do NOT say "go to local office" or anything not in the Knowledge Base
5. Reply in the SAME language the user used
6. List all steps or items completely - do not summarize or shorten
7. If the Knowledge Base does not have the answer, say exactly:
"माफ गर्नुहोस्, यो जानकारी मसँग छैन। कृपया dotm.gov.np मा जानुहोस्।"
8. Always end with: applydlnew.dotm.gov.np

You must treat the Knowledge Base as the ONLY source of truth.
Never use your own knowledge about Nepal, government, or licenses."""

# Load knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
n = len(chunks)

print("Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)


def build_doc_text(chunk):
    text = chunk["topic"] + " " + chunk["summary"]
    alias = ALIASES.get(chunk["chunk_id"], "")
    return (text + " " + alias).strip()


# Real chunks first, anchors after. Anchor indices are >= n.
all_texts = [build_doc_text(c) for c in chunks] + ANCHORS


def load_or_build_embeddings(texts):
    # The key changes whenever model, chunks, aliases or anchors change,
    # so the cache can never go stale silently.
    key = hashlib.md5((MODEL_NAME + "||".join(texts)).encode("utf-8")).hexdigest()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key:
            print("Loaded embeddings from cache.")
            return cached["embeddings"]
    print("Building embeddings (first run or data changed)...")
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"key": key, "embeddings": embeddings}, f)
    return embeddings


all_embeddings = load_or_build_embeddings(all_texts)
print(f"Ready: {n} chunks, {len(ANCHORS)} anchors.")


def retrieve(question):
    """Return top-k chunks, or None if the question looks off-topic."""
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = all_embeddings @ q

    best = int(np.argmax(scores))
    if best >= n:
        print(f"Rejected as off-topic (closest anchor: {ANCHORS[best - n]!r})")
        return None

    top = np.argsort(scores[:n])[::-1][:TOP_K]
    print("Retrieved: " + ", ".join(f"{chunks[i]['chunk_id']} ({scores[i]:.2f})" for i in top))
    return [chunks[i] for i in top]


def get_answer(question):
    retrieved = retrieve(question)
    if retrieved is None:
        return NOT_FOUND

    knowledge = "\n\n---\n\n".join(c["content"] for c in retrieved)
    user_message = f"Knowledge Base:\n{knowledge}\n\nUser Question: {question}"

    response = ollama.chat(
        model=LLM_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print("\nDoTM Chatbot ready. Type your question (or 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print("\nBot: ", end="")
        print(get_answer(question))
        print()