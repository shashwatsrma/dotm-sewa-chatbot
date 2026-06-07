import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Step 1 — Load your knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Step 2 — Load the multilingual embedding model
# This model understands both Nepali and English
print("Loading embedding model...")
# embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# embedder = SentenceTransformer("intfloat/multilingual-e5-large")
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
# Step 3 — Convert all chunk topics into embeddings
# We do this once at the start so it is fast later
chunk_texts = [chunk["topic"] + " " + chunk["summary"] for chunk in chunks]
chunk_embeddings = embedder.encode(chunk_texts)
print(f"Loaded {len(chunks)} chunks successfully.")

# Step 4 — Function to find the most relevant chunk
def find_relevant_chunk(question):
    # Convert the question into an embedding
    question_embedding = embedder.encode([question])[0]
    
    # Compare question against all chunks using cosine similarity
    similarities = []
    for chunk_emb in chunk_embeddings:
        # Cosine similarity formula
        similarity = np.dot(question_embedding, chunk_emb) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(chunk_emb)
        )
        similarities.append(similarity)
    
    # Find the chunk with highest similarity score
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    
    print(f"Best match: {chunks[best_index]['topic']} (score: {best_score:.2f})")
    return chunks[best_index]

# Step 5 — Function to get answer from Llama
# def get_answer(question):
#     # Find the right chunk first
#     relevant_chunk = find_relevant_chunk(question)
    
#     # Build the system instruction
#     system_prompt = """तपाईं नेपाल सरकारको यातायात व्यवस्था विभाग (DoTM) को एक सहायक हुनुहुन्छ।
# तपाईंले नागरिकहरूलाई सवारी सम्बन्धी सेवाहरूको बारेमा सहयोग गर्नुपर्छ।

# यी नियमहरू सधैं पालना गर्नुहोस्:
# - प्रयोगकर्ताले जुन भाषामा सोध्छन् सोही भाषामा जवाफ दिनुहोस्
# - सरल र सजिलो भाषा प्रयोग गर्नुहोस् जुन सबै नागरिकले बुझ्न सकुन्
# - कदमहरू र कागजातहरू numbered list मा राख्नुहोस्
# - तलको knowledge base बाट मात्र जवाफ दिनुहोस्
# - थाहा नभएको कुरा ईमानदारीपूर्वक भन्नुहोस्
# - जवाफको अन्तमा सधैं यो वेबसाइट उल्लेख गर्नुहोस्: applydlnew.dotm.gov.np"""

#     # Build the full message with chunk content
#     user_message = f"""Knowledge Base:
# {relevant_chunk['content']}

# User Question: {question}"""

#     # Send to Llama and get answer
#     response = ollama.chat(
#         model="llama3.1:8b",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message}
#         ]
#     )
    
#     return response["message"]["content"]



def get_answer(question):
    relevant_chunk = find_relevant_chunk(question)
    
    # Check similarity score
    question_embedding = embedder.encode([question])[0]
    similarities = []
    for chunk_emb in chunk_embeddings:
        similarity = np.dot(question_embedding, chunk_emb) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(chunk_emb)
        )
        similarities.append(similarity)
    
    best_score = max(similarities)
    
    # If score too low, ask for clarification
    if best_score < 0.30:
        return "माफ गर्नुहोस्, मैले तपाईंको प्रश्न राम्रोसँग बुझिनँ। कृपया अझ स्पष्ट रूपमा सोध्नुहोस्। \n\nSorry, I could not understand your question clearly. Please ask more specifically about driving license services."
    
    # Continue with normal answer
#     system_prompt = """तपाईं नेपाल सरकारको यातायात व्यवस्था विभाग (DoTM) को एक सहायक हुनुहुन्छ।
# तपाईंले नागरिकहरूलाई सवारी सम्बन्धी सेवाहरूको बारेमा सहयोग गर्नुपर्छ।

# यी नियमहरू सधैं पालना गर्नुहोस्:
# - प्रयोगकर्ताले जुन भाषामा सोध्छन् सोही भाषामा जवाफ दिनुहोस्
# - सरल र सजिलो भाषा प्रयोग गर्नुहोस्
# - कदमहरू numbered list मा राख्नुहोस्
# - तलको knowledge base बाट मात्र जवाफ दिनुहोस्
# - थाहा नभएको कुरा ईमानदारीपूर्वक भन्नुहोस्
# - जवाफको अन्तमा सधैं भन्नुहोस्: applydlnew.dotm.gov.np"""

    system_prompt = """You are a DoTM Nepal government services assistant.

    STRICT RULES - you must follow these exactly:
    1. ONLY use information from the Knowledge Base provided below
    2. Do NOT add any information that is not in the Knowledge Base
    3. Do NOT say "go to local office" or anything not in the Knowledge Base
    4. Reply in the SAME language the user used
    5. List all steps or items completely - do not summarize or shorten
    6. If the Knowledge Base does not have the answer, say exactly:
    "माफ गर्नुहोस्, यो जानकारी मसँग छैन। कृपया dotm.gov.np मा जानुहोस्।"
    7. Always end with: applydlnew.dotm.gov.np

    You must treat the Knowledge Base as the ONLY source of truth.
    Never use your own knowledge about Nepal, government, or licenses."""

    user_message = f"""Knowledge Base:
    {relevant_chunk['content']}

User Question: {question}"""

    response = ollama.chat(
        # model="qwen2.5:7b", #recommended 1
        # model="llama3.1:8b",
        # model="gemini-2.5-pro:8b",
        model='gemma2:9b', # recommended 2
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    answer = response["message"]["content"]
    return answer
    # return response["message"]["content"]


# Step 6 — Simple test loop
if __name__ == "__main__":
    print("\nDoTM Chatbot ready. Type your question (or 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print("\nBot: ", end="")
        answer = get_answer(question)
        print(answer)
        print()