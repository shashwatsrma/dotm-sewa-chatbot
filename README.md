# 🇳🇵 DoTM Sewa Chatbot — Nepal Government Transport Services Assistant

A bilingual AI-powered chatbot for Nepal's Department of Transport Management (DoTM) that helps citizens get accurate information about driving license services in **Nepali, English, and Romanized Nepali** — all running **100% locally** on your machine with zero cost and no internet dependency after setup.

---

## 📌 What This Project Does

Citizens can ask questions like:
- "नयाँ लाइसेन्स कसरी बनाउने?"
- "How do I apply for a new driving license?"
- "license renew kasari garne?"

And get clear, structured, accurate answers instantly — without visiting a government office or navigating complex websites.

---

## 🖥️ Interface Preview

- Dark navy interface with Nepal government crimson and gold theme
- Bilingual chat bubbles — Nepali and English
- Suggested question chips for common queries
- Typing indicator while AI is thinking
- Fully responsive layout

---

## 🏗️ Architecture

```
User Question (Nepali / English / Romanized Nepali)
        ↓
index.html — Chat Interface (HTML + CSS + JS)
        ↓
server.py — Flask REST API (port 5000)
        ↓
rag.py — RAG Pipeline
        ↓
Multilingual Embedding Model (intfloat/multilingual-e5-large-instruct)
        ↓
Cosine Similarity Search across Knowledge Base Chunks
        ↓
Most Relevant Chunk Retrieved from knowledge_base.json
        ↓
Gemma2:9b (via Ollama) generates citizen-friendly answer
        ↓
Answer displayed in same language as user's question
```

This is a **RAG (Retrieval Augmented Generation)** architecture — the AI only answers from verified DoTM knowledge base data, never from general training knowledge.

---

## 📂 Project Structure

```
dotm-sewa-chatbot/
│
├── rag.py                  # Core RAG pipeline — embeddings + retrieval + LLM
├── server.py               # Flask REST API server (middleman)
├── index.html              # Chat interface (frontend)
├── style.css               # Stylesheet for chat interface
├── knowledge_base.json     # Structured DoTM knowledge base (prototype — 6 chunks)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧠 Knowledge Base

The knowledge base is compiled from:
- **DoTM Karyawidhi Nirdesika 2060** (यातायात कार्यविधि निर्देशिका २०६०)
- **DoTM Official User Manual** for Online Driving License System
- **applydlnew.dotm.gov.np** — live portal verification

Current prototype covers **6 chunks** across driving license services:

| Chunk ID | Topic |
|---|---|
| dotm_dl_new_office | New license application (medical at office) |
| dotm_dl_new_clinic | New license application (medical at clinic) |
| dotm_dl_renewal | License renewal process |
| dotm_dl_add_category | Add vehicle category to existing license |
| dotm_dl_categories | All license categories and vehicle types |
| dotm_dl_warnings | Important warnings for online applicants |

> Full knowledge base expansion to 300-400 chunks is in progress covering all DoTM services.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | `intfloat/multilingual-e5-large-instruct` |
| Language Model | `gemma2:9b` via Ollama |
| Similarity Search | Cosine similarity (NumPy) |
| Backend | Python Flask |
| Frontend | HTML + CSS + JavaScript |
| Languages Supported | Nepali (Devanagari), English, Romanized Nepali |
| Runtime | Python 3.9+ |
| Cost | Rs. 0 — fully free, fully local |

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.com) installed
- macOS / Linux (Windows with WSL also works)
- Google Chrome (for best experience)

### Step 1 — Clone the repository
```bash
git clone https://github.com/shashwatsrma/dotm-sewa-chatbot.git
cd dotm-sewa-chatbot
```

### Step 2 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download the LLM model
```bash
ollama pull gemma2:9b
```

### Step 5 — Start the backend server
```bash
python server.py
```

### Step 6 — Open the chat interface
Open `index.html` in Google Chrome. That's it — your chatbot is running.

---

## 💬 Example Interactions

```
You: how do i apply for new license?
Bot: १. applydlnew.dotm.gov.np मा जानुहोस्
     २. मोबाइल नम्बर र captcha हाल्नुहोस्
     ३. SMS बाट आएको MPIN हाल्नुहोस्
     ...
     applydlnew.dotm.gov.np

You: license renew kasari garne?
Bot: १. applydlnew.dotm.gov.np मा जानुहोस्
     २. मोबाइल नम्बर र MPIN हाल्नुहोस्
     ...

You: k ho football?
Bot: माफ गर्नुहोस्, यो जानकारी मसँग छैन। कृपया dotm.gov.np मा जानुहोस्।
```

---

## 📊 Performance (Prototype — 6 chunks)

| Question Type | Similarity Score | Result |
|---|---|---|
| Nepali (Devanagari) | 0.88 — 0.91 | ✅ Correct |
| English | 0.84 — 0.86 | ✅ Correct |
| Romanized Nepali | 0.82 — 0.87 | ✅ Correct |
| Irrelevant questions | < 0.30 | ✅ Correctly rejected |

---

## 🔭 Roadmap

- [x] Phase 1 — Structured bilingual knowledge base from DoTM documents
- [x] Phase 2 — Multilingual RAG pipeline (embeddings + local LLM)
- [x] Phase 3 — Flask REST API server
- [x] Phase 4 — Polished bilingual chat interface
- [ ] Phase 5 — Voice input (Nepali speech-to-text)
- [ ] Expand knowledge base to 300-400 chunks
- [ ] Conversation history (multi-turn memory)
- [ ] Add citizenship, passport, vehicle registration services
- [ ] Embeddings cache for fast startup

---

## 🎓 Academic Context

This project is developed as a PROTOTYPE **BCT Final Year Major Project** at IOE, Tribhuvan University, Nepal.

**Primary Contributions:**
1. Structured bilingual knowledge base compiled from official DoTM documents
2. Multilingual semantic retrieval pipeline for Nepali government services
3. Fully local deployment — no cloud dependency, no API cost
4. Voice-enabled citizen interface (upcoming)
5. System is model-agnostic — LLM can be swapped without changing the pipeline

---

## 📞 Data Sources

- [DoTM Official Website](https://dotm.gov.np)
- [Online Driving License Portal](https://applydlnew.dotm.gov.np)
- DoTM Karyawidhi Nirdesika 2060 (यातायात कार्यविधि निर्देशिका २०६०)
- DoTM User Manual for Online Driving License System

---

## ⚠️ Disclaimer

This chatbot is built for educational purposes as a BCT final year project. Always verify critical information directly from the official DoTM website at [dotm.gov.np](https://dotm.gov.np).

---

## 👤 Author

**Shashwat** — BCT Student, IOE Tribhuvan University, Nepal
GitHub: [shashwatsrma](https://github.com/shashwatsrma)
