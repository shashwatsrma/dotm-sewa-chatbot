# 🇳🇵 DoTM Sewa Chatbot — Nepal Government Transport Services Assistant

A bilingual AI-powered chatbot for Nepal's Department of Transport Management (DoTM) that helps citizens get accurate information about driving license services in **Nepali, English, and Romanized Nepali** — all running **100% locally** on your machine with no internet dependency after setup.

---

## 📌 What This Project Does

Citizens can ask questions like:
- "नयाँ लाइसेन्स कसरी बनाउने?"
- "How do I apply for a new driving license?"
- "license renew kasari garne?"

And get clear, structured, accurate answers instantly — without visiting a government office or navigating complex websites.

---

## 🏗️ Architecture

```
User Question (Nepali / English / Romanized Nepali)
        ↓
Multilingual Embedding Model (intfloat/multilingual-e5-large-instruct)
        ↓
Cosine Similarity Search across Knowledge Base Chunks
        ↓
Most Relevant Chunk Retrieved
        ↓
Gemma2:9b (via Ollama) generates citizen-friendly answer
        ↓
Answer in same language as user's question
```

This is a **RAG (Retrieval Augmented Generation)** architecture — the AI only answers from verified DoTM knowledge base data, never from general training knowledge.

---

## 📂 Project Structure

```
dotm-chatbot/
│
├── rag.py                  # Core RAG pipeline — embeddings + retrieval + LLM
├── knowledge_base.json     # Structured DoTM knowledge base (6 chunks, prototype)
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
| Languages Supported | Nepali (Devanagari), English, Romanized Nepali |
| Runtime | Python 3.9+ |
| Dependencies | sentence-transformers, ollama, numpy |

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.com) installed
- macOS / Linux (Windows with WSL also works)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dotm-chatbot.git
cd dotm-chatbot
```

### Step 2 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download the LLM model
```bash
ollama pull gemma2:9b
```

### Step 5 — Run the chatbot
```bash
python3 rag.py
```

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

- [x] Core RAG pipeline
- [x] Multilingual embedding support (Nepali + English + Romanized)
- [x] Local LLM via Ollama (no API dependency)
- [x] Confidence threshold fallback for irrelevant questions
- [ ] Flask REST API server
- [ ] Web chat interface (HTML/CSS/JS)
- [ ] Voice input (Nepali speech-to-text)
- [ ] Expand knowledge base to 300-400 chunks
- [ ] Add citizenship, passport, vehicle registration services
- [ ] Embeddings cache for fast startup

---

## 🎓 Academic Context

This project is developed as a **BCT Final Year Major Project** at IOE, Tribhuvan University, Nepal.

**Primary Contributions:**
1. Structured bilingual knowledge base from official DoTM documents
2. Multilingual semantic retrieval pipeline for Nepali government services
3. Voice-enabled citizen interface (upcoming)
4. Fully local deployment — no cloud dependency

---

## 📋 Requirements

Create `requirements.txt` with:
```
sentence-transformers
numpy
ollama
```

---

## 📞 Data Sources

- [DoTM Official Website](https://dotm.gov.np)
- [Online Driving License Portal](https://applydlnew.dotm.gov.np)
- DoTM Karyawidhi Nirdesika 2060 (यातायात कार्यविधि निर्देशिका २०६०)
- DoTM User Manual for Online Driving License System

---

## ⚠️ Disclaimer

This chatbot is built for educational purposes as a final year project. Always verify critical information directly from the official DoTM website at [dotm.gov.np](https://dotm.gov.np).

---

## 👤 Author

**Shashwat** — BCT Student, IOE Tribhuvan University, Nepal
