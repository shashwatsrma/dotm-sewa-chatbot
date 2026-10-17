import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading model...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

chunk_ids = [c["chunk_id"] for c in chunks]
n = len(chunks)

IN_SCOPE = [
    ("how do I renew my driving license?", "english", "dotm_dl_renewal"),
    ("license renew kasari garne?", "romanized", "dotm_dl_renewal"),
    ("लाइसेन्स नवीकरण कसरी गर्ने?", "nepali", "dotm_dl_renewal"),
    ("how to add a category to my existing license?", "english", "dotm_dl_add_category"),
    ("existing license ma category thapna kasari garne?", "romanized", "dotm_dl_add_category"),
    ("मेरो लाइसेन्समा नयाँ श्रेणी कसरी थप्ने?", "nepali", "dotm_dl_add_category"),
    ("what are the license categories and vehicle types?", "english", "dotm_dl_categories"),
    ("license ko category haru k k chan?", "romanized", "dotm_dl_categories"),
    ("लाइसेन्सका श्रेणीहरू के के छन्?", "nepali", "dotm_dl_categories"),
    ("what should I be careful about when applying online?", "english", "dotm_dl_warnings"),
    ("online apply garda ke ke sawdhani line?", "romanized", "dotm_dl_warnings"),
    ("अनलाइन आवेदन दिँदा के के सावधानी अपनाउनु पर्छ?", "nepali", "dotm_dl_warnings"),
    ("apply for a new license with medical check at the DoTM office", "english", "dotm_dl_new_office"),
    ("naya license ko lagi office ma medical garera kasari apply garne?", "romanized", "dotm_dl_new_office"),
    ("कार्यालयमा स्वास्थ्य परीक्षण गरेर नयाँ लाइसेन्स कसरी बनाउने?", "nepali", "dotm_dl_new_office"),
    ("apply for a new license when the medical check is done at a clinic", "english", "dotm_dl_new_clinic"),
    ("clinic bata medical garera naya license kasari banaune?", "romanized", "dotm_dl_new_clinic"),
    ("क्लिनिकमा स्वास्थ्य परीक्षण गरेर नयाँ लाइसेन्स कसरी बनाउने?", "nepali", "dotm_dl_new_clinic"),
]

# Sample questions added to each chunk's embedded text (doc expansion)
ALIASES = {
    "dotm_dl_renewal": "renew driving license expired license nabikaran | license renew garna | लाइसेन्स नवीकरण म्याद सकिएको",
    "dotm_dl_add_category": "add vehicle category to old license | category thap purano license ma | पुरानो लाइसेन्समा थप श्रेणी",
    "dotm_dl_categories": "license categories vehicle types bike scooter car jeep bus truck | kun gadi ko license kun category | सवारी प्रकार श्रेणी",
    "dotm_dl_warnings": "important warnings precautions mistakes before applying online | online form bharda dhyan dinu parne kura | चेतावनी सावधानी",
    "dotm_dl_new_office": "first time new driving license medical check at DoTM office | naya license office ma medical | नयाँ लाइसेन्स कार्यालयमा स्वास्थ्य परीक्षण",
    "dotm_dl_new_clinic": "first time new driving license medical check at clinic doctor | naya license clinic ma medical | नयाँ लाइसेन्स क्लिनिक चिकित्सक",
}

ANCHORS = [
    "football cricket sports match score",
    "weather forecast rain temperature",
    "politics prime minister president government election",
    "cooking recipe food momo",
    "citizenship certificate passport tax and other government services",
    "movies music entertainment celebrity",
    "mobile phones laptops technology shopping",
    "bitcoin stock market money investment",
    "aja ko mausam pani parchha tapkram",
    "rastrapati pradhanmantri rajniti sarkar chunav",
    "khel football cricket match",
    "khana pakaune recipe momo",
    "खेल फुटबल क्रिकेट म्याच",
    "मौसम पानी तापक्रम",
    "राजनीति प्रधानमन्त्री राष्ट्रपति चुनाव",
    "खाना पकाउने रेसिपी",
    "general knowledge geography science history",
    "programming code computer help",
]

# Completely new off-topic questions
OUT_TEST = [
    "nepal ko pradhanmantri ko hun?",
    "aaja kathmandu ko temperature kati chha?",
    "who won the IPL this year?",
    "pizza kasari banaune?",
    "mero ghar ko tax kasari tirne?",
    "visa application kasari garne?",
    "सुन्दर कविता लेख्नुहोस्",
    "how tall is mount everest?",
    "birth certificate kasari banaune?",
    "write python code to sort a list",
]

def make_docs(use_alias):
    docs = []
    for c in chunks:
        text = c["topic"] + " " + c["summary"]
        if use_alias:
            text += " " + ALIASES[c["chunk_id"]]
        docs.append(text)
    return docs

in_q = embedder.encode([q for q, _, _ in IN_SCOPE], normalize_embeddings=True)
out_q = embedder.encode(OUT_TEST, normalize_embeddings=True)
anchor_emb = embedder.encode(ANCHORS, normalize_embeddings=True)

summary = []
for name, use_alias in [("A topic+summary", False), ("D topic+summary+aliases", True)]:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    doc_emb = embedder.encode(make_docs(use_alias), normalize_embeddings=True)
    all_emb = np.vstack([doc_emb, anchor_emb])

    # Top-k recall among real chunks
    real_scores = in_q @ doc_emb.T
    hits = {1: 0, 2: 0, 3: 0}
    per_script = {"english": [0, 0], "romanized": [0, 0], "nepali": [0, 0]}
    for i, (q, script, expected) in enumerate(IN_SCOPE):
        ranked = [chunk_ids[j] for j in np.argsort(real_scores[i])[::-1]]
        per_script[script][1] += 1
        if ranked[0] == expected:
            per_script[script][0] += 1
        else:
            print(f"MISS [{script}] {q}\n     expected {expected}, top3 = {ranked[:3]}")
        for k in (1, 2, 3):
            if expected in ranked[:k]:
                hits[k] += 1

    print("\nTop-1 per script:", {s: f"{a}/{b}" for s, (a, b) in per_script.items()})
    print(f"Recall@1 {hits[1]}/18 | Recall@2 {hits[2]}/18 | Recall@3 {hits[3]}/18")

    # Anchor rejection
    wrongly_rejected = int(((in_q @ all_emb.T).argmax(axis=1) >= n).sum())
    out_best = (out_q @ all_emb.T).argmax(axis=1)
    rejected = int((out_best >= n).sum())
    print(f"\nIn-scope wrongly rejected: {wrongly_rejected}/18")
    print(f"Off-topic rejected: {rejected}/{len(OUT_TEST)}")
    for i, q in enumerate(OUT_TEST):
        if out_best[i] < n:
            print(f"  LEAKED to {chunk_ids[out_best[i]]:22} {q}")

    summary.append((name, hits[1], hits[3], wrongly_rejected, rejected))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Variant':28} {'R@1':>6} {'R@3':>6} {'wrong rej':>10} {'OOS rej':>9}")
for name, r1, r3, wr, rej in summary:
    print(f"{name:28} {r1:>4}/18 {r3:>4}/18 {wr:>10} {rej:>6}/{len(OUT_TEST)}")