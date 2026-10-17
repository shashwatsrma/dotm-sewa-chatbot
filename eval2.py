import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading model...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

chunk_ids = [c["chunk_id"] for c in chunks]
n = len(chunks)

# (question, script, expected chunk_id)
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

# Fake "off-topic" chunks. If the best match is one of these, we reject.
ANCHORS = [
    "football cricket sports match score",
    "weather forecast rain temperature",
    "politics prime minister government election",
    "cooking recipe food momo",
    "citizenship certificate passport tax and other government services",
    "movies music entertainment celebrity",
    "mobile phones laptops technology shopping",
    "bitcoin stock market money investment",
]

# Held-out off-topic questions (different wording from the anchors)
OUT_TEST = [
    "who won the world cup?",
    "kathmandu ma aja pani parchha?",
    "how to bake a cake",
    "bitcoin ko price kati ho?",
    "nepal ko rastrapati ko ho?",
    "naya passport banauna kati din lagchha?",
    "best laptop under 80000 rupees",
    "k ho football?",
    "what is the weather today?",
    "citizenship certificate kasari banaune?",
]

VARIANTS = {
    "A topic+summary (current)": lambda c: c["topic"] + " " + c["summary"],
    "B topic+summary+content": lambda c: c["topic"] + " " + c["summary"] + " " + c["content"],
    "C topic+content": lambda c: c["topic"] + " " + c["content"],
}

# Embed queries and anchors once
in_q = embedder.encode([q for q, _, _ in IN_SCOPE], normalize_embeddings=True)
out_q = embedder.encode(OUT_TEST, normalize_embeddings=True)
anchor_emb = embedder.encode(ANCHORS, normalize_embeddings=True)

summary = []

for name, text_fn in VARIANTS.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    doc_emb = embedder.encode([text_fn(c) for c in chunks], normalize_embeddings=True)
    all_emb = np.vstack([doc_emb, anchor_emb])   # real chunks first, anchors after

    # In-scope questions
    in_scores = in_q @ all_emb.T
    best = in_scores.argmax(axis=1)
    per_script = {"english": [0, 0], "romanized": [0, 0], "nepali": [0, 0]}
    wrongly_rejected = 0

    for i, (q, script, expected) in enumerate(IN_SCOPE):
        per_script[script][1] += 1
        b = int(best[i])
        if b >= n:
            wrongly_rejected += 1
            print(f"MISS [{script}] {q}\n     expected {expected}, REJECTED as off-topic")
        elif chunk_ids[b] == expected:
            per_script[script][0] += 1
        else:
            print(f"MISS [{script}] {q}\n     expected {expected}, got {chunk_ids[b]}")

    total_ok = sum(v[0] for v in per_script.values())
    print("\nTop-1 accuracy (rejections count as misses):")
    for script, (ok, total) in per_script.items():
        print(f"  {script:10} {ok}/{total}")
    print(f"  {'overall':10} {total_ok}/{len(IN_SCOPE)}")

    # Off-topic questions
    out_scores = out_q @ all_emb.T
    out_best = out_scores.argmax(axis=1)
    rejected = 0
    print("\nOff-topic questions:")
    for i, q in enumerate(OUT_TEST):
        b = int(out_best[i])
        if b >= n:
            rejected += 1
            print(f"  rejected OK     {q}")
        else:
            print(f"  LEAKED to {chunk_ids[b]:22} {q}")

    print(f"\nOff-topic rejected: {rejected}/{len(OUT_TEST)}")
    summary.append((name, total_ok, wrongly_rejected, rejected))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Variant':30} {'in-scope ok':>12} {'wrongly rej':>12} {'off-topic rej':>14}")
for name, ok, wr, rej in summary:
    print(f"{name:30} {ok:>9}/{len(IN_SCOPE)} {wr:>12} {rej:>11}/{len(OUT_TEST)}")