import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("knowledge_base.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading model...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

chunk_ids = [c["chunk_id"] for c in chunks]
chunk_texts = [c["topic"] + " " + c["summary"] for c in chunks]
chunk_embeddings = embedder.encode(chunk_texts, normalize_embeddings=True)

TASK = ("Given a citizen's question about Nepal government transport services, "
        "retrieve the relevant service instructions")


def make_query(question, use_prefix):
    if use_prefix:
        return f"Instruct: {TASK}\nQuery: {question}"
    return question


def get_scores(question, use_prefix):
    q = embedder.encode([make_query(question, use_prefix)],
                        normalize_embeddings=True)[0]
    return chunk_embeddings @ q


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

OUT_OF_SCOPE = [
    "k ho football?",
    "what is the weather today?",
    "who is the prime minister of Nepal?",
    "momo banaune recipe deu",
    "citizenship certificate kasari banaune?",
    "passport renew garna k k chahinchha?",
]


def run(use_prefix):
    label = "WITH prefix" if use_prefix else "NO prefix"
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    per_script = {"english": [0, 0], "romanized": [0, 0], "nepali": [0, 0]}
    in_top, in_margin = [], []

    for question, script, expected in IN_SCOPE:
        s = get_scores(question, use_prefix)
        order = np.argsort(s)[::-1]
        best = chunk_ids[order[0]]
        top, margin = float(s[order[0]]), float(s[order[0]] - s[order[1]])
        in_top.append(top)
        in_margin.append(margin)

        per_script[script][1] += 1
        if best == expected:
            per_script[script][0] += 1
        else:
            print(f"MISS [{script}] {question}\n     expected {expected}, got {best} ({top:.2f})")

    print("\nTop-1 accuracy:")
    for script, (ok, total) in per_script.items():
        print(f"  {script:10} {ok}/{total}")
    total_ok = sum(v[0] for v in per_script.values())
    print(f"  {'overall':10} {total_ok}/{len(IN_SCOPE)}")

    out_top, out_margin = [], []
    print("\nOut-of-scope scores:")
    for question in OUT_OF_SCOPE:
        s = get_scores(question, use_prefix)
        order = np.argsort(s)[::-1]
        top, margin = float(s[order[0]]), float(s[order[0]] - s[order[1]])
        out_top.append(top)
        out_margin.append(margin)
        print(f"  {top:.2f} (margin {margin:.3f})  {question}")

    print("\nCan a threshold separate in-scope from out-of-scope?")
    print(f"  top score : in-scope min {min(in_top):.2f} | out-of-scope max {max(out_top):.2f}")
    if min(in_top) > max(out_top):
        print(f"  -> YES, threshold around {(min(in_top) + max(out_top)) / 2:.2f}")
    else:
        print("  -> NO, ranges overlap, so a single score cutoff will not work")
    print(f"  margin    : in-scope min {min(in_margin):.3f} | out-of-scope max {max(out_margin):.3f}")


run(use_prefix=False)
run(use_prefix=True)