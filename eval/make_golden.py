"""Usage: uv run python -m eval.make_golden   -> writes eval/golden.json (REVIEW IT BY HAND)"""
import json, random, time
from collections import defaultdict
from app import config
from app.generate import call_llm

N_PER_SOURCE = 12

chunks = [json.loads(l) for l in open(config.CHUNKS_PATH, encoding="utf-8")]
by_source = defaultdict(list)
for c in chunks:
    if len(c["text"].split()) >= 80 and not c["ocr"]:
        by_source[c["source"]].append(c)

random.seed(0)
picked = []
for src, pool in by_source.items():
    picked += random.sample(pool, min(N_PER_SOURCE, len(pool)))

SYS = ("Write ONE exam-style question that can be answered using only the passage. "
       "Make it self-contained (no phrases like 'in the passage' or 'the text'), "
       "and paraphrase instead of copying phrases from the passage. Output only the question.")

golden = []
for n, c in enumerate(picked, 1):
    q, _ = call_llm(SYS, c["text"])
    golden.append({"question": q.strip(), "answerable": True,
                   "source": c["source"], "pages": [c["page"]], "passage": c["text"][:300]})
    print(f"[{n}/{len(picked)}] {q.strip()[:80]}")
    time.sleep(2)

# Unanswerable questions. Add 3-4 physics ones that your chapters do NOT cover.
for q in ["What is the capital of Australia?",
          "Explain how CRISPR gene editing works.",
          "What is the time complexity of quicksort?",
          "Who wrote Hamlet?",
          "How do vaccines train the immune system?",
          "Derive the Schwarzschild radius of a black hole."]:
    golden.append({"question": q, "answerable": False})

json.dump(golden, open("eval/golden.json", "w", encoding="utf-8"), indent=2)
print(f"\nWrote {len(golden)} items to eval/golden.json")
print("Now open it, delete bad questions, and check each 'pages' value.")