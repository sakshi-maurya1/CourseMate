"""Usage: uv run python -m eval.run_gen_eval [eval/golden.json]
Exits with code 1 if any quality gate fails."""
import json, os, re, sys, time
from app import config
from app.generate import answer, call_llm, REFUSAL
from app.retrieval import Retriever

JUDGE_MODEL = os.getenv("JUDGE_MODEL", config.LLM_MODEL)
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0.59"))    # USD per 1M tokens: set to current Groq pricing
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0.79"))
GATES = {  # metric: (min or max, threshold)
    "recall@5": (">=", 0.80), "faithfulness": (">=", 0.85), "cited_rate": (">=", 0.90),
    "citation_valid": (">=", 0.90), "refusal_acc": (">=", 0.80), "false_refusal": ("<=", 0.15),
}
CITE = re.compile(r"\[([^\]\[,]+),\s*p\.?\s*(\d+)")
FAITH_SYS = ('You grade whether an ANSWER is supported by the CONTEXT. Split the answer into factual claims '
             'and count how many are fully supported by the context. Ignore citation markers. '
             'Reply ONLY with JSON: {"total": <int>, "supported": <int>}')
REL_SYS = ('Rate how well the ANSWER addresses the QUESTION, from 1 (irrelevant) to 5 (fully answers). '
           'Reply ONLY with JSON: {"score": <int>}')


def avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def judge(system, user):
    time.sleep(2)  # stay under Groq free-tier rate limits
    old = config.LLM_MODEL
    config.LLM_MODEL = JUDGE_MODEL
    try:
        text, _ = call_llm(system, user)
    finally:
        config.LLM_MODEL = old
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main(path):
    golden = json.load(open(path))
    r = Retriever()
    rows = []
    for n, g in enumerate(golden, 1):
        t0 = time.perf_counter()
        passages = r.search(g["question"], "rerank", k=5)
        text, usage = answer(g["question"], passages)
        lat = (time.perf_counter() - t0) * 1000
        refused = REFUSAL.rstrip(".").lower() in text.lower()
        row = {"question": g["question"], "answerable": g["answerable"], "refused": refused,
               "latency_ms": lat, "tok_in": usage["in"], "tok_out": usage["out"], "answer": text}
        if g["answerable"]:
            row["hit"] = any(p["source"] == g["source"] and p["page"] in g["pages"] for p in passages)
            if not refused:
                cites = CITE.findall(text)
                valid = {(p["source"], p["page"]) for p in passages}
                row["cited"] = bool(cites)
                if cites:
                    row["cite_valid"] = avg([(s.strip(), int(pg)) in valid for s, pg in cites])
                ctx = "\n\n".join(p["text"] for p in passages)
                f = judge(FAITH_SYS, f"CONTEXT:\n{ctx}\n\nANSWER:\n{text}")
                if f and f.get("total"):
                    row["faith"] = min(f["supported"] / f["total"], 1.0)
                rel = judge(REL_SYS, f"QUESTION: {g['question']}\n\nANSWER:\n{text}")
                if rel and "score" in rel:
                    row["relevance"] = (rel["score"] - 1) / 4
        rows.append(row)
        print(f"[{n}/{len(golden)}] {'REFUSED' if refused else 'answered'}: {g['question'][:70]}")
        time.sleep(2)

    ans = [x for x in rows if x["answerable"]]
    unans = [x for x in rows if not x["answerable"]]
    given = [x for x in ans if not x["refused"]]
    lats = sorted(x["latency_ms"] for x in rows)
    cost = (sum(x["tok_in"] for x in rows) * PRICE_IN + sum(x["tok_out"] for x in rows) * PRICE_OUT) / 1e6
    m = {
        "recall@5": avg([x["hit"] for x in ans]),
        "faithfulness": avg([x["faith"] for x in given if "faith" in x]),
        "answer_relevance": avg([x["relevance"] for x in given if "relevance" in x]),
        "cited_rate": avg([x["cited"] for x in given]),
        "citation_valid": avg([x["cite_valid"] for x in given if "cite_valid" in x]),
        "refusal_acc": avg([x["refused"] for x in unans]),
        "false_refusal": avg([x["refused"] for x in ans]),
    }
    print("\n| Metric | Value | Gate | Status |\n|---|---|---|---|")
    ok = True
    for k, v in m.items():
        gate = GATES.get(k)
        if gate:
            passed = v >= gate[1] if gate[0] == ">=" else v <= gate[1]
            ok &= passed
            print(f"| {k} | {v:.2f} | {gate[0]} {gate[1]} | {'PASS' if passed else 'FAIL'} |")
        else:
            print(f"| {k} | {v:.2f} | n/a | n/a |")
    print(f"| p95 latency (ms) | {lats[int(0.95 * (len(lats) - 1))]:.0f} | n/a | n/a |")
    print(f"| cost / 100 queries (USD) | {cost / len(rows) * 100:.4f} | n/a | n/a |")

    worst = sorted([x for x in given if "faith" in x], key=lambda x: x["faith"])[:3]
    if worst:
        print("\nLeast faithful answers (read these):")
        for x in worst:
            print(f"- {x['faith']:.2f} | {x['question']}")
    json.dump({"metrics": m, "rows": rows}, open("eval/results.json", "w"), indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.json")