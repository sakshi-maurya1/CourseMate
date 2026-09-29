"""Usage: python -m eval.run_eval eval/golden.json
Golden format: see golden.example.json (write 20-30 items, incl. 5-10 unanswerable)."""
import json, statistics, sys, time
from app import config
from app.retrieval import Retriever

K = 5


def main(path):
    golden = json.load(open(path))
    r = Retriever()
    rows = []
    for mode in ["dense", "hybrid", "rerank"]:
        hits, rr, lat, refused_ok, n_unans = 0, [], [], 0, 0
        ans = [g for g in golden if g["answerable"]]
        for g in golden:
            t = time.perf_counter()
            res = r.search(g["question"], mode, k=K)
            lat.append((time.perf_counter() - t) * 1000)
            if g["answerable"]:
                rank = next((i + 1 for i, p in enumerate(res)
                             if p["source"] == g["source"] and p["page"] in g["pages"]), None)
                hits += rank is not None
                rr.append(1 / rank if rank else 0)
            elif mode == "rerank":
                n_unans += 1
                refused_ok += res[0]["score"] < config.REFUSE_THRESHOLD
        lat.sort()
        rows.append((mode, hits / len(ans), statistics.mean(rr),
                     lat[int(0.95 * (len(lat) - 1))],
                     f"{refused_ok}/{n_unans}" if mode == "rerank" else "n/a"))
    print("| Config | Recall@5 | MRR | p95 retrieval ms | Refusal on unanswerable |")
    print("|---|---|---|---|---|")
    for m, rec, mrr, p95, ref in rows:
        print(f"| {m} | {rec:.2f} | {mrr:.2f} | {p95:.0f} | {ref} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.json")
