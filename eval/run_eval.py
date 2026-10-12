"""Usage: python -m eval.run_eval eval/golden.json
Golden format: see golden.example.json (write 20-30 items, incl. 5-10 unanswerable)."""
import json, statistics, sys, time
from app import config
from app.retrieval import Retriever

K = 5


def main(path):
    golden = json.load(open(path, encoding="utf-8"))
    r = Retriever()
    rows = []
    for mode in ["dense", "hybrid", "rerank"]:
        hits, rr, lat, refused_ok, n_unans = 0, [], [], 0, 0
        top_ans, top_unans = [], []
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
                if mode == "rerank":
                    top_ans.append(res[0]["score"])
            elif mode == "rerank":
                n_unans += 1
                top_unans.append((res[0]["score"], g.get("near_topic", False)))
                refused_ok += res[0]["score"] < config.REFUSE_THRESHOLD
        lat.sort()
        rows.append((mode, hits / len(ans), statistics.mean(rr),
                     lat[int(0.95 * (len(lat) - 1))],
                     f"{refused_ok}/{n_unans}" if mode == "rerank" else "n/a"))
    print("| Config | Recall@5 | MRR | p95 retrieval ms | Refusal on unanswerable |")
    print("|---|---|---|---|---|")
    for m, rec, mrr, p95, ref in rows:
        print(f"| {m} | {rec:.2f} | {mrr:.2f} | {p95:.0f} | {ref} |")
    threshold_report(top_ans, top_unans)


def threshold_report(top_ans, top_unans):
    """Top rerank scores per group, and how each candidate REFUSE_THRESHOLD would behave."""
    far = [s for s, near in top_unans if not near]
    near = [s for s, near in top_unans if near]
    print(f"\nreranker: {config.RERANK_MODEL}  current threshold: {config.REFUSE_THRESHOLD}")
    for name, xs in [("answerable", top_ans), ("unanswerable, off-topic", far),
                     ("unanswerable, near-topic", near)]:
        if xs:
            print(f"top score, {name:<25} min {min(xs):.3f}  median {statistics.median(xs):.3f}  max {max(xs):.3f}")
    print("\n| Threshold | Refused off-topic | Refused near-topic | False refusals |")
    print("|---|---|---|---|")
    for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"| {t:.2f} | {sum(s < t for s in far)}/{len(far)} | {sum(s < t for s in near)}/{len(near)} "
              f"| {sum(s < t for s in top_ans)}/{len(top_ans)} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.json")
