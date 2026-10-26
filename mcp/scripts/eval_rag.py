"""Offline retrieval evaluation against tests/rag_golden.jsonl.

    python -m scripts.eval_rag                  # all golden sets, bm25 / semantic / hybrid
    python -m scripts.eval_rag --golden bitext  # one golden set (hand | bitext)
    python -m scripts.eval_rag --mode hybrid -v # one mode, list misses
    python -m scripts.eval_rag --min-mrr 0.8    # exit 1 if hybrid MRR is lower (CI gate)

A chunk counts as relevant when its first line (the FAQ question) is in the
query's `relevant` list. The KB has near-duplicate entries, so several
headings can be acceptable for one query. Queries with an empty `relevant`
list are unanswerable and are reported separately.
"""

import argparse
import json
import sys
from pathlib import Path

from app.rag.documents import load_policy_chunks
from app.rag.retriever import retrieve_context

TESTS = Path(__file__).resolve().parents[1] / "tests"
GOLDEN_FILES = {"hand": TESTS / "rag_golden.jsonl", "bitext": TESTS / "rag_golden_bitext.jsonl"}
MODES = ("bm25", "semantic", "hybrid")
K = 5


def heading(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def load_golden(names: list[str]) -> list[dict]:
    rows = [
        json.loads(line)
        for name in names
        for line in GOLDEN_FILES[name].read_text().splitlines()
        if line.strip()
    ]
    known = {heading(text) for text, _ in load_policy_chunks()}
    unknown = [h for row in rows for h in row["relevant"] if h not in known]
    if unknown:
        sys.exit(f"Golden set references headings not in the KB: {sorted(set(unknown))}")
    return rows


def evaluate(mode: str, rows: list[dict], verbose: bool) -> dict:
    hit1 = hit3 = hit5 = rr = 0.0
    answerable = [r for r in rows if r["relevant"]]
    unanswerable = [r for r in rows if not r["relevant"]]
    misses = []

    for row in answerable:
        result = retrieve_context(row["query"], k=K, mode=mode)
        ranked = [heading(c.text) for c in result.chunks]
        rank = next((i for i, h in enumerate(ranked, 1) if h in row["relevant"]), None)
        if rank:
            hit1 += rank == 1
            hit3 += rank <= 3
            hit5 += rank <= 5
            rr += 1 / rank
        else:
            misses.append((row["query"], ranked[:3]))
        if verbose and rank != 1:
            print(f"  [{mode}] rank={rank} {row['query']!r} -> {ranked[:3]}")

    top_scores = []
    for row in unanswerable:
        result = retrieve_context(row["query"], k=K, mode=mode)
        top_scores.append(result.chunks[0].score if result.chunks else 0.0)

    n = len(answerable)
    return {
        "mode": mode,
        "n": n,
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "hit@5": hit5 / n,
        "mrr": rr / n,
        "unanswerable_top_score_max": max(top_scores, default=0.0),
        "unanswerable_returned_chunks": sum(s > 0 for s in top_scores),
        "unanswerable_n": len(unanswerable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--golden", choices=sorted(GOLDEN_FILES), action="append",
                        help="golden set(s) to use; default: all")
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    rows = load_golden(args.golden or sorted(GOLDEN_FILES))
    modes = [args.mode] if args.mode else list(MODES)
    results = [evaluate(m, rows, args.verbose) for m in modes]

    print(f"\n{len(rows)} queries ({results[0]['n']} answerable), k={K}")
    print(f"{'mode':<10}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}{'unans.max':>11}{'unans.ret':>11}")
    for r in results:
        print(
            f"{r['mode']:<10}{r['hit@1']:>8.3f}{r['hit@3']:>8.3f}{r['hit@5']:>8.3f}{r['mrr']:>8.3f}"
            f"{r['unanswerable_top_score_max']:>11.3f}"
            f"{r['unanswerable_returned_chunks']:>8}/{r['unanswerable_n']}"
        )

    if args.min_mrr is not None:
        gated = results[-1]
        if gated["mrr"] < args.min_mrr:
            print(f"FAIL: {gated['mode']} MRR {gated['mrr']:.3f} < {args.min_mrr}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
