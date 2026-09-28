"""Benchmark queries demonstrating document grounding.

Run against the in-process graph (default) or a running API:

    python tests_sample_queries.py
    python tests_sample_queries.py --api-url http://localhost:8000
    python tests_sample_queries.py --output sample_results.json

Exit code is non-zero if an in-scope question is refused or the
out-of-scope question is answered.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

# (query, expect_grounded_answer)
SAMPLE_QUERIES: list[tuple[str, bool]] = [
    ("What is Agentic AI according to the eBook?", True),
    ("How do AI agents differ from traditional automation systems?", True),
    ("What are the core components of an Agentic Architecture?", True),
    ("What role does memory play in Agentic AI workflows?", True),
    ("What challenges or risks of Agentic AI does the eBook discuss?", True),
    ("Who won the 2022 FIFA World Cup?", False),  # must be refused
]


def query_api(api_url: str, query: str) -> dict:
    import requests

    response = requests.post(f"{api_url.rstrip('/')}/chat", json={"query": query}, timeout=120)
    response.raise_for_status()
    return response.json()


def query_graph(graph, query: str) -> dict:
    state = graph.invoke({"question": query})
    cited = set(state.get("cited_chunks", []))
    return {
        "answer": state["answer"],
        "retrieved_chunks": [
            {
                "rank": c["rank"],
                "page": c["page"],
                "similarity_score": c["similarity"],
                "cited": c["rank"] in cited,
                "text": c["text"],
            }
            for c in state.get("context", [])
        ],
        "confidence_score": state.get("score", 0.0),
        "retrieval_score": state.get("retrieval_score", 0.0),
        "grounded": state.get("grounded", False),
    }


def print_result(index: int, query: str, expected: bool, result: dict) -> bool:
    passed = result["grounded"] == expected
    status = "PASS" if passed else "FAIL"
    print("=" * 100)
    print(f"Q{index}: {query}")
    print(f"[{status}] expected {'grounded answer' if expected else 'refusal'} | "
          f"confidence={result['confidence_score']:.3f} retrieval={result['retrieval_score']:.3f}")
    print("-" * 100)
    print(textwrap.fill(result["answer"], width=100, replace_whitespace=False))
    print("-" * 100)
    for chunk in result["retrieved_chunks"]:
        snippet = " ".join(chunk["text"].split())[:140]
        marker = "*" if chunk.get("cited") else " "
        print(f" {marker}[{chunk['rank']}] p.{chunk['page']} sim={chunk['similarity_score']:.3f}  {snippet}…")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", help="base URL of a running FastAPI server; omit to run in-process")
    parser.add_argument("--output", help="write all results to this JSON file")
    args = parser.parse_args()

    if args.api_url:
        run = lambda q: query_api(args.api_url, q)  # noqa: E731
    else:
        from src.graph import build_rag_graph

        graph = build_rag_graph()
        run = lambda q: query_graph(graph, q)  # noqa: E731

    results, passed = [], 0
    for i, (query, expected) in enumerate(SAMPLE_QUERIES, start=1):
        result = run(query)
        passed += print_result(i, query, expected, result)
        results.append({"query": query, "expect_grounded": expected, **result})

    print("=" * 100)
    print(f"{passed}/{len(SAMPLE_QUERIES)} queries behaved as expected ('*' = chunk cited in answer)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")

    return 0 if passed == len(SAMPLE_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
