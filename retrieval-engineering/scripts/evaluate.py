"""
scripts/evaluate.py
----------------------
CLI: run the retrieval evaluation dataset against the local index and
print Recall@1/3/5 and MRR.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_engineering.embeddings.embedder import Embedder
from retrieval_engineering.evaluation.metrics import evaluate, load_questions
from retrieval_engineering.retrieval.semantic_retriever import SemanticRetriever
from retrieval_engineering.vector_store.local_store import LocalVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "index" / "kb"
DEFAULT_QUESTIONS_PATH = REPO_ROOT / "evaluation" / "retrieval_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    args = parser.parse_args()

    store = LocalVectorStore.load(args.index_path)
    retriever = SemanticRetriever(embedder=Embedder(), vector_store=store)
    questions = load_questions(args.questions_path)

    report = evaluate(questions, retrieve_fn=lambda q, k: retriever.retrieve(q, top_k=k), k_values=[1, 3, 5])

    print(f"Evaluated {report['summary']['num_questions']} question(s)\n")
    for key, value in report["summary"].items():
        if key != "num_questions":
            print(f"{key}: {value:.3f}")

    print("\nPer-question detail:")
    for row in report["per_question"]:
        if row["recall@1"] == 1.0:
            tag = "OK "
        elif row["recall@5"] == 1.0:
            tag = "~  "
        else:
            tag = "MISS"
        print(f"  [{tag}] {row['question']!r} -> expected {row['expected_sources']}, got {row['retrieved_sources'][:3]}")


if __name__ == "__main__":
    main()
