import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def term_coverage(text: str, terms: list[str]) -> float | None:
    if not terms:
        return None
    normalized = _normalized(text)
    matched = [
        any(_normalized(option) in normalized for option in term.split("|"))
        for term in terms
    ]
    return sum(matched) / len(matched)


def evaluate_case(case, db, engine, llm, top_k=6):
    start = time.perf_counter()
    query_vector = engine.generate_embeddings([case["question"]])[0]
    embedding_seconds = time.perf_counter() - start

    start = time.perf_counter()
    retrieved = db.search(query_vector, limit=top_k)
    retrieval_seconds = time.perf_counter() - start

    context = "\n".join(item.content for item in retrieved)
    start = time.perf_counter()
    answer = llm.generate_text(
        "Answer only from the provided document context. If it does not contain "
        "the answer, say that clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {case['question']}"
    )
    generation_seconds = time.perf_counter() - start

    expected_pages = set(case["expected_pages"])
    retrieved_pages = {
        item.metadata.get("page")
        for item in retrieved
        if item.metadata.get("page") is not None
    }
    recall = (
        len(expected_pages & retrieved_pages) / len(expected_pages)
        if expected_pages
        else None
    )
    expected_terms = case.get("expected_terms", [])
    context_coverage = term_coverage(context, expected_terms)
    answer_coverage = term_coverage(answer, expected_terms)
    refusal_markers = (
        "not contain",
        "does not mention",
        "don't have",
        "do not have",
        "does not provide",
        "not provided",
        "not specified",
        "not stated",
        "i don't know",
        "i do not know",
        "cannot determine",
        "unable to determine",
        "not enough information",
    )
    refusal_detected = any(marker in _normalized(answer) for marker in refusal_markers)

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "recall_at_k": recall,
        "evidence_term_coverage": context_coverage,
        "answer_term_coverage": answer_coverage,
        "expects_refusal": bool(case.get("expects_refusal", False)),
        "refusal_detected": refusal_detected,
        "embedding_seconds": embedding_seconds,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "sources": [
            {"page": item.metadata.get("page"), "text": item.content}
            for item in retrieved
        ],
    }


def run_configuration(
    pdf_path, cases, chunk_size, chunk_overlap, embedding_model, settings
):
    from db.sqlite_db import SQLiteVectorDB
    from document_parser import DocumentParser
    from embedding_engine import EmbeddingEngine
    from services import build_llm

    parser = DocumentParser(str(pdf_path), settings=settings)
    pages = parser.clean_documents()
    chunks = parser.split_documents(chunk_size, chunk_overlap)
    if not chunks:
        raise RuntimeError("Evaluation PDF produced no text chunks.")

    engine = EmbeddingEngine(model_name=embedding_model, settings=settings)
    embed_start = time.perf_counter()
    vectors = engine.generate_embeddings([chunk.page_content for chunk in chunks])
    embedding_seconds = time.perf_counter() - embed_start
    llm = build_llm(settings)

    with tempfile.TemporaryDirectory(prefix="localrag-eval-") as temp_dir:
        db = SQLiteVectorDB(str(Path(temp_dir) / "evaluation.sqlite"))
        try:
            rows = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                metadata = {
                    key: chunk.metadata.get(key)
                    for key in ("document_id", "document_name", "page", "chunk")
                }
                rows.append((chunk.page_content, metadata, vector))
            db.insert_documents(rows)
            case_results = [evaluate_case(case, db, engine, llm) for case in cases]
        finally:
            db.close()

    recall_scores = [
        r["recall_at_k"] for r in case_results if r["recall_at_k"] is not None
    ]
    evidence_scores = [
        r["evidence_term_coverage"]
        for r in case_results
        if r["evidence_term_coverage"] is not None
    ]
    answer_scores = [
        r["answer_term_coverage"]
        for r in case_results
        if r["answer_term_coverage"] is not None
    ]
    negative_cases = [r for r in case_results if r["expects_refusal"]]
    return {
        "configuration": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "llm_model": settings.llm_model,
            "top_k": 6,
            "document": str(pdf_path.relative_to(ROOT)),
            "pages": len(pages),
            "chunks": len(chunks),
        },
        "summary": {
            "mean_recall_at_k": _mean(recall_scores),
            "mean_evidence_term_coverage": _mean(evidence_scores),
            "mean_answer_term_coverage": _mean(answer_scores),
            "negative_case_refusal_rate": _mean(
                [float(r["refusal_detected"]) for r in negative_cases]
            ),
            "index_embedding_seconds": embedding_seconds,
            "mean_query_embedding_seconds": _mean(
                [r["embedding_seconds"] for r in case_results]
            ),
            "mean_retrieval_seconds": _mean(
                [r["retrieval_seconds"] for r in case_results]
            ),
            "mean_generation_seconds": _mean(
                [r["generation_seconds"] for r in case_results]
            ),
        },
        "cases": case_results,
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def main(argv=None):
    from settings import get_settings

    parser = argparse.ArgumentParser(
        description="Evaluate LocalRAG on the bundled benchmark."
    )
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[1000])
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--embedding-models", nargs="+", default=["all-MiniLM-L6-v2"])
    parser.add_argument("--output", type=Path, default=Path("evaluation-report.json"))
    args = parser.parse_args(argv)
    if any(
        size <= 0 or args.chunk_overlap < 0 or args.chunk_overlap >= size
        for size in args.chunk_sizes
    ):
        parser.error(
            "Each chunk size must be positive and larger than --chunk-overlap."
        )

    cases = json.loads((ROOT / "evaluation" / "questions.json").read_text())
    try:
        settings = get_settings()
    except Exception as exc:
        parser.error(f"Invalid application configuration: {exc}")
    if settings.llm_model == "dummy":
        parser.error(
            "Configure a real LLM_MODEL in .env before running the evaluation."
        )
    report = {"runs": []}
    try:
        for chunk_size in args.chunk_sizes:
            for embedding_model in args.embedding_models:
                report["runs"].append(
                    run_configuration(
                        ROOT / "samples" / "sample.pdf",
                        cases,
                        chunk_size,
                        args.chunk_overlap,
                        embedding_model,
                        settings,
                    )
                )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
