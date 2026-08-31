"""Command-line entry point: `ragcheck evaluate dataset.json [options]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluator import RagEvaluator
from . import report as report_mod


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragcheck",
        description="Evaluate a RAG pipeline's retrieval and generation quality.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run evaluation over a dataset JSON file")
    evaluate.add_argument(
        "dataset",
        type=str,
        help="Path to a JSON file containing a list of query records "
        "(question, retrieved_doc_ids, relevant_doc_ids, retrieved_contexts, answer, ground_truth)",
    )
    evaluate.add_argument("-k", type=int, default=5, help="Cutoff for precision@k/recall@k/ndcg@k (default: 5)")
    evaluate.add_argument("--report", type=str, default=None, help="Write a Markdown report to this path")
    evaluate.add_argument("--chart", type=str, default=None, help="Write a PNG bar chart of aggregate metrics here")
    evaluate.add_argument("--json-out", type=str, default=None, help="Write raw results as JSON to this path")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        return _run_evaluate(args)

    parser.print_help()
    return 1


def _run_evaluate(args: argparse.Namespace) -> int:
    path = Path(args.dataset)
    if not path.exists():
        print(f"error: dataset file not found: {path}", file=sys.stderr)
        return 1

    try:
        dataset = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: could not parse {path} as JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(dataset, list) or not dataset:
        print("error: dataset JSON must be a non-empty list of query records", file=sys.stderr)
        return 1

    evaluator = RagEvaluator(k=args.k)
    result = evaluator.evaluate(dataset)

    print(f"Evaluated {len(result)} quer{'y' if len(result) == 1 else 'ies'}.\n")
    print("Aggregate metrics:")
    for key, value in sorted(result.aggregate.items()):
        print(f"  {key:20s} {value:.3f}")

    if args.report:
        Path(args.report).write_text(report_mod.to_markdown(result))
        print(f"\nMarkdown report written to {args.report}")

    if args.chart:
        report_mod.save_chart(result, args.chart)
        print(f"Chart written to {args.chart}")

    if args.json_out:
        payload = {"aggregate": result.aggregate, "per_query": result.per_query}
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"Raw results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
