"""Turns an EvalResult into human-readable output: a Markdown report and an
optional PNG bar chart of the aggregate metrics."""

from __future__ import annotations

from .evaluator import EvalResult


def to_markdown(result: EvalResult, title: str = "RAG Evaluation Report") -> str:
    lines = [f"# {title}", ""]

    lines += ["## Aggregate Metrics", "", "| Metric | Score |", "|---|---|"]
    for key, value in sorted(result.aggregate.items()):
        lines.append(f"| {key} | {value:.3f} |")
    lines.append("")

    lines.append(f"## Per-Query Detail ({len(result.per_query)} queries)")
    lines.append("")

    columns = sorted({key for row in result.per_query for key in row.keys()})
    if columns:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in result.per_query:
            cells = []
            for col in columns:
                value = row.get(col, "")
                cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def save_chart(result: EvalResult, path: str) -> None:
    """Writes a bar chart of the aggregate metrics to `path` (PNG). Imports
    matplotlib lazily so importing ragcheck doesn't require it unless you
    actually ask for a chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = sorted(result.aggregate.items())
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("RAG Evaluation — Aggregate Metrics")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
