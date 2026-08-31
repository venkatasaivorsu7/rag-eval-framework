import json
from pathlib import Path

from ragcheck.cli import main


def _write_dataset(tmp_path: Path) -> Path:
    dataset = [
        {
            "question": "How do I reset MFA?",
            "retrieved_doc_ids": ["kb-1", "kb-2"],
            "relevant_doc_ids": ["kb-1"],
            "retrieved_contexts": ["Reset MFA from the user's authentication methods page."],
            "answer": "Reset MFA from the authentication methods page.",
            "ground_truth": "Go to authentication methods and reset MFA.",
        }
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    return path


def test_cli_evaluate_happy_path(tmp_path, capsys):
    dataset_path = _write_dataset(tmp_path)
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "results.json"

    exit_code = main(["evaluate", str(dataset_path), "--report", str(report_path), "--json-out", str(json_path)])

    assert exit_code == 0
    assert report_path.exists()
    assert "Aggregate Metrics" in report_path.read_text()
    assert json_path.exists()

    payload = json.loads(json_path.read_text())
    assert "aggregate" in payload
    assert "per_query" in payload

    captured = capsys.readouterr()
    assert "Aggregate metrics" in captured.out


def test_cli_missing_dataset_file_returns_error(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.json"
    exit_code = main(["evaluate", str(missing)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_invalid_json_returns_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    exit_code = main(["evaluate", str(bad)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "could not parse" in captured.err
