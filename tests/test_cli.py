"""The CLI must print an error, never a traceback, on bad input."""
import subprocess
import sys
from pathlib import Path

import pytest


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "src", *args],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("args", [
    ["search", ""],
    ["search", "lora", "--k", "0"],
    ["search", "lora", "--k", "abc"],
    ["search", "lora", "--k", "True"],
    ["search", "lora", "--index_dir", "/nonexistent"],
    ["index", "--max_chunk_size", "5000"],
    ["index", "--max_chunk_size", "x"],
    ["index", "--raw_dir", "/nonexistent"],
    ["search_dataset", "--dataset_path", "/nonexistent.json",
     "--save_directory", "/tmp"],
    ["evaluate", "--student_search_results_path", "/nonexistent.json",
     "--dataset_path", "/nonexistent.json"],
])
def test_bad_input_is_a_clean_error(args: list[str]) -> None:
    result = run(*args)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert result.stderr.startswith("Error:")


def test_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    result = run("answer_dataset", "--student_search_results_path",
                 str(bad), "--save_directory", str(tmp_path))
    assert result.returncode == 1
    assert "not a valid StudentSearchResults" in result.stderr
