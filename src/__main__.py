"""CLI entry point: uv run python -m src <command> [options]."""
import os
import sys
from typing import TypeVar

import fire
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.evaluation import recall_at_k
from src.generator import Generator
from src.indexer import build_index
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import Retriever

RAW_DIR = "data/raw"
INDEX_DIR = "data/processed"
MAX_CHUNK_SIZE = 2000  # the moulinette rejects longer sources

Model = TypeVar("Model", bound=BaseModel)


class CLIError(Exception):
    """A user-facing error: printed as one line, never as a traceback."""


def _check_positive_int(value: object, name: str, maximum: int = 0) -> int:
    """Return value if it is an int in [1, maximum] (no max if 0)."""
    # bool is a subclass of int, but `--k True` makes no sense.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CLIError(f"{name} must be a positive integer, got {value!r}")
    if maximum and value > maximum:
        raise CLIError(f"{name} must be at most {maximum}, got {value}")
    return value


def _check_query(query: object) -> str:
    """Return the query as a non-empty string."""
    text = str(query).strip()
    if not text:
        raise CLIError("empty query")
    return text


def _read_json(path: object, model: type[Model]) -> Model:
    """Read and validate a JSON file with a pydantic model."""
    try:
        with open(str(path), encoding="utf-8") as f:
            return model.model_validate_json(f.read())
    except OSError as e:
        raise CLIError(f"cannot read {path}: {e.strerror}") from e
    except ValidationError as e:
        raise CLIError(
            f"{path} is not a valid {model.__name__} JSON "
            f"({e.error_count()} errors)"
        ) from e


def _write_json(data: BaseModel, directory: object, file_name: str) -> str:
    """Write data as indented JSON to directory/file_name."""
    path = os.path.join(str(directory), file_name)
    try:
        os.makedirs(str(directory), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=2))
    except OSError as e:
        raise CLIError(f"cannot write {path}: {e.strerror}") from e
    return path


def _load_retriever(index_dir: str) -> Retriever:
    """Load the index or explain how to build it."""
    try:
        return Retriever(index_dir)
    except Exception as e:  # missing files, corrupted JSON or arrays...
        raise CLIError(
            f"no valid index in {index_dir}, run `index` first ({e})"
        ) from e


def _load_generator() -> Generator:
    """Load the LLM (downloads it on first use)."""
    try:
        return Generator()
    except Exception as e:  # no network, disk full, corrupted cache...
        raise CLIError(f"cannot load the language model: {e}") from e


def _print_sources(sources: list[MinimalSource]) -> None:
    """Print one source per line: path [first:last]."""
    for s in sources:
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")


class RagCLI:
    """RAG over the vLLM codebase: index, search, answer, evaluate."""

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE,
              raw_dir: str = RAW_DIR, index_dir: str = INDEX_DIR) -> None:
        """Chunk every file of raw_dir and save a BM25 index in index_dir."""
        size = _check_positive_int(max_chunk_size, "max_chunk_size",
                                   MAX_CHUNK_SIZE)
        if not os.path.isdir(str(raw_dir)):
            raise CLIError(f"corpus directory {raw_dir} not found")
        count = build_index(str(raw_dir), str(index_dir), size)
        if count == 0:
            raise CLIError(f"no file to index in {raw_dir}")
        print(f"Ingestion complete! Indexed {count} chunks "
              f"under {index_dir}/")

    def search(self, query: str, k: int = 5,
               index_dir: str = INDEX_DIR) -> None:
        """Print the top-k sources for one query."""
        text = _check_query(query)
        k = _check_positive_int(k, "k")
        sources = _load_retriever(str(index_dir)).search(text, k)
        if not sources:
            print("No relevant source found.")
        _print_sources(sources)

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10, index_dir: str = INDEX_DIR) -> None:
        """Search every question of a dataset; save StudentSearchResults."""
        k = _check_positive_int(k, "k")
        dataset = _read_json(dataset_path, RagDataset)
        retriever = _load_retriever(str(index_dir))
        results = [
            MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=retriever.search(q.question, k),
            )
            for q in tqdm(dataset.rag_questions, desc="Searching")
        ]
        output = StudentSearchResults(search_results=results, k=k)
        path = _write_json(output, save_directory,
                           os.path.basename(str(dataset_path)))
        print(f"Saved student_search_results to {path}")

    def answer(self, query: str, k: int = 5,
               index_dir: str = INDEX_DIR) -> None:
        """Answer one query with Qwen3 using the top-k sources."""
        text = _check_query(query)
        k = _check_positive_int(k, "k")
        sources = _load_retriever(str(index_dir)).search(text, k)
        print(_load_generator().answer(text, sources))
        print("\nSources:")
        _print_sources(sources)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Answer every question of a search results file."""
        path = str(student_search_results_path)
        results = _read_json(path, StudentSearchResults)
        total = len(results.search_results)
        print(f"Loaded {total} questions")
        generator = _load_generator()
        answers: list[MinimalAnswer] = []
        for r in tqdm(results.search_results, desc="Answering"):
            answers.append(MinimalAnswer(
                question_id=r.question_id,
                question=r.question,
                retrieved_sources=r.retrieved_sources,
                answer=generator.answer(r.question, r.retrieved_sources),
            ))
        print(f"Processed {len(answers)} of {total} questions")
        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=results.k)
        out = _write_json(output, save_directory, os.path.basename(path))
        print(f"Saved student_search_results_and_answer to {out}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print recall@1/3/5/10 against a ground-truth dataset."""
        student = _read_json(student_search_results_path,
                             StudentSearchResults)
        dataset = _read_json(dataset_path, RagDataset)
        truth = {
            q.question_id: q.sources for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }
        if not truth:
            raise CLIError(f"{dataset_path} has no answered question")
        found = {r.question_id: r.retrieved_sources
                 for r in student.search_results}
        print(f"Questions evaluated: {len(truth)}")
        for k in (1, 3, 5, 10):
            total = sum(recall_at_k(found.get(qid, []), gold, k)
                        for qid, gold in truth.items())
            print(f"Recall@{k}: {total / len(truth):.3f}")


def main() -> None:
    """Run the Fire CLI; turn every error into a one-line message."""
    try:
        fire.Fire(RagCLI)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # last safety net: never show a traceback
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
