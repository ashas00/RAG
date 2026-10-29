"""CLI entry point: uv run python -m src <command>."""
import os

import fire
from pydantic import ValidationError
from tqdm import tqdm

from src.evaluation import recall_at_k
from src.generator import Generator
from src.indexer import build_index, load_index, search
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

RAW_DIR = "data/raw"
INDEX_DIR = "data/processed"


def _to_sources(found: list[dict[str, object]]) -> list[MinimalSource]:
    """Convert index chunk metadata into MinimalSource objects."""
    return [
        MinimalSource(
            file_path=str(c["file_path"]),
            first_character_index=int(str(c["start"])),
            last_character_index=int(str(c["end"])),
        )
        for c in found
    ]


def _load_generator() -> Generator | None:
    """Load the LLM, printing an error instead of crashing."""
    try:
        return Generator()
    except Exception as e:  # network, missing deps, corrupted cache...
        print(f"Error: cannot load model: {e}")
        return None


class RagCLI:
    """RAG system command-line interface."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunk data/raw and build the index in data/processed."""
        if not 0 < max_chunk_size <= 2000:
            print("Error: max_chunk_size must be between 1 and 2000")
            return
        n = build_index(RAW_DIR, INDEX_DIR, max_chunk_size)
        print(f"Ingestion complete! Indexed {n} chunks under {INDEX_DIR}/")

    def search(self, query: str, k: int = 5) -> None:
        """Print top-k sources for one query."""
        query = str(query).strip()
        if not query:
            print("Error: empty query")
            return
        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer")
            return
        try:
            retriever, chunks = load_index(INDEX_DIR)
        except (OSError, ValueError):
            print("Error: index not found, run `index` first")
            return
        for c in search(retriever, chunks, query, k):
            print(f"{c['file_path']} [{c['start']}:{c['end']}]")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10) -> None:
        """Search every question of a dataset and save JSON."""
        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer")
            return
        try:
            with open(str(dataset_path), encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except OSError as e:
            print(f"Error: cannot read dataset: {e}")
            return
        except ValidationError:
            print(f"Error: {dataset_path} is not a valid RagDataset JSON")
            return
        try:
            retriever, chunks = load_index(INDEX_DIR)
        except (OSError, ValueError):
            print("Error: index not found, run `index` first")
            return

        results: list[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching"):
            sources = _to_sources(search(retriever, chunks, q.question, k))
            results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources,
            ))

        output = StudentSearchResults(search_results=results, k=k)
        out_path = os.path.join(
            str(save_directory), os.path.basename(str(dataset_path))
        )
        try:
            os.makedirs(str(save_directory), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
        except OSError as e:
            print(f"Error: cannot write results: {e}")
            return
        print(f"Saved student_search_results to {out_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """Answer one query with the LLM."""
        query = str(query).strip()
        if not query:
            print("Error: empty query")
            return
        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer")
            return
        try:
            retriever, chunks = load_index(INDEX_DIR)
        except (OSError, ValueError):
            print("Error: index not found, run `index` first")
            return
        sources = _to_sources(search(retriever, chunks, query, k))
        generator = _load_generator()
        if generator is None:
            return
        print(generator.answer(query, sources))
        print("\nSources:")
        for s in sources:
            print(f"  {s.file_path} "
                  f"[{s.first_character_index}:{s.last_character_index}]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for saved search results."""
        path = str(student_search_results_path)
        try:
            with open(path, encoding="utf-8") as f:
                results = StudentSearchResults.model_validate_json(f.read())
        except OSError as e:
            print(f"Error: cannot read search results: {e}")
            return
        except ValidationError:
            print(f"Error: {path} is not a valid StudentSearchResults JSON")
            return
        print(f"Loaded {len(results.search_results)} questions")
        generator = _load_generator()
        if generator is None:
            return

        answers: list[MinimalAnswer] = []
        for r in tqdm(results.search_results, desc="Answering"):
            answers.append(MinimalAnswer(
                question_id=r.question_id,
                question=r.question,
                retrieved_sources=r.retrieved_sources,
                answer=generator.answer(r.question, r.retrieved_sources),
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k
        )
        out_path = os.path.join(str(save_directory), os.path.basename(path))
        try:
            os.makedirs(str(save_directory), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
        except OSError as e:
            print(f"Error: cannot write answers: {e}")
            return
        print(f"Saved student_search_results_and_answer to {out_path}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Compute recall@1/3/5/10 locally (for own testing)."""
        try:
            with open(str(student_search_results_path),
                      encoding="utf-8") as f:
                student = StudentSearchResults.model_validate_json(f.read())
            with open(str(dataset_path), encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
        except OSError as e:
            print(f"Error: cannot read file: {e}")
            return
        except ValidationError:
            print("Error: invalid JSON format in one of the files")
            return

        truth = {
            q.question_id: q.sources for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }
        found = {r.question_id: r.retrieved_sources
                 for r in student.search_results}
        if not truth:
            print("Error: dataset has no answered questions")
            return
        for k in (1, 3, 5, 10):
            total = sum(
                recall_at_k(found.get(qid, []), gold, k)
                for qid, gold in truth.items()
            )
            print(f"Recall@{k}: {total / len(truth):.3f}")


def main() -> None:
    """Run the Fire CLI."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
