PYTHON = uv run python
DOCS = data/datasets/UnansweredQuestions/dataset_docs_public.json
RESULTS = data/output/search_results/UnansweredQuestions

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

# Full retrieval pipeline on the public docs dataset.
run:
	$(PYTHON) -m src index --max_chunk_size 2000
	$(PYTHON) -m src search_dataset --dataset_path $(DOCS) --k 10 \
		--save_directory $(RESULTS)

debug:
	$(PYTHON) -m pdb -m src index --max_chunk_size 2000

clean:
	find . -path ./.venv -prune -o -type d \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) \
		-exec rm -rf {} +

lint:
	uv run flake8 . --exclude .venv,data
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude .venv,data
	uv run mypy . --strict

test:
	uv run pytest -q
