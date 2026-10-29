install:
	uv sync
run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ */__pycache__ .mypy_cache .pytest_cache

lint:
	uv run flake8 . --exclude .venv,data
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude .venv,data
	uv run mypy . --strict
