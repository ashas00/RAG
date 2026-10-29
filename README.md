*This project has been created as part of the 42 curriculum by <login>.*

# RAG against the machine

## Description

A **Retrieval-Augmented Generation** system that answers questions about
the [vLLM](https://github.com/vllm-project/vllm) 0.10.1 codebase.

1. **Index**: every `.py`, `.md` and `.txt` file of `data/raw/` is split
   into chunks of at most 2000 characters and stored in a BM25 index.
2. **Retrieve**: a question is matched against the index; the top-k chunks
   are returned as source locations (file path + character range).
3. **Augment**: the text of those chunks is put in the prompt.
4. **Generate**: `Qwen/Qwen3-0.6B` writes an answer grounded in them.

Retrieval quality is measured with recall@k (a gold source counts as found
when a retrieved chunk of the same file overlaps it with IoU >= 0.05).

The whole project runs on CPU **without PyTorch**: the language model is
run by a small NumPy implementation of the Qwen3 forward pass
(`src/qwen.py`) that reads the official weights.

## Instructions

Requirements: Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make install          # uv sync
```

Expected layout (data is not committed):

```
data/raw/vllm-0.10.1/                      # the corpus
data/datasets/{UnansweredQuestions,AnsweredQuestions}/*.json
data/processed/                            # created by `index`
data/output/search_results/<Scope>/        # created by `search_dataset`
data/output/search_results_and_answer/<Scope>/
```

Commands (`uv run python -m src <command> [options]`):

| Command | Options | Does |
|---|---|---|
| `index` | `--max_chunk_size 2000` | chunk `data/raw/`, save index to `data/processed/` |
| `search QUERY` | `--k 5` | print the top-k sources |
| `search_dataset` | `--dataset_path --save_directory --k 10` | write a `StudentSearchResults` JSON |
| `answer QUERY` | `--k 5` | print an answer and its sources |
| `answer_dataset` | `--student_search_results_path --save_directory` | write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate` | `--student_search_results_path --dataset_path` | print recall@1/3/5/10 |

`index`, `search`, `search_dataset` and `answer` also accept `--raw_dir` /
`--index_dir` to use other locations.

The first `answer`/`answer_dataset` call downloads `Qwen/Qwen3-0.6B`
(about 1.5 GB) into the Hugging Face cache.

Makefile: `install`, `run` (index + search the docs dataset), `debug`
(pdb), `clean`, `lint`, `lint-strict`, `test`.

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000
# Ingestion complete! Indexed 13175 chunks under data/processed/

uv run python -m src search "How to configure the OpenAI server?" --k 3
# data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0:1972]
# ...

uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Bad input (empty query, `--k 0`, missing or malformed file, no index...)
prints a one-line `Error: ...` on stderr and exits with code 1.

## System architecture

```
src/
  __main__.py   Fire CLI, argument checks, JSON in/out, error handling
  models.py     pydantic models exchanged between stages
  chunker.py    Python and Markdown/text chunking
  indexer.py    tokenizer + build and save the BM25 index
  retriever.py  load the index, score, re-rank, return MinimalSource
  generator.py  prompt building (context budget) + answer generation
  qwen.py       Qwen3 forward pass and greedy decoding in NumPy
  evaluation.py IoU and recall@k
tests/          pytest suite
```

```
data/raw ─▶ chunker ─▶ indexer ─▶ data/processed (bm25/ + chunks.json)
                                        │
question ─▶ retriever ◀─────────────────┘
               │ top-k MinimalSource
               ▼
           generator ─▶ reads the spans from data/raw ─▶ prompt ─▶ Qwen3
               │
               ▼
           answer (JSON)
```

Only the spans are stored in `chunks.json`; the generator reads the text
back from the corpus, so the index stays small.

## Chunking strategy

Both strategies first list *cut points* (offsets where cutting is
natural), then greedily merge the pieces between them into chunks of at
most `max_chunk_size` characters. A piece that is still too long is cut at
its last newline that fits (or hard-cut if there is none). Chunks never
overlap and every chunk is <= 2000 characters, as the moulinette requires.

- **Python** (`.py`): cut points are the start of every top-level
  statement and of every statement inside a class body (so methods), found
  with `ast`. Decorators stay with their function or class. A file that
  does not parse falls back to blank lines.
- **Markdown / text** (`.md`, `.txt`): cut points are headings (`# ...`)
  and paragraph breaks (blank lines).

Effect of the chunk size (Recall@5, public datasets):

| max_chunk_size | docs | code |
|---|---|---|
| 1000 | 0.83 | 0.81 |
| 1500 | 0.85 | 0.85 |
| 2000 | 0.87 | 0.84 |

Bigger chunks hold more words for BM25 to match and still overlap the gold
span enough (IoU >= 0.05), so 2000 is the default.

## Retrieval method

**BM25** (library `bm25s`, defaults k1=1.5, b=0.75).

Tokenizer: lowercase alphanumeric words, plus the parts of `snake_case`
and `CamelCase` identifiers (`load_lora_adapter` gives `load_lora_adapter`,
`load`, `lora`, `adapter`). The full identifier lets a question that quotes
it match exactly; the parts let a paraphrase match. Very common English
words and the word "vllm" (present everywhere) are dropped.

Each chunk is indexed with the words of its **file path** too:
`docs/features/lora.md` adds `docs`, `features`, `lora`, `md`.

Ranking: BM25 gives the 50 best candidates; chunks from `tests/` and
`benchmarks/` keep 60% of their score (they repeat the words of the code
they test and often outrank the implementation); the k best are returned.
Chunks with a score of 0 (no common word) are never returned.

## Performance analysis

Measured on 4 CPU cores with the provided moulinette (`--k 10`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|
| docs (100 q.) | 0.63 | 0.84 | **0.87** | 0.91 | 0.80 |
| code (99 q.) | 0.55 | 0.78 | **0.83** | 0.86 | 0.50 |

- Indexing the whole corpus: about **6 s** (limit: 5 min).
- `search_dataset` on 100 questions: about **0.3 s** (limit: 90 s / 200).
- Index on disk: about 30 MB.

Answer generation (NumPy Qwen3-0.6B, float32, same 4 cores):

- Model load: about 10-15 s (download of ~1.5 GB the first time).
- Prompt processing: about 6 s for 1000 tokens, 20 s for 2300 tokens
  (the prompt is processed in blocks of 256 tokens with the cache).
- Decoding: about 13 tokens/s.
- So one answer takes roughly 15-35 s and `answer_dataset` on 100
  questions about 30-50 minutes. RAM: about 3.5 GB.

What each change brought (Recall@5 docs / code):

| Change | docs | code |
|---|---|---|
| BM25 on `.py` + `.md`, path words | 0.83 | 0.82 |
| + index `.txt` (`CMakeLists.txt` is a gold source) | 0.86 | 0.82 |
| + tests/benchmarks weight 0.6 | 0.87 | 0.83 |

## Design decisions

- **BM25 over TF-IDF**: it saturates repeated words and normalises by
  chunk length, which suits chunks of very different sizes.
- **Spans, not text, in the index**: smaller index, and answers are always
  built from the real file content.
- **No PyTorch**: `torch` + `transformers` are several GB. Qwen3 is a
  plain decoder-only transformer, so its forward pass fits in ~270 lines
  of NumPy: RMSNorm, grouped-query attention with per-head q/k norm, RoPE,
  SwiGLU MLP, a key/value cache, and greedy decoding. Weights are the
  official `Qwen/Qwen3-0.6B` safetensors (bf16, converted to float32).
  The tokenizer is the official `tokenizer.json` loaded with `tokenizers`.
  The implementation was checked against Hugging Face `transformers` on
  random Qwen3 models (fp32 and bf16 weights, tied and untied embeddings):
  logits agree to 1e-6 and greedy decoding gives the same tokens.
- **Prompt**: system prompt asks to answer only from the context, briefly,
  and to say when the context is not enough. Sources are added best
  first, each with its file path, until about 8000 characters (~2500
  tokens). Thinking mode is turned off (empty `<think>` block) because the
  0.6B model rambles otherwise. Greedy decoding with a light repetition
  penalty (1.1) avoids loops.
- **Errors**: each command validates its arguments; expected errors become
  a `CLIError` printed in one line; `main()` also catches anything
  unexpected so the CLI never shows a traceback.

## Challenges faced

- **Docs questions about non-Markdown files**: some gold sources are in
  `CMakeLists.txt`; indexing `.txt` fixed them.
- **Tests outranking code**: test files mention the same identifiers as
  the implementation many times; down-weighting them fixed several code
  questions without hiding them completely.
- **Questions vs code vocabulary**: questions paraphrase, code uses
  identifiers. Splitting identifiers and adding path words bridged most of
  the gap.
- **Heavy deep-learning stack**: replaced by the NumPy runtime; the
  key/value cache keeps decoding fast enough on CPU, and a test checks
  that cached decoding gives the same logits as a full forward pass.

## Resources

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks* (2020) — https://arxiv.org/abs/2005.11401
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009)
- `bm25s` — https://github.com/xhluca/bm25s
- Qwen3 technical report — https://arxiv.org/abs/2505.09388 and model card
  https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face `transformers` Qwen3 implementation (reference for the
  NumPy port) — `models/qwen3/modeling_qwen3.py`
- Su et al., *RoFormer: Rotary Position Embedding* —
  https://arxiv.org/abs/2104.09864
- safetensors format — https://github.com/huggingface/safetensors
- Python Fire — https://github.com/google/python-fire

### Use of AI

An AI assistant (Claude) was used to:
- review the existing code and list what was missing against the subject;
- refactor the CLI error handling and write the pytest suite;
- write the NumPy Qwen3 runtime and check it against `transformers`;
- run the recall experiments (file types, folder weights, chunk sizes);
- draft this README.

All code was read, tested and is understood by the author.
