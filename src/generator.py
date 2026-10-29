"""Answer generation with Qwen3-0.6B from retrieved sources."""
from src.models import MinimalSource
from src.qwen import Qwen3

MODEL_NAME = "Qwen/Qwen3-0.6B"
# Roughly 2000-2500 tokens: enough context, still fast enough on CPU.
MAX_CONTEXT_CHARS = 8000
MAX_NEW_TOKENS = 256
NO_CONTEXT_ANSWER = "I could not find relevant information in the codebase."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM codebase. "
    "Use ONLY the provided context. If the context does not contain the "
    "answer, say that you don't know. Answer in 1-4 short sentences and "
    "mention exact names (functions, flags, endpoints) when relevant."
)


def read_source(source: MinimalSource) -> str:
    """Return the text of a source span, or "" if the file is unreadable."""
    try:
        with open(source.file_path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return ""
    return text[source.first_character_index:source.last_character_index]


def build_context(sources: list[MinimalSource],
                  max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join source snippets (best first) into one string within budget."""
    parts: list[str] = []
    total = 0
    for i, source in enumerate(sources, 1):
        snippet = read_source(source)
        if not snippet.strip():
            continue
        block = f"[{i}] {source.file_path}\n{snippet}"
        if parts and total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Qwen3 chat format, with the empty <think> block that turns off
    the model's "thinking" mode (same as enable_thinking=False)."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\n"
        f"Question: {question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


class Generator:
    """Turns (question, sources) into an answer with Qwen3."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Download the model on first run (cached afterwards) and load it."""
        # Imported here so index/search never need these packages.
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        model_dir = snapshot_download(
            model_name,
            allow_patterns=["config.json", "tokenizer.json", "*.safetensors"],
        )
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.model = Qwen3(model_dir)
        self.stop_ids = {
            i for i in (self.tokenizer.token_to_id("<|im_end|>"),
                        self.tokenizer.token_to_id("<|endoftext|>"))
            if i is not None
        }

    def answer(self, question: str, sources: list[MinimalSource],
               max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Generate an answer grounded in the given sources."""
        context = build_context(sources)
        if not context:
            return NO_CONTEXT_ANSWER
        prompt = build_prompt(question, context)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        out = self.model.generate(ids, max_new_tokens, self.stop_ids,
                                  repetition_penalty=1.1)
        text: str = self.tokenizer.decode(out, skip_special_tokens=True)
        return text.strip()
