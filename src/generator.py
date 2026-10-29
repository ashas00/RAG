"""Answer generation with Qwen3-0.6B from retrieved sources."""
from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
# ~2000-2500 tokens of context: fits the model and keeps CPU speed sane.
MAX_CONTEXT_CHARS = 8000

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


def build_context(sources: list[MinimalSource]) -> str:
    """Join source snippets into one context string within the budget."""
    parts: list[str] = []
    total = 0
    for i, source in enumerate(sources, 1):
        snippet = read_source(source)
        if not snippet.strip():
            continue
        block = f"[{i}] {source.file_path}\n{snippet}"
        if parts and total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


class Generator:
    """Wraps the Qwen model and turns (question, sources) into an answer."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load tokenizer and model (downloads them on first run)."""
        # Imported here: torch/transformers are slow to import, and
        # index/search commands must not pay that cost.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        )
        self.model.eval()

    def answer(self, question: str, sources: list[MinimalSource],
               max_new_tokens: int = 256) -> str:
        """Generate an answer grounded in the given sources."""
        import torch

        context = build_context(sources)
        if not context:
            return "I could not find relevant information in the codebase."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: no <think> reasoning block
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.split("</think>")[-1].strip()
