"""Supported base LLMs and the per-model conventions derived from them.

Adding a backbone means adding one entry here (plus the matching MODEL_TAG case
in scripts/preprocess/distributed_extract_cot.sh, which cannot import Python).
"""

SUPPORTED_BASE_MODELS = {
    # hf id                     : (output tag,      num_hidden_layers)
    "Qwen/Qwen3-0.6B":          ("qwen0.6b",       28),
    "HuggingFaceTB/SmolLM3-3B": ("smollm3b",       36),
    "allenai/Olmo-3-7B-Think":  ("olmo3-7b-think", 32),
}


def _entry(model: str):
    try:
        return SUPPORTED_BASE_MODELS[model]
    except KeyError:
        raise ValueError(
            f"Unsupported base model {model!r}; supported: {list(SUPPORTED_BASE_MODELS)}"
        ) from None


def model_tag(model: str) -> str:
    """Short tag used in output directory names, e.g. 'qwen0.6b'."""
    return _entry(model)[0]


def num_layers(model: str) -> int:
    return _entry(model)[1]


def output_root(model: str) -> str:
    """Root directory for everything produced with this base model: outputs_<tag>."""
    return f"outputs_{model_tag(model)}"
