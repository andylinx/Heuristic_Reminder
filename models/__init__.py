"""Shared model helpers used by the release pipelines."""

from .vllm_client import VLLMClient

_GUARD_EXPORTS = {
    "LlamaGuard",
    "DEFAULT_MODEL_PATH",
    "parse_guard_output",
    "parse_guard_label",
    "extract_response_after_think",
    "build_messages_for_text",
}

__all__ = ["VLLMClient", *_GUARD_EXPORTS]


def __getattr__(name):
    if name in _GUARD_EXPORTS:
        from . import llama_guard as _llama_guard

        value = getattr(_llama_guard, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
