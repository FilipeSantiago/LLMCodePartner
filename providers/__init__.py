"""LLM providers + selection.

Each provider module exposes: `NAME`, plain-chat `stream`/`complete`, and a tool
`run(run, prompt, system, specs)` strategy. `active()` picks the one named by the
`LLM_PROVIDER` env var (default "claude").
"""
import os

from providers import claude, ollama

_PROVIDERS = {"claude": claude, "ollama": ollama}


def active():
    return _PROVIDERS.get(os.getenv("LLM_PROVIDER", "claude").lower(), claude)
