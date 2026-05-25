"""Hermes 4 client via Chutes, using the openai Python SDK."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_hermes_client() -> OpenAI:
    """Return an OpenAI client pointed at the Chutes Hermes 4 endpoint."""
    api_key = os.getenv("CHUTES_API_KEY")
    base_url = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
    if not api_key:
        raise RuntimeError(
            "CHUTES_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def get_hermes_model() -> str:
    return os.getenv("CHUTES_MODEL", "NousResearch/Hermes-4-405B")
