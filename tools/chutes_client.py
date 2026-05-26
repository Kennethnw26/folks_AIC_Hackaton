from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://llm.chutes.ai/v1/chat/completions"

MODEL_VISION = "google/gemma-4-31B-turbo-TEE"
MODEL_MATCH = "deepseek-ai/DeepSeek-V3.2-TEE"
MODEL_FRAUD = "deepseek-ai/DeepSeek-V3.2-TEE"

_RETRY_DELAYS = [3, 8]  # seconds to wait before retry 1 and retry 2


async def _post_with_retry(headers: dict, body: dict, timeout: float) -> dict | None:
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(_BASE_URL, headers=headers, json=body)
                if r.status_code == 429:
                    if attempt < len(_RETRY_DELAYS):
                        wait = _RETRY_DELAYS[attempt]
                        logger.warning("Chutes 429 rate limit — retrying in %ds (attempt %d/3)", wait, attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.error("Chutes 429 rate limit — all retries exhausted")
                        return None
                r.raise_for_status()
                return r.json()
        except httpx.TimeoutException:
            logger.warning("Chutes request timed out (attempt %d/3, timeout=%.0fs)", attempt + 1, timeout)
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(_RETRY_DELAYS[attempt])
        except Exception as e:
            logger.error("Chutes request failed: %s", e)
            return None
    return None


async def call_chutes(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    timeout: float = 30.0,
) -> BaseModel | None:
    headers = {
        "Authorization": f"Bearer {settings.chutes_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 1000,
    }
    data = await _post_with_retry(headers, body, timeout)
    if data is None:
        return None
    try:
        content = data["choices"][0]["message"]["content"]
        return response_model.model_validate_json(content)
    except Exception as e:
        logger.error("Chutes response parse failed (model=%s): %s", model, e)
        return None


async def call_chutes_vision(
    model: str,
    image_b64: str,
    system_prompt: str,
    user_text: str,
    response_model: type[BaseModel],
    timeout: float = 60.0,  # vision models are slow
) -> BaseModel | None:
    headers = {
        "Authorization": f"Bearer {settings.chutes_api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
        # Note: response_format omitted for vision — not supported by all vision models
    }
    data = await _post_with_retry(headers, body, timeout)
    if data is None:
        return None
    try:
        content = data["choices"][0]["message"]["content"]
        # Strip markdown code fences if model wraps output
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return response_model.model_validate_json(content.strip())
    except Exception as e:
        logger.error("Chutes vision parse failed (model=%s): %s", model, e)
        return None
