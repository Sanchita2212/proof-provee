import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional, Tuple
from app.config import settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """
    Unified provider adapter supporting:
    - groq (via OpenAI-compatible endpoint)
    - opencode (official evaluation endpoint with structured JSON mode)
    - openai / generic openai-compatible
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self.api_key = api_key or settings.LLM_API_KEY

        # Adjust default endpoint path if necessary
        if "openai" in self.base_url and not self.base_url.endswith("/chat/completions") and not self.base_url.endswith("/v1"):
            self.endpoint_url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        elif not self.base_url.endswith("/chat/completions"):
            self.endpoint_url = f"{self.base_url.rstrip('/')}/chat/completions"
        else:
            self.endpoint_url = self.base_url

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Asynchronous completion call with retry handling."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=90.0) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(self.endpoint_url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("No choices returned from LLM provider")
                    content = choices[0]["message"].get("content", "")
                    if not content and choices[0].get("finish_reason") == "length":
                        raise ValueError("LLM generation truncated by token length limit")
                    return content
                except Exception as e:
                    logger.warning(f"LLM request attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        raise RuntimeError(f"LLM generation failed after 3 attempts: {e}")
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))

    async def generate_completion_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Asynchronous completion call returning content and raw provider usage details."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=90.0) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(self.endpoint_url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("No choices returned from LLM provider")
                    content = choices[0]["message"].get("content", "")
                    usage = data.get("usage")
                    if not content and choices[0].get("finish_reason") == "length":
                        raise ValueError("LLM generation truncated by token length limit")
                    return content, usage
                except Exception as e:
                    logger.warning(f"LLM request attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        raise RuntimeError(f"LLM generation failed after 3 attempts: {e}")
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))

    def generate_completion_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Synchronous completion call for scripts and evaluators."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=90.0) as client:
            for attempt in range(3):
                try:
                    resp = client.post(self.endpoint_url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("No choices returned from LLM provider")
                    content = choices[0]["message"].get("content", "")
                    return content
                except Exception as e:
                    logger.warning(f"Sync LLM attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        raise RuntimeError(f"Sync LLM call failed after 3 attempts: {e}")
                    import time
                    time.sleep(1.5 * (attempt + 1))

llm_provider = LLMProvider()
