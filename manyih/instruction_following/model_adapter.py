"""
Model adapter that wraps common backends with the generate()/generate_chat()
interface expected by the instruction-following eval scripts.

Usage:
    from manyih.instruction_following.model_adapter import create_model

    model = create_model("bedrock:us-east-1:us.anthropic.claude-opus-4-6-v1")
    model = create_model("anthropic/claude-3.5-sonnet", api_key="sk-...")
    model = create_model("vllm:localhost:8000:meta-llama/Llama-3.1-8B-Instruct")
"""

import hashlib
import json
import os

from manyih.common.config import parse_model_string
from manyih.common.cache import ResponseCache


class _CacheShim:
    """Compatibility shim so eval scripts that access model.cache don't break."""
    add_n = 0

    def save_cache(self):
        pass


class ModelAdapter:
    """
    Base adapter wrapping src/common/ clients with generate()/generate_chat().

    Provides:
    - generate(query, ...) -> str | None  (single prompt, for LLM-judge calls)
    - generate_chat(messages, ...) -> str | None  (multi-turn, for predictions)
    - cache attribute with no-op save_cache() for compatibility
    """

    def __init__(self, backend, model_name, response_cache=None):
        self.backend = backend
        self.model_name = model_name
        self._response_cache = response_cache
        self.cache = _CacheShim()

    def _strip_think_tags(self, text):
        """Strip </think> tags from reasoning model responses."""
        if text and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text

    def _cache_key_for_messages(self, messages, model, temperature, max_tokens):
        """Generate a cache key for a messages array."""
        content = f"{model}:{temperature}:{max_tokens}:{json.dumps(messages, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def generate(self, query, max_tokens=8192, temperature=0.0):
        """
        Generate a response from a single prompt (used by LLM judge).

        Returns:
            str or None on failure
        """
        # Check cache
        if temperature == 0.0 and self._response_cache:
            cached = self._response_cache.get(
                query, self.model_name, temperature, max_tokens=max_tokens
            )
            if cached:
                return self._strip_think_tags(cached.get("content"))

        try:
            result = self.backend.call_api(
                prompt=query,
                max_tokens=max_tokens,
                temperature=temperature,
                retry_attempts=3,
                retry_delay=2.0,
            )
            content = result.get("content", "")

            # Save to cache
            if temperature == 0.0 and self._response_cache:
                self._response_cache.set(
                    query, self.model_name, temperature, result, max_tokens=max_tokens
                )

            return self._strip_think_tags(content)

        except Exception as e:
            print(f"generate() error: {e}")
            return None

    def generate_chat(self, messages, max_tokens=8192, temperature=0.0):
        """
        Generate a response from a multi-turn messages array.

        Args:
            messages: List of dicts [{role, content}, ...]

        Returns:
            str or None on failure
        """
        # Cache key includes all messages to distinguish prompts that share
        # the same user message but differ in system prompt
        cache_prompt = json.dumps(messages, sort_keys=True)
        if temperature == 0.0 and self._response_cache:
            cached = self._response_cache.get(
                cache_prompt, self.model_name, temperature, max_tokens=max_tokens
            )
            if cached:
                return self._strip_think_tags(cached.get("content"))

        try:
            result = self._call_chat(messages, max_tokens, temperature)
            content = result.get("content", "")

            # Save to cache
            if temperature == 0.0 and self._response_cache:
                self._response_cache.set(
                    cache_prompt, self.model_name, temperature, result,
                    max_tokens=max_tokens
                )

            return self._strip_think_tags(content)

        except Exception as e:
            print(f"generate_chat() error: {e}")
            return None

    def _call_chat(self, messages, max_tokens, temperature):
        """Backend-specific multi-turn call. Override in subclasses."""
        raise NotImplementedError


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenRouter and vLLM/local model backends."""

    def _call_chat(self, messages, max_tokens, temperature):
        return self.backend.call_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_attempts=3,
            retry_delay=2.0,
        )


class BedrockAdapter(ModelAdapter):
    """Adapter for Amazon Bedrock backend."""

    def _call_chat(self, messages, max_tokens, temperature):
        return self.backend.call_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_attempts=3,
            retry_delay=2.0,
        )


def create_model(model_string, cache_dir=None, api_key=None, max_tokens=8192):
    """
    Factory function to create a ModelAdapter from a model string.

    Args:
        model_string: Model identifier. Supported formats:
            - "anthropic/claude-3.5-sonnet"  (OpenRouter)
            - "bedrock:us-east-1:us.anthropic.claude-opus-4-6-v1"  (Bedrock)
            - "vllm:localhost:8000:model-name"  (local vLLM)
        cache_dir: Optional directory for ResponseCache. If None, no caching.
        api_key: Optional API key (for OpenRouter).
        max_tokens: Default max tokens (not used here, kept for interface compat).

    Returns:
        ModelAdapter instance with generate()/generate_chat() methods
    """
    parsed = parse_model_string(model_string)
    backend_type = parsed["backend"]
    model_name = parsed["model"]

    # Set up response cache
    response_cache = None
    if cache_dir:
        response_cache = ResponseCache(cache_dir=cache_dir)

    if backend_type == "bedrock":
        from manyih.common.bedrock_client import BedrockClient
        backend = BedrockClient(
            model_id=model_name,
            region_name=parsed["region"],
        )
        return BedrockAdapter(backend, model_name, response_cache)

    elif backend_type == "vllm":
        from manyih.common.api_client import LocalModelClient
        backend = LocalModelClient(
            model=model_name,
            base_url=parsed["base_url"],
        )
        return OpenAICompatibleAdapter(backend, model_name, response_cache)

    else:
        # OpenRouter (default)
        from manyih.common.api_client import OpenRouterClient
        from manyih.common.config import get_api_key
        resolved_key = api_key or get_api_key()
        backend = OpenRouterClient(
            api_key=resolved_key,
            model=model_name,
            provider=parsed.get("provider"),
        )
        return OpenAICompatibleAdapter(backend, model_name, response_cache)
