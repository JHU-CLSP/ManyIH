"""
Amazon Bedrock API client using the Converse API.

This module provides:
- BedrockClient: Synchronous client for Bedrock Converse API
- AsyncBedrockClient: Async wrapper using asyncio.to_thread()
- CachingAsyncBedrockClient: Async client with response caching
"""

import json
import time
from typing import Dict, Optional

# Default read timeout for Bedrock API calls (seconds).
# Opus-class models can take 60-120s on complex prompts, so the default
# boto3 timeout (60s) is not enough.
_DEFAULT_READ_TIMEOUT = 300


class BedrockClient:
    """
    Synchronous client for Amazon Bedrock Converse API.

    Unlike the OpenRouter/local clients, this uses boto3 instead of HTTP/urllib.
    The call_api() method returns the same dict format: {content, model, usage, raw_response}.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        profile_name: Optional[str] = None
    ):
        """
        Initialize Bedrock client.

        Args:
            model_id: Bedrock model identifier (e.g. "us.anthropic.claude-sonnet-4-6")
            region_name: AWS region (default: us-east-1)
            profile_name: Optional AWS profile name
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock support.\n"
                "Install with: pip install boto3\n"
                "Or: uv sync --extra bedrock"
            )

        self.model_id = model_id
        self.region_name = region_name

        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        import botocore.config
        boto_config = botocore.config.Config(
            read_timeout=_DEFAULT_READ_TIMEOUT,
            connect_timeout=10,
        )

        session = boto3.Session(**session_kwargs)
        self.client = session.client(
            "bedrock-runtime", region_name=region_name, config=boto_config
        )

    def call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 120,
        reasoning: bool = True
    ) -> Dict:
        """
        Call Bedrock Converse API with retry logic.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds (unused — boto3 manages its own timeouts)
            reasoning: Enable adaptive thinking with effort "high" (default: True)

        Returns:
            Dict with 'content', 'model', 'usage', and 'raw_response' keys
        """
        # Build messages in Bedrock Converse format
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        # Build kwargs
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        # Enable adaptive thinking via additionalModelRequestFields
        # Temperature must be 1 when thinking is enabled.
        if reasoning:
            kwargs["inferenceConfig"]["temperature"] = 1
            kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            }

        for attempt in range(retry_attempts):
            try:
                response = self.client.converse(**kwargs)
                return self._parse_response(response)

            except Exception as e:
                error_name = type(e).__name__

                # Retry on throttling or server errors
                if error_name in ("ThrottlingException", "ServiceUnavailableException") or "Throttling" in str(e):
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Bedrock {error_name}. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif "ClientError" in error_name or "ServiceException" in error_name:
                    if attempt < retry_attempts - 1:
                        print(f"Bedrock error: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise
                else:
                    print(f"Bedrock unexpected error: {e}")
                    if attempt < retry_attempts - 1:
                        print(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise

        raise Exception("Max retry attempts reached")

    def call_chat(
        self,
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 120
    ) -> Dict:
        """
        Call Bedrock Converse API with a pre-built messages array.

        Converts OpenAI-format messages [{role, content}, ...] to Bedrock
        Converse format. System messages are extracted into the `system` kwarg.

        Args:
            messages: List of message dicts in OpenAI format [{role, content}, ...]
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds (unused — boto3 manages its own timeouts)

        Returns:
            Dict with 'content', 'model', 'usage', and 'raw_response' keys
        """
        # Separate system messages from conversation messages
        system_parts = []
        converse_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })

        kwargs = {
            "modelId": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }

        if system_parts:
            kwargs["system"] = system_parts

        for attempt in range(retry_attempts):
            try:
                response = self.client.converse(**kwargs)
                return self._parse_response(response)

            except Exception as e:
                error_name = type(e).__name__

                if error_name in ("ThrottlingException", "ServiceUnavailableException") or "Throttling" in str(e):
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Bedrock {error_name}. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif "ClientError" in error_name or "ServiceException" in error_name:
                    if attempt < retry_attempts - 1:
                        print(f"Bedrock error: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise
                else:
                    print(f"Bedrock unexpected error: {e}")
                    if attempt < retry_attempts - 1:
                        print(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise

        raise Exception("Max retry attempts reached")

    def _parse_response(self, response: Dict) -> Dict:
        """
        Normalize Bedrock Converse response to the standard dict format.

        Bedrock response structure:
            response["output"]["message"]["content"] is a list of blocks.
            Without thinking: [{"text": "..."}]
            With thinking: [{"text": "thinking summary..."}, {"text": "response..."}]
            response["usage"]["inputTokens"] / ["outputTokens"]

        Returns:
            Dict with 'content', 'model', 'usage', and 'raw_response' keys
        """
        # Extract text content, skipping any non-text blocks
        content_blocks = response["output"]["message"]["content"]
        text_parts = [block["text"] for block in content_blocks if "text" in block]
        content = text_parts[-1] if text_parts else ""

        # Normalize usage
        bedrock_usage = response.get("usage", {})
        usage = {
            "prompt_tokens": bedrock_usage.get("inputTokens", 0),
            "completion_tokens": bedrock_usage.get("outputTokens", 0),
            "total_tokens": bedrock_usage.get("totalTokens",
                                              bedrock_usage.get("inputTokens", 0) + bedrock_usage.get("outputTokens", 0)),
        }

        return {
            "content": content.strip(),
            "model": self.model_id,
            "usage": usage,
            "raw_response": response,
        }

    def _extract_content(self, api_response: Dict) -> str:
        """Extract content from a Bedrock raw response.

        This mirrors BaseAPIClient._extract_content() so that callers
        (e.g. OpenRouterTester.extract_prediction) can use the same interface.
        """
        try:
            blocks = api_response["output"]["message"]["content"]
            text_parts = [b["text"] for b in blocks if "text" in b]
            return text_parts[-1].strip() if text_parts else ""
        except (KeyError, IndexError, TypeError):
            # If the response was already parsed by _parse_response, the raw_response
            # may have been passed through. Try direct content field.
            if "content" in api_response and isinstance(api_response["content"], str):
                return api_response["content"].strip()
            return ""


class AsyncBedrockClient:
    """
    Async wrapper around BedrockClient using asyncio.to_thread().

    The call_api() signature accepts a `session` param (ignored) for API
    compatibility with AsyncOpenRouterClient.
    """

    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        profile_name: Optional[str] = None
    ):
        self.model_id = model_id
        self.region_name = region_name
        self._sync_client = BedrockClient(
            model_id=model_id,
            region_name=region_name,
            profile_name=profile_name,
        )
        # Expose model attribute for compatibility with AsyncOpenRouterClient
        self.model = model_id

    async def call_api(
        self,
        session=None,  # ignored — for API compat with AsyncOpenRouterClient
        prompt: str = "",
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        reasoning: bool = True
    ) -> Dict:
        """
        Call Bedrock API asynchronously via thread pool.

        Args:
            session: Ignored (present for API compatibility)
            prompt: User prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
            reasoning: Enable adaptive thinking with effort "high" (default: True)

        Returns:
            API response dict with 'content', 'model', 'usage', 'raw_response'
        """
        import asyncio

        return await asyncio.to_thread(
            self._sync_client.call_api,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            reasoning=reasoning,
        )


class CachingAsyncBedrockClient(AsyncBedrockClient):
    """Async Bedrock client with built-in response caching."""

    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        profile_name: Optional[str] = None,
        cache=None,  # ResponseCache instance
    ):
        super().__init__(model_id=model_id, region_name=region_name, profile_name=profile_name)
        self.cache = cache

    async def call_api(
        self,
        session=None,
        prompt: str = "",
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        reasoning: bool = True,
        use_cache: bool = True
    ) -> Dict:
        """
        Call Bedrock API with optional caching.

        Uses the same cache key format as CachingAsyncOpenRouterClient.
        """
        # Bedrock forces temperature=1 when thinking is enabled;
        # use effective temperature in cache key so old no-thinking
        # entries (temp=0.0, reasoning=True) don't produce stale hits.
        effective_temp = 1 if reasoning else temperature

        # Check cache first
        if use_cache and self.cache:
            cache_prompt = f"{system_prompt or ''}|||{prompt}"
            cached = self.cache.get(
                cache_prompt,
                self.model,
                effective_temp,
                reasoning,
                system_prompt,
                max_tokens=max_tokens,
            )
            if cached:
                return cached

        # Call parent implementation
        response = await super().call_api(
            session=session,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            reasoning=reasoning,
        )

        # Cache the response
        if use_cache and self.cache:
            cache_prompt = f"{system_prompt or ''}|||{prompt}"
            self.cache.set(
                cache_prompt,
                self.model,
                effective_temp,
                response,
                reasoning,
                system_prompt,
                max_tokens=max_tokens,
            )

        return response
