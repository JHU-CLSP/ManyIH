"""
Unified API client with both sync and async support.

This module provides:
- BaseAPIClient: Base synchronous client for OpenAI-compatible APIs
- OpenRouterClient: Synchronous client for OpenRouter API
- LocalModelClient: Synchronous client for local models (vLLM, sglang, etc.)
- AsyncOpenRouterClient: Asynchronous API client with aiohttp
- CachingAsyncOpenRouterClient: Async client with response caching
"""

import json
import time
from typing import Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# Default endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"


class BaseAPIClient:
    """
    Base synchronous client for OpenAI-compatible chat completion APIs.

    This class provides the core functionality that works with any OpenAI-compatible
    endpoint (OpenRouter, vLLM, sglang, ollama, etc.).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict] = None
    ):
        """
        Initialize base API client.

        Args:
            base_url: Full URL for the chat completions endpoint
            model: Model identifier
            api_key: Optional API key (some local servers don't require it)
            extra_headers: Optional additional headers to include in requests
            extra_body: Optional additional fields to include in request body
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers. Override in subclasses for custom headers."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _build_request_body(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Dict:
        """Build request body. Override in subclasses for custom fields."""
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        data.update(self.extra_body)
        return data

    def call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 120
    ) -> Dict:
        """
        Call the API with retry logic.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds

        Returns:
            API response dictionary with 'content' key containing the response text
        """
        headers = self._build_headers()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = self._build_request_body(messages, max_tokens, temperature)

        for attempt in range(retry_attempts):
            try:
                request = Request(
                    self.base_url,
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )

                with urlopen(request, timeout=timeout) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return {
                        'content': self._extract_content(result),
                        'model': result.get('model'),
                        'usage': result.get('usage'),
                        'raw_response': result
                    }

            except HTTPError as e:
                error_body = e.read().decode('utf-8')
                print(f"HTTP Error {e.code}: {error_body}")

                if e.code == 429:  # Rate limit
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif e.code >= 500:  # Server error
                    if attempt < retry_attempts - 1:
                        print(f"Server error. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise
                else:
                    raise

            except URLError as e:
                print(f"URL Error: {e.reason}")
                if attempt < retry_attempts - 1:
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise

            except Exception as e:
                print(f"Unexpected error: {e}")
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
        Call the API with a pre-built messages array.

        Unlike call_api() which constructs messages from prompt/system_prompt,
        this method passes the messages list directly, supporting multi-turn
        conversations.

        Args:
            messages: List of message dicts [{role, content}, ...]
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds

        Returns:
            API response dictionary with 'content' key containing the response text
        """
        headers = self._build_headers()
        data = self._build_request_body(messages, max_tokens, temperature)

        for attempt in range(retry_attempts):
            try:
                request = Request(
                    self.base_url,
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )

                with urlopen(request, timeout=timeout) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return {
                        'content': self._extract_content(result),
                        'model': result.get('model'),
                        'usage': result.get('usage'),
                        'raw_response': result
                    }

            except HTTPError as e:
                error_body = e.read().decode('utf-8')
                print(f"HTTP Error {e.code}: {error_body}")

                if e.code == 429:  # Rate limit
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif e.code >= 500:  # Server error
                    if attempt < retry_attempts - 1:
                        print(f"Server error. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        raise
                else:
                    raise

            except URLError as e:
                print(f"URL Error: {e.reason}")
                if attempt < retry_attempts - 1:
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise

            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt < retry_attempts - 1:
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise

        raise Exception("Max retry attempts reached")

    def _extract_content(self, api_response: Dict) -> str:
        """Extract content from API response."""
        try:
            return api_response['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError) as e:
            print(f"\n⚠️  Error extracting content: {e}")
            print(f"Full API response structure:")
            print(json.dumps(api_response, indent=2))

            # Try alternative response structures
            if 'choices' in api_response and len(api_response['choices']) > 0:
                choice = api_response['choices'][0]
                print(f"Choice structure: {json.dumps(choice, indent=2)}")

                # Try: choices[0].text (for some completion models)
                if 'text' in choice:
                    print("Found content in choices[0].text")
                    return choice['text'].strip()

                # Try: choices[0].delta.content (for streaming)
                if 'delta' in choice and 'content' in choice['delta']:
                    print("Found content in choices[0].delta.content")
                    return choice['delta']['content'].strip()

            # Try: output (some models use this)
            if 'output' in api_response:
                print(f"Found output field: {api_response['output']}")
                if isinstance(api_response['output'], str):
                    return api_response['output'].strip()
                elif isinstance(api_response['output'], dict) and 'text' in api_response['output']:
                    return api_response['output']['text'].strip()

            # Try: content (direct field)
            if 'content' in api_response:
                print(f"Found direct content field")
                return api_response['content'].strip()

            print("❌ Could not find content in any known location")
            return ""


class LocalModelClient(BaseAPIClient):
    """
    Synchronous client for locally-served models (vLLM, sglang, ollama, etc.).

    These servers typically provide OpenAI-compatible endpoints without requiring
    authentication or special headers.
    """

    def __init__(
        self,
        model: str,
        base_url: str = LOCAL_DEFAULT_BASE_URL,
        api_key: Optional[str] = None
    ):
        """
        Initialize local model client.

        Args:
            model: Model identifier (as loaded in the server)
            base_url: Server URL (default: http://localhost:8000/v1/chat/completions)
            api_key: Optional API key (most local servers don't require one)
        """
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key
        )


class OpenRouterClient(BaseAPIClient):
    """
    Synchronous OpenRouter API client.

    Extends BaseAPIClient with OpenRouter-specific features like
    site ranking headers and reasoning configuration.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        reasoning_config: Optional[Dict] = None,
        provider: Optional[Dict] = None
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (see https://openrouter.ai/models)
            site_url: Optional site URL for ranking
            app_name: Optional app name for ranking
            reasoning_config: Optional reasoning configuration dict with 'enabled' and 'effort' keys
            provider: Optional provider configuration dict (e.g. {"order": ["anthropic"], "allow_fallbacks": false})
        """
        self.site_url = site_url or "https://github.com"
        self.app_name = app_name or "instruction-hierarchy-benchmark"
        self.reasoning_config = reasoning_config
        self.provider = provider

        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            model=model,
            api_key=api_key,
            extra_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name
            }
        )

    def _build_request_body(
        self,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> Dict:
        """Build request body with OpenRouter-specific reasoning parameter."""
        data = super()._build_request_body(messages, max_tokens, temperature)

        # Add reasoning parameter if enabled (OpenRouter-specific)
        if self.reasoning_config and self.reasoning_config.get('enabled'):
            data["reasoning"] = {
                "effort": self.reasoning_config.get('effort', 'high')
            }

        # Add provider configuration if set
        if self.provider:
            data["provider"] = self.provider

        return data


class AsyncOpenRouterClient:
    """Asynchronous OpenRouter API client using aiohttp."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: Optional[str] = None,
        site_url: str = "https://github.com",
        app_name: str = "instruction-hierarchy-benchmark",
        provider: Optional[Dict] = None
    ):
        """
        Initialize async OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model identifier
            base_url: Optional API endpoint URL (defaults to OpenRouter)
            site_url: Site URL for ranking
            app_name: App name for ranking
            provider: Optional provider configuration dict (e.g. {"order": ["anthropic"], "allow_fallbacks": false})
        """
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.app_name = app_name
        self.provider = provider
        self.base_url = base_url or "https://openrouter.ai/api/v1/chat/completions"

    async def call_api(
        self,
        session,  # aiohttp.ClientSession
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        reasoning: bool = True
    ) -> Dict:
        """
        Call OpenRouter API asynchronously.

        Args:
            session: aiohttp ClientSession for making requests
            prompt: User prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
            reasoning: Whether to enable extended thinking/reasoning

        Returns:
            API response dictionary with 'content' key containing the response text
        """
        import asyncio  # Import here to avoid requiring aiohttp for sync client

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning:
            data["reasoning"] = {"enabled": True, "effort": "high"}
        if self.provider:
            data["provider"] = self.provider

        # Make request with retry logic
        for attempt in range(retry_attempts):
            try:
                # Import aiohttp here to avoid requiring it for sync client
                import aiohttp

                async with session.post(
                    self.base_url,
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=1800)
                ) as response:
                    if response.status == 429:  # Rate limit
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status >= 500:
                        if attempt < retry_attempts - 1:
                            print(f"Server error {response.status}. Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            error_body = await response.text()
                            raise Exception(f"Server error {response.status}: {error_body}")

                    if response.status != 200:
                        error_body = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_body}")

                    result = await response.json()

                    # Extract content from response
                    content = self._extract_content(result)
                    return {
                        "content": content,
                        "model": result.get("model"),
                        "usage": result.get("usage"),
                        "raw_response": result
                    }

            except Exception as e:
                print(f"Request error: {e}")
                if attempt < retry_attempts - 1:
                    print(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise

        raise Exception("Max retry attempts reached")

    def _extract_content(self, api_response: Dict) -> str:
        """Extract content from API response."""
        try:
            return api_response['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError):
            # Try alternative locations
            if 'choices' in api_response and api_response['choices']:
                choice = api_response['choices'][0]
                if 'text' in choice:
                    return choice['text'].strip()
                if 'delta' in choice and 'content' in choice['delta']:
                    return choice['delta']['content'].strip()
            if 'output' in api_response:
                if isinstance(api_response['output'], str):
                    return api_response['output'].strip()
            if 'content' in api_response:
                return api_response['content'].strip()
            return ""


class CachingAsyncOpenRouterClient(AsyncOpenRouterClient):
    """Async OpenRouter API client with built-in response caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        cache=None,  # ResponseCache instance
        base_url: Optional[str] = None,
        site_url: str = "https://github.com",
        app_name: str = "instruction-hierarchy-benchmark",
        provider: Optional[Dict] = None
    ):
        """
        Initialize async OpenRouter client with caching.

        Args:
            api_key: OpenRouter API key
            model: Model identifier
            cache: ResponseCache instance for caching responses
            base_url: Optional API endpoint URL (defaults to OpenRouter)
            site_url: Site URL for ranking
            app_name: App name for ranking
            provider: Optional provider configuration dict (e.g. {"order": ["anthropic"], "allow_fallbacks": false})
        """
        super().__init__(api_key, model, base_url=base_url, site_url=site_url, app_name=app_name, provider=provider)
        self.cache = cache

    async def call_api(
        self,
        session,  # aiohttp.ClientSession
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        reasoning: bool = True,
        use_cache: bool = True
    ) -> Dict:
        """
        Call OpenRouter API asynchronously with optional caching.

        Args:
            session: aiohttp ClientSession for making requests
            prompt: User prompt to send
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
            reasoning: Whether to enable extended thinking/reasoning
            use_cache: Whether to use caching (if cache is available)

        Returns:
            API response dictionary with 'content' key containing the response text
        """
        # Check cache first if enabled
        if use_cache and self.cache:
            cache_prompt = f"{system_prompt or ''}|||{prompt}"
            cached = self.cache.get(
                cache_prompt,
                self.model,
                temperature,
                reasoning,
                system_prompt,
                max_tokens=max_tokens
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
            reasoning=reasoning
        )

        # Cache the response if caching is enabled
        if use_cache and self.cache:
            cache_prompt = f"{system_prompt or ''}|||{prompt}"
            self.cache.set(
                cache_prompt,
                self.model,
                temperature,
                response,
                reasoning,
                system_prompt,
                max_tokens=max_tokens
            )

        return response
