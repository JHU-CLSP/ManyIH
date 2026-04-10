"""
Response caching for API calls.

This module provides file-based caching to avoid redundant API calls
and reduce costs during development and testing.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


class ResponseCache:
    """File-based cache for API responses."""

    def __init__(self, cache_dir: str = ".cache/api_responses"):
        """
        Initialize response cache.

        Args:
            cache_dir: Directory to store cached responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        reasoning: bool = False,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a unique cache key based on request parameters.

        Args:
            prompt: User prompt
            model: Model identifier
            temperature: Sampling temperature
            reasoning: Whether reasoning is enabled
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens for response

        Returns:
            SHA256 hash of the combined parameters
        """
        content = f"{model}:{temperature}:{reasoning}:{max_tokens}:{system_prompt or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.

        Args:
            cache_key: The cache key

        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self,
        prompt: str,
        model: str,
        temperature: float,
        reasoning: bool = False,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Retrieve cached response if available.

        Args:
            prompt: User prompt
            model: Model identifier
            temperature: Sampling temperature
            reasoning: Whether reasoning is enabled
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens for response

        Returns:
            Cached response dict or None if not found
        """
        cache_key = self._get_cache_key(prompt, model, temperature, reasoning, system_prompt, max_tokens)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If cache file is corrupted, ignore it
                return None
        return None

    def set(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: Dict,
        reasoning: bool = False,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Store response in cache.

        Args:
            prompt: User prompt
            model: Model identifier
            temperature: Sampling temperature
            response: API response to cache
            reasoning: Whether reasoning is enabled
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens for response
        """
        cache_key = self._get_cache_key(prompt, model, temperature, reasoning, system_prompt, max_tokens)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump(response, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not write to cache: {e}")

    def clear(self):
        """Clear all cached responses."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except IOError as e:
                print(f"Warning: Could not delete cache file {cache_file}: {e}")

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics:
                - total_files: Number of cached responses
                - total_size_bytes: Total size of cache in bytes
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "total_files": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
