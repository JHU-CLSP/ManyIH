"""
Configuration management for ManyIH benchmark.

API keys are resolved from CLI arguments or environment variables.
"""

import os
from typing import Dict, Optional


def get_api_key(api_key_arg: Optional[str] = None) -> str:
    """
    Get API key from CLI argument or environment variable.

    Args:
        api_key_arg: API key from command-line argument (highest priority)

    Returns:
        API key string

    Priority order:
        1. Command-line argument
        2. Environment variable (OPENROUTER_API_KEY)

    Raises:
        ValueError: If no API key is found
    """
    if api_key_arg:
        return api_key_arg

    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if not api_key:
        raise ValueError(
            "No API key found!\n\n"
            "Options:\n"
            "1. Set environment variable: export OPENROUTER_API_KEY='your-key'\n"
            "2. Use --api-key argument\n\n"
            "Get your key at: https://openrouter.ai/keys"
        )

    return api_key


def parse_model_string(model: str) -> Dict:
    """Parse model string, detecting special backend formats.

    Supported formats:
        vllm:<host>:<port>:<model_name>       -> local vLLM server
        bedrock:<alias>                        -> Bedrock shorthand (see BEDROCK_ALIASES)
        bedrock:<region>:<model_id>            -> Amazon Bedrock (full format)
        <model_name>                           -> OpenRouter (default)

    Returns dict with keys: model, base_url, is_local, backend
        backend is one of: "openrouter", "vllm", "bedrock"
        For bedrock, also includes "region"
    """
    if model.startswith("vllm:"):
        parts = model.split(":", 3)
        if len(parts) != 4:
            raise ValueError(f"Invalid vllm model format: {model}. Expected vllm:<host>:<port>:<model_name>")
        _, host, port, model_name = parts
        return {
            "model": model_name,
            "base_url": f"http://{host}:{port}/v1/chat/completions",
            "is_local": True,
            "backend": "vllm",
        }
    if model.startswith("bedrock:"):
        parts = model.split(":", 2)
        if len(parts) == 2:
            alias = parts[1]
            if alias not in BEDROCK_ALIASES:
                raise ValueError(
                    f"Unknown bedrock alias: {alias}. "
                    f"Known aliases: {', '.join(sorted(BEDROCK_ALIASES))}. "
                    f"Or use full format: bedrock:<region>:<model_id>"
                )
            region, model_id = BEDROCK_ALIASES[alias]
        elif len(parts) == 3:
            _, region, model_id = parts
        else:
            raise ValueError(f"Invalid bedrock model format: {model}. Expected bedrock:<alias> or bedrock:<region>:<model_id>")
        return {
            "model": model_id,
            "base_url": None,
            "is_local": False,
            "backend": "bedrock",
            "region": region,
        }
    # OpenRouter: support provider routing via "provider@model" syntax
    provider = None
    if "@" in model:
        provider_name, model = model.split("@", 1)
        provider = {"order": [provider_name], "allow_fallbacks": False}
    return {"model": model, "base_url": None, "is_local": False, "backend": "openrouter", "provider": provider}


# Bedrock model aliases: alias -> (region, model_id)
BEDROCK_ALIASES = {
    "opus-4.6": ("us-east-1", "us.anthropic.claude-opus-4-6-v1"),
    "sonnet-4.6": ("us-east-1", "us.anthropic.claude-sonnet-4-6"),
}


def get_reasoning_config(reasoning_effort: Optional[str] = None) -> Optional[Dict]:
    """
    Get reasoning configuration from CLI argument.

    Args:
        reasoning_effort: Reasoning effort level (high/medium/low)

    Returns:
        Reasoning config dict or None
    """
    if reasoning_effort:
        return {
            'enabled': True,
            'effort': reasoning_effort
        }
    return None
