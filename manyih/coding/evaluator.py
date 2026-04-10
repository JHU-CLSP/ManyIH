#!/usr/bin/env python3
"""
Evaluation script for the ManyIH coding subset.

This script:
1. Formats raw coding datapoints into LLM messages
2. Calls the model API with caching to avoid redundant API calls
3. Evaluates responses using judge_response
"""

import argparse
import asyncio
import importlib.resources
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)

from .evaluate import judge_response
from .prompts import SYSTEM_PROMPT, get_system_prompt

from manyih.common import ResponseCache, CachingAsyncOpenRouterClient, CachingAsyncBedrockClient, get_api_key, parse_model_string


def get_default_data_path() -> str:
    """Get the path to the bundled coding benchmark data."""
    return str(importlib.resources.files("manyih.data").joinpath("coding.json"))


def format_datapoint_for_llm(
    datapoint: Dict,
    include_system_prompt: bool = True,
    hierarchy_format: str = "scalar",
    annotation_style: str = "inline",
) -> Dict[str, str]:
    """
    Format a raw conflicting style MBPP datapoint into a message for LLM evaluation.

    Args:
        datapoint: Raw datapoint from coding.json
        include_system_prompt: Whether to include a system prompt
        hierarchy_format: "scalar" or "ordinal" - selects matching system prompt
        annotation_style: "boundary" or "inline" - selects matching system prompt

    Returns:
        Dict with 'system_prompt' (optional) and 'user_prompt' keys
    """
    # The datapoint['prompt'] already contains the formatted task with style instructions
    user_prompt = datapoint['prompt']

    result = {"user_prompt": user_prompt}

    if include_system_prompt:
        result["system_prompt"] = get_system_prompt(hierarchy_format, annotation_style)

    return result


async def run_evaluation(
    data_file: str,
    model: str,
    api_key: str,
    output_file: str,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: str = ".cache/coding",
    delay_between_calls: float = 0.5,
    save_frequency: int = 5,
    resume: bool = False,
    max_tokens: int = 40960,
    temperature: float = 0.0,
    include_system_prompt: bool = True,
    timeout: float = 5.0,
    concurrency: int = 10,
    reasoning: bool = True,
    provider: Optional[Dict] = None,
    base_url: Optional[str] = None,
    backend: str = "openrouter",
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline on ManyIH coding subset asynchronously.

    Args:
        data_file: Path to coding.json
        model: OpenRouter model identifier
        api_key: OpenRouter API key
        output_file: Path to save results
        max_samples: Maximum samples to evaluate (None = all)
        use_cache: Whether to cache API responses
        cache_dir: Directory for cache files
        delay_between_calls: Delay between API calls (per-request)
        save_frequency: Save results every N samples
        resume: Resume from existing output file
        max_tokens: Max tokens for API response
        temperature: Sampling temperature
        include_system_prompt: Include system prompt in API calls
        timeout: Timeout for code execution during evaluation
        concurrency: Maximum number of concurrent API calls
        reasoning: Whether to enable extended thinking/reasoning (default: True)
        provider: Optional provider configuration dict for OpenRouter
        base_url: Optional base URL for the API endpoint (e.g. for vLLM servers)
        backend: Backend type ("openrouter", "vllm", or "bedrock")
        region: AWS region for Bedrock backend

    Returns:
        Dict with evaluation summary and detailed results
    """
    # Load dataset
    with open(data_file, 'r') as f:
        raw_data = json.load(f)

    # Handle both old format (list) and new format (dict with config/data)
    if isinstance(raw_data, dict) and "data" in raw_data:
        dataset = raw_data["data"]
        dataset_config = raw_data.get("config", {})
    else:
        dataset = raw_data
        dataset_config = {}

    if max_samples:
        dataset = dataset[:max_samples]

    # Read hierarchy format and annotation style from dataset config
    # (defaults for backward compat with older datasets that lack these fields)
    hierarchy_format = dataset_config.get("hierarchy_format", "scalar")
    annotation_style = dataset_config.get("annotation_style", "inline")

    # Bedrock forces temperature=1 when thinking is enabled; use effective
    # temperature for cache keys so stale no-thinking entries aren't returned.
    effective_temp = 1 if (backend == "bedrock" and reasoning) else temperature

    # Build dataset metadata from config and/or computed values
    dataset_metadata = build_dataset_metadata(dataset, dataset_config)
    print_dataset_metadata(dataset_metadata)

    # Capture all input parameters for reproducibility
    eval_config = {
        "data_file": os.path.abspath(data_file),
        "model": model,
        "output_file": os.path.abspath(output_file),
        "max_samples": max_samples,
        "use_cache": use_cache,
        "cache_dir": cache_dir,
        "delay_between_calls": delay_between_calls,
        "save_frequency": save_frequency,
        "resume": resume,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "include_system_prompt": include_system_prompt,
        "timeout": timeout,
        "concurrency": concurrency,
        "reasoning": reasoning
    }
    # breakpoint()

    print(f"\nEvaluating {len(dataset)} samples")
    print(f"Model: {model}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Reasoning: {'enabled' if reasoning else 'disabled'}")
    print(f"Concurrency: {concurrency}")
    print(f"Output: {output_file}")
    print("-" * 70)

    # Initialize client and cache
    cache = ResponseCache(cache_dir=cache_dir) if use_cache else None
    if backend == "bedrock":
        client = CachingAsyncBedrockClient(model_id=model, region_name=region or "us-east-1", cache=cache)
    else:
        client = CachingAsyncOpenRouterClient(api_key=api_key, model=model, cache=cache, provider=provider, base_url=base_url)

    # Resume from existing results
    results = []
    start_idx = 0

    if resume and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
            results = saved_data.get('results', [])
        start_idx = len(results)
        print(f"Resuming from sample {start_idx}")

    # Track statistics
    stats = {
        "total": 0,
        "test_passed": 0,
        "style_passed": 0,
        "overall_passed": 0,
        "cache_hits": 0,
        "api_calls": 0
    }

    # Track detailed failure statistics
    failure_stats = {
        "test_failures": 0,
        "style_failures": 0,
        "style_failures_by_category": {},  # style_key -> count of failures
        "test_failure_reasons": {},  # reason -> count
        "samples_with_style_failures": [],  # list of sample ids with style failures
    }

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def process_sample(
        session: aiohttp.ClientSession,
        i: int,
        datapoint: Dict
    ) -> Dict:
        """Process a single sample asynchronously."""
        async with semaphore:
            try:
                # Format datapoint for LLM
                formatted = format_datapoint_for_llm(datapoint, include_system_prompt, hierarchy_format, annotation_style)

                # Check if we have a cached response
                cache_prompt = f"{formatted.get('system_prompt', '')}|||{formatted['user_prompt']}"
                cached = cache.get(cache_prompt, model, effective_temp, reasoning, max_tokens=max_tokens) if cache else None

                is_cache_hit = False
                if cached:
                    is_cache_hit = True
                    response_content = cached['content']
                else:
                    # Call API
                    response = await client.call_api(
                        session=session,
                        prompt=formatted['user_prompt'],
                        system_prompt=formatted.get('system_prompt'),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        use_cache=use_cache,
                        reasoning=reasoning
                    )
                    response_content = response['content']

                    # Add delay between API calls
                    if delay_between_calls > 0:
                        await asyncio.sleep(delay_between_calls)

                # Evaluate response using judge_response
                expected_styles = datapoint.get('metadata', {}).get('expected_styles', {})
                test_code = datapoint['test_code']

                eval_result = judge_response(
                    response=response_content,
                    test_code=test_code,
                    expected_styles=expected_styles,
                    timeout=timeout
                )

                # Build result
                result = {
                    "id": datapoint['id'],
                    "task_id": datapoint['task_id'],
                    "response": response_content,
                    "evaluation": eval_result,
                    "expected_styles": expected_styles,
                    "n_conflicts": datapoint.get('metadata', {}).get('n_conflicts', 0),
                    "n_instructions": datapoint.get('metadata', {}).get('n_instructions', 0),
                    "n_style_groups": datapoint.get('metadata', {}).get('n_style_groups', 0),
                    "n_expected_styles": sum(len(v) for v in expected_styles.values()),
                    "_index": i,
                    "_cache_hit": is_cache_hit,
                    "_formatted_prompts": formatted,
                    "_raw_response": response,
                }

                return result

            except Exception as e:
                expected_styles = datapoint.get('metadata', {}).get('expected_styles', {})
                return {
                    "id": datapoint['id'],
                    "task_id": datapoint['task_id'],
                    "error": str(e),
                    "evaluation": {
                        "test_passed": False,
                        "style_passed": False,
                        "overall_passed": False
                    },
                    "n_conflicts": datapoint.get('metadata', {}).get('n_conflicts', 0),
                    "n_instructions": datapoint.get('metadata', {}).get('n_instructions', 0),
                    "n_style_groups": datapoint.get('metadata', {}).get('n_style_groups', 0),
                    "n_expected_styles": sum(len(v) for v in expected_styles.values()),
                    "_index": i,
                    "_cache_hit": False
                }

    # Process samples concurrently
    samples_to_process = list(enumerate(dataset[start_idx:], start=start_idx))

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_sample(session, i, datapoint)
            for i, datapoint in samples_to_process
        ]

        # Process with progress reporting
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1

            # Update stats
            stats["total"] += 1
            if result.get("_cache_hit"):
                stats["cache_hits"] += 1
            else:
                stats["api_calls"] += 1

            eval_result = result.get("evaluation", {})
            if eval_result.get('test_passed'):
                stats["test_passed"] += 1
            else:
                # Track test failure
                failure_stats["test_failures"] += 1
                test_reason = eval_result.get('test_result', 'unknown')
                # Categorize test failure reason
                if 'timed out' in str(test_reason).lower():
                    reason_key = 'timeout'
                elif 'assertion' in str(test_reason).lower():
                    reason_key = 'assertion_error'
                elif 'syntax' in str(test_reason).lower():
                    reason_key = 'syntax_error'
                elif 'error' in str(test_reason).lower():
                    reason_key = 'runtime_error'
                else:
                    reason_key = 'other'
                failure_stats["test_failure_reasons"][reason_key] = \
                    failure_stats["test_failure_reasons"].get(reason_key, 0) + 1

            if eval_result.get('style_passed'):
                stats["style_passed"] += 1
            else:
                # Track style failures by category
                failure_stats["style_failures"] += 1
                failure_stats["samples_with_style_failures"].append(result.get("id"))
                each_style = eval_result.get('each_style_passed', {})
                for style_key, passed in each_style.items():
                    if not passed:
                        failure_stats["style_failures_by_category"][style_key] = \
                            failure_stats["style_failures_by_category"].get(style_key, 0) + 1

            if eval_result.get('overall_passed'):
                stats["overall_passed"] += 1

            # Print progress
            idx = result.get("_index", 0)
            cache_status = "[CACHE]" if result.get("_cache_hit") else "[API]"
            test_status = "PASS" if eval_result.get('test_passed') else "FAIL"
            style_status = "PASS" if eval_result.get('style_passed') else "FAIL"
            overall_status = "PASS" if eval_result.get('overall_passed') else "FAIL"

            if "error" in result:
                print(f"[{completed}/{len(tasks)}] Sample {idx+1} (id={result['id']}): ERROR - {result['error']}")
            else:
                print(f"[{completed}/{len(tasks)}] Sample {idx+1} (id={result['id']}) {cache_status}: Test={test_status} Style={style_status} Overall={overall_status}")

            # Remove internal tracking fields before storing
            result.pop("_index", None)
            result.pop("_cache_hit", None)
            results.append(result)

            # Save periodically
            if completed % save_frequency == 0:
                _save_results(output_file, results, stats, eval_config, dataset_metadata, failure_stats)
                print(f"  Progress saved ({stats['total']} samples)")

    # Final save
    _save_results(output_file, results, stats, eval_config, dataset_metadata, failure_stats)

    # Compute analysis for printing
    analysis = _compute_analysis(results, stats, failure_stats)

    # Print summary
    print(f"\nCache hits: {stats['cache_hits']}, API calls: {stats['api_calls']}")
    _print_analysis("EVALUATION COMPLETE", stats, failure_stats, analysis)

    return {
        "eval_config": eval_config,
        "stats": stats,
        "failure_stats": failure_stats,
        "results": results,
        "dataset_metadata": dataset_metadata
    }


def _compute_analysis(results: List[Dict], stats: Dict, failure_stats: Dict) -> Dict[str, Any]:
    """Compute detailed analysis from results and stats.

    Args:
        results: List of evaluation results
        stats: Basic stats dict
        failure_stats: Failure statistics dict

    Returns:
        Analysis dict with percentages and breakdowns
    """
    total = stats['total']
    if total == 0:
        return {}

    test_failures = failure_stats.get("test_failures", 0)
    style_failures = failure_stats.get("style_failures", 0)

    # Compute failure breakdown
    total_failed = total - stats['overall_passed']
    only_test_failed = sum(1 for r in results
                           if not r.get('evaluation', {}).get('test_passed', True)
                           and r.get('evaluation', {}).get('style_passed', False))
    only_style_failed = sum(1 for r in results
                            if r.get('evaluation', {}).get('test_passed', True)
                            and not r.get('evaluation', {}).get('style_passed', False))
    both_failed = sum(1 for r in results
                     if not r.get('evaluation', {}).get('test_passed', True)
                     and not r.get('evaluation', {}).get('style_passed', False))

    analysis = {
        "summary": {
            "total_samples": total,
            "test_passed": stats["test_passed"],
            "test_passed_pct": round(100 * stats["test_passed"] / total, 2),
            "style_passed": stats["style_passed"],
            "style_passed_pct": round(100 * stats["style_passed"] / total, 2),
            "overall_passed": stats["overall_passed"],
            "overall_passed_pct": round(100 * stats["overall_passed"] / total, 2),
        },
        "test_failures": {
            "count": test_failures,
            "pct_of_total": round(100 * test_failures / total, 2) if total > 0 else 0,
            "by_reason": {
                reason: {
                    "count": count,
                    "pct_of_failures": round(100 * count / test_failures, 2) if test_failures > 0 else 0,
                    "pct_of_total": round(100 * count / total, 2)
                }
                for reason, count in sorted(failure_stats.get("test_failure_reasons", {}).items(), key=lambda x: -x[1])
            }
        },
        "style_failures": {
            "count": style_failures,
            "pct_of_total": round(100 * style_failures / total, 2) if total > 0 else 0,
            "by_category": {
                style_key: {
                    "count": count,
                    "pct_of_failures": round(100 * count / style_failures, 2) if style_failures > 0 else 0,
                    "pct_of_total": round(100 * count / total, 2)
                }
                for style_key, count in sorted(failure_stats.get("style_failures_by_category", {}).items(), key=lambda x: -x[1])
            }
        },
        "failure_breakdown": {
            "total_failed": total_failed,
            "total_failed_pct": round(100 * total_failed / total, 2) if total > 0 else 0,
            "only_test_failed": only_test_failed,
            "only_test_failed_pct": round(100 * only_test_failed / total, 2) if total > 0 else 0,
            "only_style_failed": only_style_failed,
            "only_style_failed_pct": round(100 * only_style_failed / total, 2) if total > 0 else 0,
            "both_failed": both_failed,
            "both_failed_pct": round(100 * both_failed / total, 2) if total > 0 else 0,
        }
    }

    # Fine-grained metrics

    # Metric 1: Per-assertion pass rate
    assertion_pass_rates = []
    total_passed_assertions = 0
    total_assertions = 0

    for result in results:
        eval_result = result.get("evaluation", {})
        if "assertion_pass_rate" in eval_result:
            assertion_pass_rates.append(eval_result["assertion_pass_rate"])
            total_passed_assertions += eval_result.get("passed_assertions", 0)
            total_assertions += eval_result.get("total_assertions", 0)

    # Metric 2: Per-style satisfaction rate
    style_satisfaction_rates = []
    style_category_stats = {}

    for result in results:
        eval_result = result.get("evaluation", {})
        each_style = eval_result.get("each_style_passed", {})

        if each_style:
            passed = sum(1 for v in each_style.values() if v)
            total_styles = len(each_style)
            rate = passed / total_styles if total_styles > 0 else 0.0
            style_satisfaction_rates.append(rate)

            for style_key, passed_flag in each_style.items():
                if style_key not in style_category_stats:
                    style_category_stats[style_key] = {"passed": 0, "total": 0}
                style_category_stats[style_key]["total"] += 1
                if passed_flag:
                    style_category_stats[style_key]["passed"] += 1

    analysis["fine_grained"] = {
        "assertion_stats": {
            "avg_pass_rate": sum(assertion_pass_rates) / len(assertion_pass_rates) if assertion_pass_rates else None,
            "total_assertions": total_assertions,
            "passed_assertions": total_passed_assertions,
        },
        "style_satisfaction": {
            "avg_satisfaction_rate": sum(style_satisfaction_rates) / len(style_satisfaction_rates) if style_satisfaction_rates else None,
            "by_category": {
                k: {"passed": v["passed"], "total": v["total"], "rate": v["passed"]/v["total"] if v["total"] > 0 else 0}
                for k, v in sorted(style_category_stats.items())
            },
        },
    }

    return analysis


def _save_results(
    output_file: str,
    results: List[Dict],
    stats: Dict,
    eval_config: Dict[str, Any],
    dataset_metadata: Optional[Dict] = None,
    failure_stats: Optional[Dict] = None
):
    """Save results to file with analysis."""
    # Don't save the full list of sample IDs in failure_stats to keep file size reasonable
    failure_stats_to_save = None
    if failure_stats:
        failure_stats_to_save = {
            "test_failures": failure_stats["test_failures"],
            "style_failures": failure_stats["style_failures"],
            "style_failures_by_category": failure_stats["style_failures_by_category"],
            "test_failure_reasons": failure_stats["test_failure_reasons"],
            "n_samples_with_style_failures": len(failure_stats.get("samples_with_style_failures", []))
        }

    # Compute analysis
    analysis = _compute_analysis(results, stats, failure_stats) if failure_stats else {}

    output_data = {
        "eval_config": eval_config,
        "dataset_metadata": dataset_metadata,
        "stats": stats,
        "failure_stats": failure_stats_to_save,
        "analysis": analysis,
        "results": results
    }
    # Create parent directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def build_dataset_metadata(dataset: List[Dict], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build dataset metadata from config and/or computed from samples.

    Args:
        dataset: List of data samples
        config: Optional config dict from dataset file (new format)

    Returns:
        Combined metadata dict
    """
    config = config or {}
    n_samples = len(dataset)

    if n_samples == 0:
        return {"n_samples": 0, "config": config}

    # Collect per-sample metadata for computed stats
    n_instructions_list = []
    n_conflicts_list = []
    n_style_groups_list = []
    n_expected_styles_list = []
    all_style_keys = set()

    for dp in dataset:
        meta = dp.get("metadata", {})
        if "n_instructions" in meta:
            n_instructions_list.append(meta["n_instructions"])
        if "n_conflicts" in meta:
            n_conflicts_list.append(meta["n_conflicts"])
        if "n_style_groups" in meta:
            n_style_groups_list.append(meta["n_style_groups"])
        if "expected_styles" in meta:
            all_style_keys.update(meta["expected_styles"].keys())
            n_expected_styles_list.append(sum(len(v) for v in meta["expected_styles"].values()))

    # Build metadata - use config values if available, otherwise compute
    metadata = {
        "n_samples": n_samples,
        "config": config,  # Include original config for full reproducibility
        "n_instructions": {
            "min": config.get("n_instructions_min") or (min(n_instructions_list) if n_instructions_list else 0),
            "max": config.get("n_instructions_max") or (max(n_instructions_list) if n_instructions_list else 0),
            "avg": sum(n_instructions_list) / len(n_instructions_list) if n_instructions_list else 0
        },
        "n_conflicts": {
            "min": config.get("n_conflicts_min") or (min(n_conflicts_list) if n_conflicts_list else 0),
            "max": config.get("n_conflicts_max") or (max(n_conflicts_list) if n_conflicts_list else 0),
            "avg": sum(n_conflicts_list) / len(n_conflicts_list) if n_conflicts_list else 0
        },
        "n_style_groups": {
            "min": min(n_style_groups_list) if n_style_groups_list else 0,
            "max": max(n_style_groups_list) if n_style_groups_list else 0,
            "avg": sum(n_style_groups_list) / len(n_style_groups_list) if n_style_groups_list else 0
        },
        "n_expected_styles": {
            "min": min(n_expected_styles_list) if n_expected_styles_list else 0,
            "max": max(n_expected_styles_list) if n_expected_styles_list else 0,
            "avg": sum(n_expected_styles_list) / len(n_expected_styles_list) if n_expected_styles_list else 0
        },
        "style_keys": sorted(list(all_style_keys))
    }

    return metadata


def print_dataset_metadata(metadata: Dict[str, Any]):
    """Print dataset metadata to console."""
    print("=" * 70)
    print("DATASET METADATA")
    print("=" * 70)
    print(f"Total samples:     {metadata['n_samples']}")

    # Print config info if available
    config = metadata.get("config", {})
    if config:
        if "mbpp_source" in config:
            print(f"Source:            {config['mbpp_source']}")
        if "seed" in config:
            print(f"Seed:              {config['seed']}")

    if metadata['n_samples'] > 0:
        instr = metadata.get('n_instructions', {})
        if instr:
            print(f"Instructions:      min={instr.get('min', 'N/A')}, max={instr.get('max', 'N/A')}, avg={instr.get('avg', 0):.1f}")

        conf = metadata.get('n_conflicts', {})
        if conf:
            print(f"Conflicts:         min={conf.get('min', 'N/A')}, max={conf.get('max', 'N/A')}, avg={conf.get('avg', 0):.1f}")

        groups = metadata.get('n_style_groups', {})
        if groups:
            print(f"Style groups:      min={groups.get('min', 'N/A')}, max={groups.get('max', 'N/A')}, avg={groups.get('avg', 0):.1f}")

        expected = metadata.get('n_expected_styles', {})
        if expected:
            print(f"Expected styles:   min={expected.get('min', 'N/A')}, max={expected.get('max', 'N/A')}, avg={expected.get('avg', 0):.1f}")

        style_keys = metadata.get('style_keys', [])
        if style_keys:
            print(f"Style keys:        {', '.join(style_keys)}")
    print("=" * 70)


def analyze_results(results_file: str, save_to_file: bool = True) -> Dict[str, Any]:
    """Analyze an existing results file and print failure statistics.

    Args:
        results_file: Path to results JSON file
        save_to_file: If True, update the results file with analysis

    Returns:
        Dict containing the analysis results
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results:
        print("No results found in file")
        return {}

    # Recompute stats and failure_stats from results
    stats = {"total": 0, "test_passed": 0, "style_passed": 0, "overall_passed": 0}
    failure_stats = {
        "test_failures": 0,
        "style_failures": 0,
        "style_failures_by_category": {},
        "test_failure_reasons": {},
    }

    for result in results:
        stats["total"] += 1
        eval_result = result.get("evaluation", {})

        if eval_result.get('test_passed'):
            stats["test_passed"] += 1
        else:
            failure_stats["test_failures"] += 1
            test_reason = eval_result.get('test_result', 'unknown')
            if 'timed out' in str(test_reason).lower():
                reason_key = 'timeout'
            elif 'assertion' in str(test_reason).lower():
                reason_key = 'assertion_error'
            elif 'syntax' in str(test_reason).lower():
                reason_key = 'syntax_error'
            elif 'error' in str(test_reason).lower():
                reason_key = 'runtime_error'
            else:
                reason_key = 'other'
            failure_stats["test_failure_reasons"][reason_key] = \
                failure_stats["test_failure_reasons"].get(reason_key, 0) + 1

        if eval_result.get('style_passed'):
            stats["style_passed"] += 1
        else:
            failure_stats["style_failures"] += 1
            each_style = eval_result.get('each_style_passed', {})
            for style_key, passed in each_style.items():
                if not passed:
                    failure_stats["style_failures_by_category"][style_key] = \
                        failure_stats["style_failures_by_category"].get(style_key, 0) + 1

        if eval_result.get('overall_passed'):
            stats["overall_passed"] += 1

    # Compute analysis using shared function
    analysis = _compute_analysis(results, stats, failure_stats)

    # Print summary
    _print_analysis(results_file, stats, failure_stats, analysis)

    # Save analysis to file
    if save_to_file:
        data["analysis"] = analysis
        data["stats"] = stats
        data["failure_stats"] = {
            "test_failures": failure_stats["test_failures"],
            "style_failures": failure_stats["style_failures"],
            "style_failures_by_category": failure_stats["style_failures_by_category"],
            "test_failure_reasons": failure_stats["test_failure_reasons"],
        }
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nAnalysis saved to: {results_file}")

    return analysis


def _print_analysis(
    title: str,
    stats: Dict,
    failure_stats: Dict,
    analysis: Dict[str, Any]
):
    """Print analysis summary to console."""
    total = stats['total']
    test_failures = failure_stats.get("test_failures", 0)
    style_failures = failure_stats.get("style_failures", 0)

    print("\n" + "=" * 70)
    print(f"ANALYSIS: {title}")
    print("=" * 70)
    print(f"Total samples:     {total}")
    print(f"Test passed:       {stats['test_passed']} ({analysis['summary']['test_passed_pct']:.1f}%)")
    print(f"Style passed:      {stats['style_passed']} ({analysis['summary']['style_passed_pct']:.1f}%)")
    print(f"Overall passed:    {stats['overall_passed']} ({analysis['summary']['overall_passed_pct']:.1f}%)")

    print("\n" + "-" * 70)
    print("FAILURE ANALYSIS")
    print("-" * 70)

    if test_failures > 0:
        print(f"\nTest Failures: {test_failures} ({analysis['test_failures']['pct_of_total']:.1f}%)")
        print("  Breakdown by reason:")
        for reason, info in analysis["test_failures"]["by_reason"].items():
            print(f"    {reason:20s}: {info['count']:4d} ({info['pct_of_failures']:5.1f}% of test failures, {info['pct_of_total']:5.1f}% of total)")
    else:
        print("\nTest Failures: 0 (0.0%)")

    if style_failures > 0:
        print(f"\nStyle Failures: {style_failures} ({analysis['style_failures']['pct_of_total']:.1f}%)")
        print("  Breakdown by style category:")
        for style_key, info in analysis["style_failures"]["by_category"].items():
            print(f"    {style_key:20s}: {info['count']:4d} ({info['pct_of_failures']:5.1f}% of style failures, {info['pct_of_total']:5.1f}% of total)")
    else:
        print("\nStyle Failures: 0 (0.0%)")

    print("\n" + "-" * 70)
    print("FAILURE SUMMARY")
    print("-" * 70)
    fb = analysis["failure_breakdown"]
    if fb["total_failed"] > 0:
        print(f"Total failed:      {fb['total_failed']} ({fb['total_failed_pct']:.1f}%)")
        print(f"  Only test failed:  {fb['only_test_failed']} ({fb['only_test_failed_pct']:.1f}%)")
        print(f"  Only style failed: {fb['only_style_failed']} ({fb['only_style_failed_pct']:.1f}%)")
        print(f"  Both failed:       {fb['both_failed']} ({fb['both_failed_pct']:.1f}%)")
    else:
        print("All samples passed!")

    # Fine-grained metrics
    fine_grained = analysis.get("fine_grained", {})
    assertion_stats = fine_grained.get("assertion_stats", {})
    style_satisfaction = fine_grained.get("style_satisfaction", {})

    print("\n" + "-" * 70)
    print("FINE-GRAINED METRICS")
    print("-" * 70)

    if assertion_stats.get("avg_pass_rate") is not None:
        print(f"\nAssertion Pass Rate (avg per sample): {assertion_stats['avg_pass_rate']*100:.1f}%")
        print(f"  Total assertions: {assertion_stats['total_assertions']}")
        print(f"  Passed assertions: {assertion_stats['passed_assertions']}")

    if style_satisfaction.get("avg_satisfaction_rate") is not None:
        print(f"\nStyle Satisfaction Rate (avg per sample): {style_satisfaction['avg_satisfaction_rate']*100:.1f}%")
        print("  By category:")
        for style_key, stats in style_satisfaction.get("by_category", {}).items():
            print(f"    {style_key:25s}: {stats['passed']:4d}/{stats['total']:4d} ({stats['rate']*100:.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on the ManyIH coding subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with OpenRouter
  python -m manyih.coding.evaluator --model anthropic/claude-3.5-sonnet

  # Evaluate with Bedrock
  python -m manyih.coding.evaluator --model bedrock:sonnet-4.6

  # Evaluate with local vLLM
  python -m manyih.coding.evaluator --model vllm:localhost:8000:meta-llama/Llama-3.1-8B-Instruct

  # Run on first 10 samples
  python -m manyih.coding.evaluator --max_samples 10

  # Resume interrupted evaluation
  python -m manyih.coding.evaluator --model <model> --resume
        """
    )

    parser.add_argument(
        "--data_file",
        default=None,
        help="Path to dataset JSON (default: bundled coding.json)"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model identifier (OpenRouter, bedrock:<alias>, or vllm:<host>:<port>:<model>)"
    )
    parser.add_argument(
        "--api_key",
        help="API key (or use OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--output",
        help="Output file (default: results/<dataset>_<model>.json)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching"
    )
    parser.add_argument(
        "--cache_dir",
        default=".cache/coding",
        help="Cache directory"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=40960,
        help="Max tokens in response (default: 40960)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Code execution timeout (seconds)"
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Don't include system prompt"
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable extended thinking/reasoning (enabled by default)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Maximum concurrent API calls (default: 50)"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        metavar="RESULTS_FILE",
        help="Analyze an existing results file instead of running evaluation"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help='Provider configuration as JSON string (e.g. \'{"order": ["anthropic"], "allow_fallbacks": false}\')'
    )

    args = parser.parse_args()

    # If --analyze is provided, just analyze the results file and exit
    if args.analyze:
        if not os.path.exists(args.analyze):
            print(f"Error: Results file not found: {args.analyze}")
            sys.exit(1)
        analyze_results(args.analyze)
        sys.exit(0)

    # Resolve data file (default to bundled data)
    data_file = args.data_file or get_default_data_path()

    # Parse model string (detect vllm:host:port:model or bedrock:region:model format)
    parsed = parse_model_string(args.model)
    model = parsed["model"]
    base_url = parsed["base_url"]
    is_local = parsed["is_local"]
    backend = parsed.get("backend", "openrouter")
    region = parsed.get("region")

    if backend == "bedrock":
        print(f"Bedrock mode: model={model}, region={region}")
    elif is_local:
        print(f"vLLM mode: model={model}, base_url={base_url}")

    # Get API key (skip for Bedrock, use dummy for local vLLM models)
    if backend == "bedrock":
        api_key = "unused"  # Bedrock uses AWS credentials
    elif is_local:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or "dummy"
    else:
        api_key = get_api_key(args.api_key)

    # Check data file
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    max_tokens = args.max_tokens
    temperature = args.temperature

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        os.makedirs("results", exist_ok=True)
        dataset_name = Path(data_file).stem
        model_name = args.model.replace('/', '_').replace('-', '_')

        suffix_parts = []
        if args.max_samples is not None:
            suffix_parts.append(f"n{args.max_samples}")
        if temperature != 0.0:
            suffix_parts.append(f"t{temperature}")
        if max_tokens != 40960:
            suffix_parts.append(f"tok{max_tokens}")
        if args.no_cache:
            suffix_parts.append("nocache")
        if args.no_system_prompt:
            suffix_parts.append("nosys")
        if args.no_reasoning:
            suffix_parts.append("noreasoning")
        if args.timeout != 5.0:
            suffix_parts.append(f"timeout{args.timeout}")
        if args.concurrency != 50:
            suffix_parts.append(f"c{args.concurrency}")

        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        output_file = f"results/{dataset_name}_{model_name}{suffix}.json"

    # Parse provider configuration
    provider = None
    if args.provider:
        try:
            provider = json.loads(args.provider)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --provider: {e}")
            sys.exit(1)

    # Run evaluation (async)
    asyncio.run(run_evaluation(
        data_file=data_file,
        model=model,
        api_key=api_key,
        output_file=output_file,
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        delay_between_calls=args.delay,
        resume=args.resume,
        max_tokens=max_tokens,
        temperature=temperature,
        include_system_prompt=not args.no_system_prompt,
        timeout=args.timeout,
        concurrency=args.concurrency,
        reasoning=not args.no_reasoning,
        provider=provider,
        base_url=base_url,
        backend=backend,
        region=region,
    ))


if __name__ == "__main__":
    main()
