#!/usr/bin/env python3
"""
Evaluation script for the ManyIH instruction-following subset.

Two-stage pipeline:
1. Prediction: generate model responses for each sample
2. Evaluation: score constraint satisfaction using code checkers + LLM judge

Usage:
    # Full pipeline (predict + evaluate)
    python -m manyih.instruction_following.evaluator \
        --model anthropic/claude-3.5-sonnet \
        --judge-model bedrock:sonnet-4.6

    # Prediction only
    python -m manyih.instruction_following.evaluator \
        --model vllm:localhost:8000:Qwen/Qwen3.5-32B --skip-eval

    # Evaluation only (from existing predictions)
    python -m manyih.instruction_following.evaluator \
        --predictions-file results/predictions.json \
        --judge-model bedrock:sonnet-4.6
"""

import argparse
import importlib.resources
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


def get_default_data_path() -> str:
    """Get the path to the bundled instruction-following benchmark data."""
    return str(importlib.resources.files("manyih.data").joinpath("instruction_following.json"))


# ---------------------------------------------------------------------------
# Checkers (inlined from original 1.evaluation_api.py)
# ---------------------------------------------------------------------------

def execute_function(function, response):
    """Execute a code checker function on a model response."""
    try:
        if not response:
            return None, "Empty response"

        local_vars = {"response": response}

        # Execute import statements
        import_statements = re.findall(
            r"^\s*(import\s+\S+|from\s+\S+\s+import\s+\S+)", function, re.MULTILINE
        )
        for statement in import_statements:
            exec(statement, globals())

        exec(function, globals(), local_vars)

        if "check_following" in local_vars:
            result = local_vars["check_following"](local_vars["response"])
            return result, None
        else:
            return None, "check_following not defined"

    except Exception as e:
        error_msg = f"Error executing function: {e}, the function is: {function}"
        logging.error(error_msg)
        return None, error_msg


def code_checker(function, response):
    return execute_function(function, response)


def llm_checker(model, response, prompt, llm_checker_type="llm"):
    try:
        if "{response}" not in prompt:
            prompt += "\n\nHere is model response: {response}"
        prompt = prompt.replace("{response}", response)

        result = model.generate(prompt, temperature=0.0)
        if result and "</think>" in result:
            result = result.split("</think>")[1].strip()
        return result, None
    except Exception as e:
        logging.error(f"LLM checker error: {e}")
        return None, str(e)


# ---------------------------------------------------------------------------
# Stage 1: Prediction
# ---------------------------------------------------------------------------

def _predict_single(entry, model_string, cache_dir, api_key, max_tokens):
    """Process a single prediction entry."""
    from .model_adapter import create_model
    model = create_model(model_string, cache_dir=cache_dir, api_key=api_key)
    input_msgs = entry["input"]

    for item in input_msgs:
        assert item["role"] in ("user", "system", "assistant"), f"Unexpected role: {item['role']}"
    assert input_msgs[-1]["role"] == "user"
    assert input_msgs[-1]["content"] != ""

    response = model.generate_chat(input_msgs, max_tokens=max_tokens)
    if not response:
        return None

    entry["output"] = {
        "content": response.strip(),
        "model_name": model_string
    }
    model.cache.save_cache()
    return entry


def run_prediction(data, model_string, cache_dir=None, api_key=None,
                   max_tokens=40960, num_workers=8):
    """Stage 1: Generate model predictions for each sample.

    Args:
        data: List of sample dicts with "input" key (list of messages).
        model_string: Model identifier string.
        cache_dir: Optional response cache directory.
        api_key: Optional API key (for OpenRouter).
        max_tokens: Maximum tokens for model response.
        num_workers: Number of parallel worker threads.

    Returns:
        List of sample dicts with added "output" key.
    """
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_predict_single, entry, model_string, cache_dir, api_key, max_tokens)
            for entry in data
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"Prediction complete: {len(results)}/{len(data)} samples succeeded")
    return results


# ---------------------------------------------------------------------------
# Stage 2: Evaluation
# ---------------------------------------------------------------------------

def _evaluate_single(args_tuple):
    """Process a single evaluation entry. Runs in a subprocess."""
    entry, model_string, cache_dir, api_key = args_tuple
    from .model_adapter import create_model
    model = create_model(model_string, cache_dir=cache_dir, api_key=api_key)

    try:
        for j, item in enumerate(entry["constraints"]):
            if "score" in entry["constraints"][j]:
                continue

            response = entry["output"]["content"]
            if "</think>" in response:
                response = response.split("</think>")[1].strip()

            eval_details = []

            for e in item["evaluation"]:
                if e["type"] == "llm_conditional_check":
                    condition_response, error = llm_checker(
                        model, response, e["exec"], llm_checker_type="llm_conditional_check"
                    )
                    detail = {"eval_type": "llm_conditional_check", "raw_output": condition_response, "error": error}
                    eval_details.append(detail)
                    if condition_response and "YES" in condition_response:
                        continue
                    else:
                        entry["constraints"][j]["score"] = None
                        break

                if e["type"] == "llm":
                    response, error = llm_checker(model, response, e["exec"], llm_checker_type="llm")
                    detail = {"eval_type": "llm", "raw_output": response, "error": error}
                    eval_details.append(detail)
                    if not response:
                        response = None
                        break
                elif e["type"] == "code":
                    response, error = code_checker(e["exec"], response)
                    detail = {"eval_type": "code", "raw_output": response if not error else None, "error": error}
                    eval_details.append(detail)
                    if error:
                        logging.error(f"Code checker error: {error}")
                        response = None

            entry["constraints"][j]["eval_details"] = eval_details

            if "score" in entry["constraints"][j]:
                continue

            if isinstance(response, str):
                entry["constraints"][j]["llm_output"] = response
                if "YES" in response:
                    response = True
                elif "NO" in response:
                    response = False
                else:
                    logging.warning(f"Unexpected response format: {response}")
                    response = None

            entry["constraints"][j]["score"] = response
        return entry

    except Exception as e:
        logging.error(f"Processing item error: {e}")
        return entry


def run_evaluation(data, model_string, cache_dir=None, api_key=None, num_workers=None):
    """Stage 2: Evaluate constraint satisfaction on predicted data.

    Args:
        data: List of sample dicts with "output" key (from prediction stage).
        model_string: Model identifier for LLM judge.
        cache_dir: Optional response cache directory.
        api_key: Optional API key (for OpenRouter).
        num_workers: Number of parallel worker processes.

    Returns:
        List of evaluated sample dicts with constraint scores.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_evaluate_single, (entry, model_string, cache_dir, api_key))
            for entry in data
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Future error: {e}")

    print(f"Evaluation complete: {len(results)} samples processed")
    return results


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def compute_accuracy(results):
    """Compute ISR (Instruction Success Rate) from evaluated results.

    Returns:
        Dict with ISR accuracy and per-sample details.
    """
    null_counts = 0
    isr_stats = {"success": 0, "total": 0}
    csr_stats = {"success": 0, "total": 0}

    for entry in results:
        constraint_scores = []
        for constraint in entry["constraints"]:
            score = constraint.get("score")
            if score is None:
                null_counts += 1
                continue
            constraint_scores.append(score)

            csr_stats["total"] += 1
            if score is True:
                csr_stats["success"] += 1

        if len(constraint_scores) > 0:
            isr_stats["total"] += 1
            if all(constraint_scores):
                isr_stats["success"] += 1

    accuracy_report = {
        "CSR": {
            "accuracy": csr_stats["success"] / csr_stats["total"] if csr_stats["total"] > 0 else 0,
            "correct": csr_stats["success"],
            "total": csr_stats["total"],
        },
        "ISR": {
            "accuracy": isr_stats["success"] / isr_stats["total"] if isr_stats["total"] > 0 else 0,
            "correct": isr_stats["success"],
            "total": isr_stats["total"],
        },
        "null_constraints": null_counts,
    }

    print(f"\nConstraint Success Rate (CSR): {accuracy_report['CSR']['accuracy']:.4f}")
    print(f"Instruction Success Rate (ISR): {accuracy_report['ISR']['accuracy']:.4f}")
    if null_counts > 0:
        print(f"Null constraints: {null_counts}")

    return accuracy_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on the ManyIH instruction-following subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (predict + evaluate)
  python -m manyih.instruction_following.evaluator \\
      --model anthropic/claude-3.5-sonnet --judge-model bedrock:sonnet-4.6

  # Use same model for both prediction and judging
  python -m manyih.instruction_following.evaluator --model bedrock:opus-4.6

  # Prediction only
  python -m manyih.instruction_following.evaluator \\
      --model vllm:localhost:8000:Qwen/Qwen3.5-32B --skip-eval

  # Evaluation only (from existing predictions)
  python -m manyih.instruction_following.evaluator \\
      --predictions-file results/predictions.json --judge-model bedrock:sonnet-4.6
        """
    )

    parser.add_argument("--data-file", default=None,
                        help="Path to dataset JSON (default: bundled instruction_following.json)")
    parser.add_argument("--model", default=None,
                        help="Model for prediction (OpenRouter, bedrock:<alias>, or vllm:<host>:<port>:<model>)")
    parser.add_argument("--judge-model", default=None,
                        help="Model for LLM judge evaluation (default: same as --model)")
    parser.add_argument("--api-key", default=None,
                        help="API key (or use OPENROUTER_API_KEY env var)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--cache-dir", default=None,
                        help="Response cache directory (default: no caching)")
    parser.add_argument("--max-tokens", type=int, default=40960,
                        help="Max tokens for prediction response (default: 40960)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--predictions-file", default=None,
                        help="Load predictions from file (skip prediction stage)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Only run prediction, skip evaluation")

    args = parser.parse_args()

    # Validate args
    if not args.predictions_file and not args.model:
        parser.error("--model is required unless --predictions-file is provided")

    # Resolve data file
    data_file = args.data_file or get_default_data_path()

    # Set up logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    log_file = output_dir / "eval.log"
    logging.basicConfig(
        filename=str(log_file),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    start_time = time.time()

    # Stage 1: Prediction
    if args.predictions_file:
        print(f"Loading predictions from {args.predictions_file}")
        with open(args.predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} predictions")
    else:
        print(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if args.max_samples:
            data = data[:args.max_samples]

        print(f"\nStage 1: Prediction ({len(data)} samples, model={args.model})")
        data = run_prediction(
            data,
            model_string=args.model,
            cache_dir=args.cache_dir,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            num_workers=args.num_workers,
        )

        # Save predictions
        pred_file = output_dir / "predictions.json"
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Predictions saved to {pred_file}")

    if args.skip_eval:
        print("Skipping evaluation (--skip-eval)")
        return

    # Stage 2: Evaluation
    judge_model = args.judge_model or args.model
    if not judge_model:
        parser.error("--judge-model (or --model) is required for evaluation")

    print(f"\nStage 2: Evaluation ({len(data)} samples, judge={judge_model})")
    results = run_evaluation(
        data,
        model_string=judge_model,
        cache_dir=args.cache_dir,
        api_key=args.api_key,
        num_workers=args.num_workers,
    )

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {results_file}")

    # Compute accuracy
    accuracy = compute_accuracy(results)

    accuracy_file = output_dir / "accuracy.json"
    with open(accuracy_file, 'w', encoding='utf-8') as f:
        json.dump(accuracy, f, indent=4, ensure_ascii=False)
    print(f"Accuracy report saved to {accuracy_file}")

    duration = time.time() - start_time
    print(f"\nTotal time: {duration:.1f}s")


if __name__ == "__main__":
    main()
