#!/usr/bin/env python3
"""
ManyIH benchmark CLI.

Usage:
    python -m manyih evaluate --subset coding --model anthropic/claude-3.5-sonnet
    python -m manyih evaluate --subset if --model bedrock:opus-4.6
    python -m manyih evaluate --subset all --model vllm:localhost:8000:Qwen/Qwen3.5-32B
    python -m manyih view --subset coding --output viewer.html
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    command = sys.argv[1]
    # Remove the subcommand from argv so the sub-parsers see the right args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "evaluate":
        run_evaluate()
    elif command == "view":
        run_view()
    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)


def print_help():
    print("""ManyIH Benchmark — Evaluate LLM instruction hierarchy resolution

Usage: python -m manyih <command> [options]

Commands:
  evaluate    Run model evaluation on a benchmark subset
  view        View benchmark data as HTML

Evaluate examples:
  python -m manyih evaluate --subset coding --model anthropic/claude-3.5-sonnet
  python -m manyih evaluate --subset if --model bedrock:opus-4.6
  python -m manyih evaluate --subset all --model vllm:localhost:8000:Qwen/Qwen3.5-32B

View examples:
  python -m manyih view --subset coding --output coding_viewer.html
  python -m manyih view --subset if --output if_viewer.html

Model string formats:
  anthropic/claude-3.5-sonnet             OpenRouter
  bedrock:<alias>                         AWS Bedrock (e.g. bedrock:sonnet-4.6)
  bedrock:<region>:<model_id>             AWS Bedrock (full)
  vllm:<host>:<port>:<model>              Local vLLM server
""")


def run_evaluate():
    import argparse

    parser = argparse.ArgumentParser(prog="manyih evaluate",
                                     description="Evaluate a model on ManyIH benchmark subsets")
    parser.add_argument("--subset", choices=["coding", "if", "all"], required=True,
                        help="Benchmark subset to evaluate")
    parser.add_argument("--model", required=True,
                        help="Model identifier")

    # Parse only known args — the rest are forwarded to subset evaluators
    args, remaining = parser.parse_known_args()

    if args.subset in ("coding", "all"):
        print("=" * 70)
        print("CODING SUBSET")
        print("=" * 70)
        _run_coding_eval(["--model", args.model] + remaining)

    if args.subset in ("if", "all"):
        print("=" * 70)
        print("INSTRUCTION-FOLLOWING SUBSET")
        print("=" * 70)
        _run_if_eval(["--model", args.model] + remaining)


def _run_coding_eval(argv):
    sys.argv = ["manyih.coding.evaluator"] + argv
    from manyih.coding.evaluator import main as coding_main
    coding_main()


def _run_if_eval(argv):
    sys.argv = ["manyih.instruction_following.evaluator"] + argv
    from manyih.instruction_following.evaluator import main as if_main
    if_main()


def run_view():
    import argparse

    parser = argparse.ArgumentParser(prog="manyih view",
                                     description="View ManyIH benchmark data")
    parser.add_argument("--subset", choices=["coding", "if"], required=True,
                        help="Benchmark subset to view")
    parser.add_argument("--input", default=None,
                        help="Input JSON file (default: bundled data)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output HTML file path")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples to show in terminal")
    parser.add_argument("--summary", action="store_true",
                        help="Show only summary statistics")

    args = parser.parse_args()

    if args.subset == "coding":
        view_args = []
        if args.input:
            view_args += ["--input", args.input]
        if args.output:
            view_args += ["--html", args.output]
        if args.summary:
            view_args += ["--summary"]
        view_args += ["--samples", str(args.samples)]

        sys.argv = ["manyih.coding.viewer"] + view_args
        from manyih.coding.viewer import main as coding_viewer_main
        coding_viewer_main()

    elif args.subset == "if":
        import importlib.resources
        input_path = args.input or str(
            importlib.resources.files("manyih.data").joinpath("instruction_following.json")
        )
        output_path = args.output or "instruction_following_viewer.html"

        sys.argv = ["manyih.instruction_following.viewer", input_path, "-o", output_path]
        from manyih.instruction_following.viewer import main as if_viewer_main
        if_viewer_main()


if __name__ == "__main__":
    main()
