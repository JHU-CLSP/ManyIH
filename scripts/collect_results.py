#!/usr/bin/env python3
"""Collect all results in the results/ directory into a summary table.

Scans for:
  - Coding results: results/coding_*.json
  - IF results:     results/if_*/accuracy.json

Usage:
    python scripts/collect_results.py [results_dir]
"""

import argparse
import json
import sys
from pathlib import Path


def load_coding_results(results_dir):
    """Load coding subset results from JSON files."""
    entries = []
    for path in sorted(results_dir.glob("coding_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        stats = data.get("stats", {})
        total = stats.get("total", 0)
        if total == 0:
            continue

        model = data.get("eval_config", {}).get("model", path.stem)
        entries.append({
            "model": model,
            "coding_acc": stats.get("overall_passed", 0) / total * 100,
            "coding_test": stats.get("test_passed", 0) / total * 100,
            "coding_style": stats.get("style_passed", 0) / total * 100,
        })
    return entries


def load_if_results(results_dir):
    """Load IF subset results from accuracy.json files."""
    entries = []
    for path in sorted(results_dir.glob("if_*/accuracy.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        isr = data.get("ISR", {})
        model = path.parent.name.replace("if_", "").replace("_", "/", 1)
        entries.append({
            "model": model,
            "if_acc": isr.get("accuracy", 0) * 100,
        })
    return entries


def merge_results(coding_results, if_results):
    """Merge coding and IF results by model name."""
    models = {}
    for entry in coding_results:
        models.setdefault(entry["model"], {}).update(entry)
    for entry in if_results:
        models.setdefault(entry["model"], {}).update(entry)
    return list(models.values())


def print_table(results):
    """Print a formatted results table."""
    if not results:
        print("No results found.")
        return

    headers = ["Model", "Overall", "Coding", "Test Acc", "Style Acc", "IF"]
    rows = []
    for r in results:
        coding = r.get("coding_acc")
        if_acc = r.get("if_acc")

        # Compute overall as average of available subsets
        available = [v for v in [coding, if_acc] if v is not None]
        overall = sum(available) / len(available) if available else None

        rows.append([
            r["model"],
            f"{overall:.1f}%" if overall is not None else "N/A",
            f"{coding:.1f}%" if coding is not None else "N/A",
            f"{r['coding_test']:.1f}%" if "coding_test" in r else "N/A",
            f"{r['coding_style']:.1f}%" if "coding_style" in r else "N/A",
            f"{if_acc:.1f}%" if if_acc is not None else "N/A",
        ])

    # Sort by overall descending (N/A last)
    def sort_key(row):
        val = row[1]
        return float(val.rstrip("%")) if val != "N/A" else -1
    rows.sort(key=sort_key, reverse=True)

    # Compute column widths
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    # Print
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))


def main():
    parser = argparse.ArgumentParser(description="Collect ManyIH evaluation results into a summary table")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Path to results directory (default: results/)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    coding = load_coding_results(results_dir)
    if_results = load_if_results(results_dir)
    merged = merge_results(coding, if_results)
    print_table(merged)


if __name__ == "__main__":
    main()
