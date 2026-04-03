#!/usr/bin/env python3
"""
Dataset viewer for Conflicting StyleMBPP.

Provides terminal and HTML visualization of generated datasets,
including test cases and style evaluation criteria.
"""

import argparse
import json
from typing import List, Dict, Tuple
from pathlib import Path

# Import style evaluation info from centralized registry
from .styles import STYLE_EVAL_INFO


def load_dataset_file(filepath: str) -> Tuple[List[Dict], Dict, Dict]:
    """Load dataset file, handling both old (list) and new (dict) formats.

    Args:
        filepath: Path to the JSON dataset file.

    Returns:
        Tuple of (dataset, config, statistics) where:
        - dataset: List of sample dictionaries
        - config: Configuration dict (empty for old format)
        - statistics: Pre-computed statistics dict (empty for old format)
    """
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Old format: flat list of samples
        return data, {}, {}
    elif isinstance(data, dict):
        # New format: {"config": ..., "statistics": ..., "data": [...]}
        return data.get("data", []), data.get("config", {}), data.get("statistics", {})
    else:
        raise ValueError(f"Unknown dataset format: {type(data)}")


def view_sample_terminal(sample: Dict, verbose: bool = False) -> str:
    """Format a single sample for terminal display."""
    lines = []
    meta = sample.get("metadata", {})

    # Header
    lines.append("=" * 80)
    lines.append(f"Sample ID: {sample['id']} | Task ID: {sample.get('task_id', 'N/A')}")
    expected_styles = meta.get('expected_styles', {})
    n_expected = sum(len(v) if isinstance(v, list) else 1 for v in expected_styles.values())
    lines.append(f"Instructions: {meta.get('n_instructions', 0)} | "
                 f"Conflicts: {meta.get('n_conflicts', 0)} | "
                 f"Style Groups: {meta.get('n_style_groups', 0)} | "
                 f"Expected Styles: {n_expected}")
    lines.append("-" * 80)

    # Original task
    lines.append("TASK:")
    orig_prompt = sample.get("original_prompt", "")[:200]
    lines.append(f"  {orig_prompt}{'...' if len(sample.get('original_prompt', '')) > 200 else ''}")
    lines.append("")

    # Test cases
    lines.append("TEST CASES:")
    test_code = sample.get("test_code", "")
    for line in test_code.split("\n")[:5]:
        if line.strip():
            lines.append(f"  {line}")
    lines.append("")

    # Instructions grouped by style
    lines.append("STYLE INSTRUCTIONS:")
    instructions = sample.get("instructions", [])

    by_style = {}
    for instr in instructions:
        key = instr.get("style_key", "unknown")
        if key not in by_style:
            by_style[key] = []
        by_style[key].append(instr)

    for style_key, instrs in sorted(by_style.items()):
        if len(instrs) > 1:
            lines.append(f"  {style_key} (CONFLICT):")
        else:
            lines.append(f"  {style_key}:")
        for instr in sorted(instrs, key=lambda x: -x['privilege']):
            winner = " ★" if instr['privilege'] == max(i['privilege'] for i in instrs) else ""
            lines.append(f"    z={instr['privilege']:.2f}: {instr['style_id']}{winner}")

    lines.append("")

    # Expected styles with evaluation info
    lines.append("EXPECTED STYLES & EVALUATION:")
    for key, val in meta.get("expected_styles", {}).items():
        # Handle both old format (string) and new format (list)
        style_ids = val if isinstance(val, list) else [val]
        lines.append(f"  {key}: {', '.join(style_ids)}")
        for style_id in style_ids:
            if style_id in STYLE_EVAL_INFO:
                lines.append(f"    → Check: {STYLE_EVAL_INFO[style_id]['check']}")

    if verbose:
        lines.append("")
        lines.append("-" * 80)
        lines.append("FULL PROMPT:")
        lines.append(sample.get("prompt", ""))

    lines.append("=" * 80)
    return "\n".join(lines)


def view_dataset_summary(dataset: List[Dict], config: Dict = None, statistics: Dict = None) -> str:
    """Generate dataset summary statistics.

    Args:
        dataset: List of sample dictionaries.
        config: Optional configuration dict from new format files.
        statistics: Optional pre-computed statistics from new format files.

    Returns:
        Formatted string with dataset summary.
    """
    if not dataset:
        return "Empty dataset"

    n = len(dataset)
    config = config or {}
    statistics = statistics or {}

    # Use pre-computed stats if available, otherwise compute
    if statistics:
        avg_instr = statistics.get("n_instructions", {}).get("avg", 0)
        avg_conf = statistics.get("n_conflicts", {}).get("avg", 0)
        style_group_pct = statistics.get("style_group_usage_pct", {})
    else:
        total_instr = sum(d['metadata']['n_instructions'] for d in dataset)
        total_conf = sum(d['metadata']['n_conflicts'] for d in dataset)
        avg_instr = total_instr / n
        avg_conf = total_conf / n
        # Compute style_group_pct from samples
        style_group_counts = {}
        for sample in dataset:
            for key in sample['metadata'].get('expected_styles', {}):
                style_group_counts[key] = style_group_counts.get(key, 0) + 1
        style_group_pct = {k: round(v / n * 100, 1) for k, v in style_group_counts.items()}

    lines = [
        "=" * 60,
        "DATASET SUMMARY",
        "=" * 60,
        f"Total samples:        {n}",
    ]

    # Add config info if available
    if config.get("mbpp_source"):
        lines.append(f"Source:               {config['mbpp_source']}")
    if config.get("seed") is not None:
        lines.append(f"Seed:                 {config['seed']}")

    # Compute avg style groups and avg expected styles
    if statistics:
        avg_sg = statistics.get("n_style_groups", {}).get("avg", 0)
        avg_es = statistics.get("n_expected_styles", {}).get("avg", 0)
    else:
        avg_sg = sum(d['metadata'].get('n_style_groups', 0) for d in dataset) / n if n > 0 else 0
        avg_es = sum(
            sum(len(v) if isinstance(v, list) else 1 for v in d['metadata'].get('expected_styles', {}).values())
            for d in dataset
        ) / n if n > 0 else 0

    lines.extend([
        f"Avg instructions:     {avg_instr:.1f}",
        f"Avg conflicts:        {avg_conf:.1f}",
        f"Avg style groups:     {avg_sg:.1f}",
        f"Avg expected styles:  {avg_es:.1f}",
        "",
        "Style group usage:",
    ])

    for key, pct in sorted(style_group_pct.items(), key=lambda x: -x[1]):
        lines.append(f"  {key}: {pct:.1f}%")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_html_report(dataset: List[Dict], output_path: str,
                         config: Dict = None, statistics: Dict = None) -> str:
    """Generate an HTML report for the dataset with test cases and evaluation info.

    Args:
        dataset: List of sample dictionaries.
        output_path: Path to write the HTML report.
        config: Optional configuration dict from new format files.
        statistics: Optional pre-computed statistics from new format files.

    Returns:
        Path to the generated HTML report.
    """
    config = config or {}
    statistics = statistics or {}

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Conflicting StyleMBPP Dataset Viewer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .config-info { margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; }
        .config-info span { display: inline-block; background: #e9ecef; padding: 4px 8px;
                           border-radius: 4px; margin: 3px; font-size: 0.85em; }
        /* Collapsible sample cards */
        details.sample-details { margin-bottom: 15px; }
        details.sample-details > summary {
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); cursor: pointer;
            list-style: none; display: flex; justify-content: space-between; align-items: center;
        }
        details.sample-details > summary::-webkit-details-marker { display: none; }
        details.sample-details > summary::before {
            content: '▶'; margin-right: 10px; transition: transform 0.2s;
        }
        details.sample-details[open] > summary::before { transform: rotate(90deg); }
        details.sample-details[open] > summary { border-radius: 8px 8px 0 0; }
        details.sample-details > .sample-content {
            background: white; padding: 20px; border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: -1px;
        }
        .sample-id { font-size: 1.2em; font-weight: bold; color: #4CAF50; }
        .meta { color: #666; font-size: 0.9em; }
        .task { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 15px;
                border-left: 4px solid #4CAF50; }
        .test-cases { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 4px;
                      margin-bottom: 15px; font-family: 'Consolas', monospace; font-size: 0.85em;
                      overflow-x: auto; }
        .test-cases .keyword { color: #569cd6; }
        .test-cases .function { color: #dcdcaa; }
        .test-cases .string { color: #ce9178; }
        .test-cases .number { color: #b5cea8; }
        .instructions { margin-bottom: 15px; }
        .instruction { padding: 10px 12px; margin: 5px 0; border-radius: 4px; font-size: 0.9em; }
        .conflict { background: #fff3cd; border-left: 3px solid #ffc107; }
        .winner { background: #d4edda; border-left: 3px solid #28a745; }
        .normal { background: #e9ecef; border-left: 3px solid #6c757d; }
        .style-key { font-weight: bold; color: #495057; }
        .privilege { color: #6c757d; font-family: monospace; }
        .expected { background: #e7f3ff; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
        .expected h4 { margin-top: 0; color: #0066cc; }
        .expected-item { display: inline-block; background: #0066cc; color: white;
                        padding: 4px 8px; border-radius: 4px; margin: 3px; font-size: 0.85em; }
        .eval-section { background: #f0f7ff; padding: 15px; border-radius: 4px; margin-bottom: 15px;
                       border: 1px solid #b8d4ff; }
        .eval-section h4 { margin-top: 0; color: #0056b3; font-size: 0.95em; }
        .eval-rule { background: white; padding: 10px; margin: 8px 0; border-radius: 4px;
                    border-left: 3px solid #17a2b8; }
        .eval-rule .rule-name { font-weight: bold; color: #17a2b8; }
        .eval-rule .check { color: #333; margin: 5px 0; }
        .eval-rule .examples { font-size: 0.85em; color: #666; }
        .eval-rule code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px;
                         font-family: monospace; font-size: 0.9em; }
        .pass { color: #28a745; }
        .fail { color: #dc3545; }
        .collapsible { cursor: pointer; user-select: none; padding: 8px; background: #f8f9fa;
                      border-radius: 4px; margin-top: 10px; }
        .collapsible:hover { background: #e9ecef; }
        .content { display: none; padding: 10px; background: #fafafa; border-radius: 4px;
                   margin-top: 10px; white-space: pre-wrap; font-family: monospace; font-size: 0.85em; }
        .show { display: block; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .stat-box { background: #4CAF50; color: white; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; }
        .stat-label { font-size: 0.9em; opacity: 0.9; }
        .reference { background: #fff8e1; padding: 15px; border-radius: 4px; margin-bottom: 15px;
                    border-left: 4px solid #ffc107; }
        .reference h4 { margin-top: 0; color: #856404; }
        .reference pre { background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 4px;
                        overflow-x: auto; font-size: 0.85em; }
        .style-usage { margin-top: 15px; }
        .style-usage h4 { margin-bottom: 10px; color: #495057; }
        .style-bar { display: flex; align-items: center; margin: 5px 0; }
        .style-bar-label { width: 120px; font-size: 0.85em; }
        .style-bar-fill { height: 20px; background: #4CAF50; border-radius: 4px; }
        .style-bar-value { margin-left: 10px; font-size: 0.85em; color: #666; }
    </style>
</head>
<body>
    <h1>🎯 Conflicting StyleMBPP Dataset Viewer</h1>
"""

    # Summary section
    n = len(dataset)
    if statistics:
        avg_instr = statistics.get("n_instructions", {}).get("avg", 0)
        avg_conf = statistics.get("n_conflicts", {}).get("avg", 0)
        style_group_pct = statistics.get("style_group_usage_pct", {})
    else:
        total_instr = sum(d['metadata']['n_instructions'] for d in dataset) if dataset else 0
        total_conf = sum(d['metadata']['n_conflicts'] for d in dataset) if dataset else 0
        avg_instr = total_instr / n if n > 0 else 0
        avg_conf = total_conf / n if n > 0 else 0
        # Compute style_group_pct
        style_group_counts = {}
        for sample in dataset:
            for key in sample['metadata'].get('expected_styles', {}):
                style_group_counts[key] = style_group_counts.get(key, 0) + 1
        style_group_pct = {k: round(v / n * 100, 1) for k, v in style_group_counts.items()} if n > 0 else {}

    # Compute avg style groups and avg expected styles
    if statistics:
        avg_sg = statistics.get("n_style_groups", {}).get("avg", 0)
        avg_es = statistics.get("n_expected_styles", {}).get("avg", 0)
    else:
        avg_sg = sum(d['metadata'].get('n_style_groups', 0) for d in dataset) / n if n > 0 else 0
        avg_es = sum(
            sum(len(v) if isinstance(v, list) else 1 for v in d['metadata'].get('expected_styles', {}).values())
            for d in dataset
        ) / n if n > 0 else 0

    html += f"""
    <div class="summary">
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{n}</div>
                <div class="stat-label">Samples</div>
            </div>
            <div class="stat-box" style="background: #2196F3;">
                <div class="stat-value">{avg_instr:.1f}</div>
                <div class="stat-label">Avg Instructions</div>
            </div>
            <div class="stat-box" style="background: #ff9800;">
                <div class="stat-value">{avg_conf:.1f}</div>
                <div class="stat-label">Avg Conflicts</div>
            </div>
            <div class="stat-box" style="background: #9C27B0;">
                <div class="stat-value">{avg_sg:.1f}</div>
                <div class="stat-label">Avg Style Groups</div>
            </div>
            <div class="stat-box" style="background: #009688;">
                <div class="stat-value">{avg_es:.1f}</div>
                <div class="stat-label">Avg Expected Styles</div>
            </div>
        </div>
"""

    # Add config info if available
    if config:
        html += '        <div class="config-info">\n'
        if config.get("mbpp_source"):
            html += f'            <span><strong>Source:</strong> {config["mbpp_source"]}</span>\n'
        if config.get("seed") is not None:
            html += f'            <span><strong>Seed:</strong> {config["seed"]}</span>\n'
        if config.get("n_style_groups_range"):
            html += f'            <span><strong>Style Groups:</strong> {config["n_style_groups_range"]}</span>\n'
        if config.get("n_instructions_range"):
            html += f'            <span><strong>Instructions:</strong> {config["n_instructions_range"]}</span>\n'
        if config.get("n_conflicts_range"):
            html += f'            <span><strong>Conflicts:</strong> {config["n_conflicts_range"]}</span>\n'
        html += '        </div>\n'

    # Add style usage chart
    if style_group_pct:
        max_pct = max(style_group_pct.values()) if style_group_pct else 100
        html += '        <div class="style-usage">\n'
        html += '            <h4>Style Group Usage:</h4>\n'
        for key, pct in sorted(style_group_pct.items(), key=lambda x: -x[1]):
            bar_width = int((pct / max_pct) * 200) if max_pct > 0 else 0
            html += f'''            <div class="style-bar">
                <span class="style-bar-label">{key}</span>
                <div class="style-bar-fill" style="width: {bar_width}px;"></div>
                <span class="style-bar-value">{pct:.1f}%</span>
            </div>
'''
        html += '        </div>\n'

    html += '    </div>\n'

    # Individual samples - show ALL samples with collapsible details
    for sample in dataset:
        meta = sample.get("metadata", {})
        instructions = sample.get("instructions", [])
        test_code = sample.get("test_code", "")
        reference_code = sample.get("reference_code", "")

        # Group instructions by style
        by_style = {}
        for instr in instructions:
            key = instr.get("style_key", "unknown")
            if key not in by_style:
                by_style[key] = []
            by_style[key].append(instr)

        # Syntax highlight test code
        test_highlighted = test_code.replace("assert", '<span class="keyword">assert</span>')
        test_highlighted = test_highlighted.replace("==", '<span class="keyword">==</span>')
        test_highlighted = test_highlighted.replace("True", '<span class="keyword">True</span>')
        test_highlighted = test_highlighted.replace("False", '<span class="keyword">False</span>')

        expected_styles = meta.get('expected_styles', {})
        n_expected = sum(len(v) if isinstance(v, list) else 1 for v in expected_styles.values())

        html += f"""
    <details class="sample-details">
        <summary>
            <span class="sample-id">Sample #{sample['id']}</span>
            <span class="meta">Task {sample.get('task_id', 'N/A')} |
                {meta.get('n_instructions', 0)} instructions |
                {meta.get('n_conflicts', 0)} conflicts |
                {meta.get('n_style_groups', 0)} style groups |
                {n_expected} expected styles</span>
        </summary>
        <div class="sample-content">
            <div class="task">
                <strong>📝 Task:</strong> {sample.get('original_prompt', '')}
            </div>

            <div class="test-cases">
                <strong>🧪 Test Cases:</strong><br><br>{test_highlighted.replace(chr(10), '<br>')}
            </div>

            <div class="instructions">
                <strong>📋 Style Instructions:</strong>
"""

        for style_key, instrs in sorted(by_style.items()):
            is_conflict = len(instrs) > 1
            max_priv = max(i['privilege'] for i in instrs)

            for instr in sorted(instrs, key=lambda x: -x['privilege']):
                is_winner = instr['privilege'] == max_priv
                css_class = "winner" if is_winner and is_conflict else ("conflict" if is_conflict else "normal")
                winner_mark = " ★ WINNER" if is_winner and is_conflict else ""

                html += f"""
                <div class="instruction {css_class}">
                    <span class="privilege">z={instr['privilege']:.2f}</span>
                    <span class="style-key">[{instr['style_key']}]</span>
                    {instr['description']}{winner_mark}
                </div>
"""

        # Expected styles
        html += """
            </div>
            <div class="expected">
                <h4>🏆 Expected Styles (Winners):</h4>
"""
        for key, val in meta.get("expected_styles", {}).items():
            # Handle both old format (string) and new format (list)
            style_ids = val if isinstance(val, list) else [val]
            html += f'                <span class="expected-item">{key}: {", ".join(style_ids)}</span>\n'

        # Evaluation criteria section
        html += """
            </div>
            <div class="eval-section">
                <h4>📊 Style Evaluation Criteria:</h4>
"""
        for key, val in meta.get("expected_styles", {}).items():
            # Handle both old format (string) and new format (list)
            style_ids = val if isinstance(val, list) else [val]
            for style_id in style_ids:
                if style_id in STYLE_EVAL_INFO:
                    info = STYLE_EVAL_INFO[style_id]
                    html += f"""
                <div class="eval-rule">
                    <div class="rule-name">{key} → {style_id}</div>
                    <div class="check">✓ {info['check']}</div>
                    <div class="examples">
                        <span class="pass">✓ Pass:</span> <code>{info['example_pass'][:50]}</code><br>
                        <span class="fail">✗ Fail:</span> <code>{info['example_fail'][:50]}</code>
                    </div>
                </div>
"""

        # Reference code (collapsible)
        if reference_code:
            ref_escaped = reference_code.replace('<', '&lt;').replace('>', '&gt;')
            html += f"""
            </div>
            <div class="collapsible" onclick="this.nextElementSibling.classList.toggle('show')">
                📖 Click to show reference solution...
            </div>
            <div class="content"><pre>{ref_escaped}</pre></div>
"""
        else:
            html += "            </div>\n"

        # Full prompt (collapsible)
        prompt_escaped = sample.get('prompt', '').replace('<', '&lt;').replace('>', '&gt;')
        html += f"""
            <div class="collapsible" onclick="this.nextElementSibling.classList.toggle('show')">
                📄 Click to show full prompt...
            </div>
            <div class="content">{prompt_escaped}</div>
        </div>
    </details>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="View ManyIH coding subset dataset")
    parser.add_argument("--input", type=str, help="Input JSON dataset file (default: bundled coding.json)")
    parser.add_argument("--html", type=str, help="Output HTML report path")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples to display in terminal")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full prompts in terminal")
    parser.add_argument("--summary", action="store_true",
                        help="Show only summary statistics")
    args = parser.parse_args()

    if args.input:
        input_path = args.input
    else:
        import importlib.resources
        input_path = str(importlib.resources.files("manyih.data").joinpath("coding.json"))

    dataset, config, statistics = load_dataset_file(input_path)
    print(f"Loaded {len(dataset)} samples from {input_path}\n")

    print(view_dataset_summary(dataset, config, statistics))
    print()

    if not args.summary:
        n_show = min(args.samples, len(dataset))
        print(f"Showing {n_show} sample(s):\n")
        for sample in dataset[:n_show]:
            print(view_sample_terminal(sample, verbose=args.verbose))
            print()

    if args.html:
        output = generate_html_report(dataset, args.html, config, statistics)
        print(f"\nHTML report saved to: {output}")


if __name__ == "__main__":
    main()
