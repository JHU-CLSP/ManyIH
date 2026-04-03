#!/usr/bin/env python3
"""AgentIF Dataset Diff Viewer.

Generates an interactive HTML report comparing an original AgentIF dataset
with its hierarchy-injected version, showing system prompt diffs, constraint
changes, and conflict group structure.

Usage:
    # Diff mode (before vs after):
    python ext/agentif/viewer.py \
        --before ext/agentif/data/eval-s205_10ex.json \
        --after ext/agentif/data/pipeline/s205_10ex-opus46/hierarchy/eval_with_hierarchy.json \
        -o diff_report.html

    # Single-file mode (view one dataset):
    python ext/agentif/viewer.py \
        ext/agentif/data/pipeline/0309-30sample/hierarchy/eval_with_hierarchy.json \
        -o viewer.html
"""

import argparse
import difflib
import html
import json
from collections import Counter
from pathlib import Path


def load_dataset(path):
    """Load a JSON dataset (expected to be a list of sample dicts)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"Unexpected dataset format in {path}")


def match_samples(before, after):
    """Pair samples by 'id' (if present) or by index. Returns list of (before_sample, after_sample)."""
    has_ids = all("id" in s for s in before) and all("id" in s for s in after)
    if has_ids:
        after_by_id = {s["id"]: s for s in after}
        pairs = []
        for b in before:
            sid = b["id"]
            if sid not in after_by_id:
                raise ValueError(f"Sample id={sid} not found in 'after' dataset")
            pairs.append((b, after_by_id[sid]))
        return pairs
    # Fall back to index-based matching
    if len(before) != len(after):
        raise ValueError(f"Datasets have different lengths ({len(before)} vs {len(after)}) and no 'id' field for matching")
    return list(zip(before, after))


def compute_summary_stats(pairs):
    """Compute aggregate statistics across all paired samples."""
    stats = {
        "n_samples": len(pairs),
        "before_constraints": [],
        "after_active": [],
        "after_suppressed": [],
        "after_generated": [],
        "n_conflict_groups": [],
        "conflictable_counts": [],
        "non_conflictable_counts": [],
    }
    for b, a in pairs:
        stats["before_constraints"].append(len(b.get("constraints", [])))
        active = a.get("constraints", [])
        suppressed = a.get("suppressed_constraints", [])
        stats["after_active"].append(len(active))
        stats["after_suppressed"].append(len(suppressed))
        generated = [c for c in active + suppressed if c.get("other_info", {}).get("is_generated_conflict")]
        stats["after_generated"].append(len(generated))
        hmeta = a.get("hierarchy_metadata", {})
        stats["n_conflict_groups"].append(hmeta.get("n_conflict_groups", 0))
        # Count conflictable vs not among active constraints
        conflictable = sum(1 for c in active if c.get("conflictable", {}).get("is_conflictable"))
        non_conflictable = sum(1 for c in active if not c.get("conflictable", {}).get("is_conflictable"))
        stats["conflictable_counts"].append(conflictable)
        stats["non_conflictable_counts"].append(non_conflictable)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "n_samples": stats["n_samples"],
        "avg_before_constraints": avg(stats["before_constraints"]),
        "avg_after_active": avg(stats["after_active"]),
        "avg_after_suppressed": avg(stats["after_suppressed"]),
        "avg_generated": avg(stats["after_generated"]),
        "avg_conflict_groups": avg(stats["n_conflict_groups"]),
        "total_conflict_groups": sum(stats["n_conflict_groups"]),
        "total_suppressed": sum(stats["after_suppressed"]),
        "avg_conflictable": avg(stats["conflictable_counts"]),
        "avg_non_conflictable": avg(stats["non_conflictable_counts"]),
    }


def diff_text(before_text, after_text):
    """Generate an inline HTML diff of two text blocks using SequenceMatcher."""
    before_lines = before_text.splitlines(keepends=True)
    after_lines = after_text.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(None, before_lines, after_lines)

    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in before_lines[i1:i2]:
                parts.append(f'<div class="diff-line diff-same">{html.escape(line.rstrip())}</div>')
        elif tag == "insert":
            for line in after_lines[j1:j2]:
                parts.append(f'<div class="diff-line diff-add">{html.escape(line.rstrip())}</div>')
        elif tag == "delete":
            for line in before_lines[i1:i2]:
                parts.append(f'<div class="diff-line diff-del">{html.escape(line.rstrip())}</div>')
        elif tag == "replace":
            for line in before_lines[i1:i2]:
                parts.append(f'<div class="diff-line diff-del">{html.escape(line.rstrip())}</div>')
            for line in after_lines[j1:j2]:
                parts.append(f'<div class="diff-line diff-add">{html.escape(line.rstrip())}</div>')
    return "\n".join(parts)


def privilege_color(z):
    """Return a CSS color for a privilege level."""
    if z >= 75:
        return "#1b7a2b"
    elif z >= 50:
        return "#2d6a4f"
    elif z >= 25:
        return "#b8860b"
    else:
        return "#c0392b"


def constraint_status(c):
    """Determine the display status of a constraint in the 'after' dataset."""
    is_generated = c.get("other_info", {}).get("is_generated_conflict", False)
    is_suppressed = c.get("suppressed", False)
    is_winner = c.get("is_winner", False)

    if is_generated and is_winner:
        return "injected-winner"
    elif is_generated and is_suppressed:
        return "injected-suppressed"
    elif is_winner:
        return "winner"
    elif is_suppressed:
        return "suppressed"
    elif c.get("conflict_group_id"):
        return "in-conflict"
    else:
        return "unchanged"


STATUS_LABELS = {
    "winner": ("Winner", "#28a745", "#d4edda"),
    "suppressed": ("Suppressed", "#dc3545", "#f8d7da"),
    "injected-winner": ("Injected Winner", "#28a745", "#d4edda"),
    "injected-suppressed": ("Injected Suppressed", "#e67e22", "#fff3cd"),
    "in-conflict": ("In Conflict", "#0066cc", "#e7f3ff"),
    "unchanged": ("Unchanged", "#6c757d", "#e9ecef"),
}


def render_constraint_row(c, status):
    """Render a single constraint as a table row with expandable details."""
    label_text, label_color, bg_color = STATUS_LABELS[status]
    desc = html.escape(c.get("desc", ""))
    truncated = desc[:120] + "..." if len(desc) > 120 else desc
    full_json = html.escape(json.dumps(c, indent=2, ensure_ascii=False))

    c_types = c.get("type", [])
    if isinstance(c_types, str):
        c_types = [c_types]
    ctype = ", ".join(c_types)
    conflictable = c.get("conflictable", {})
    conflictable_str = ""
    if conflictable:
        if conflictable.get("is_conflictable"):
            conflictable_str = '<span style="color:#28a745">Yes</span>'
        else:
            conflictable_str = '<span style="color:#999">No</span>'

    priv = c.get("privilege")
    priv_html = ""
    if priv is not None:
        color = privilege_color(priv)
        priv_html = f'<span style="color:{color};font-weight:bold">z={priv}</span>'

    group = c.get("conflict_group_id") or ""
    cid = c.get("id", "")

    text_decoration = "line-through" if "suppressed" in status.lower() else "none"

    return f"""<tr style="background:{bg_color}">
        <td style="font-family:monospace">{cid}</td>
        <td style="text-decoration:{text_decoration}">
            <details class="constraint-detail">
                <summary>{truncated}</summary>
                <pre class="constraint-json">{full_json}</pre>
            </details>
        </td>
        <td>{ctype}</td>
        <td>{conflictable_str}</td>
        <td>{priv_html}</td>
        <td style="font-family:monospace">{html.escape(str(group))}</td>
        <td><span style="color:{label_color};font-weight:bold">{label_text}</span></td>
    </tr>"""


def render_conflict_group(group, constraint_lookup):
    """Render a conflict group card."""
    group_id = group["group_id"]
    member_ids = group["member_ids"]
    privileges = group.get("privileges", {})
    winner_id = group.get("winner_id")

    # Sort members by privilege descending
    members = []
    for mid in member_ids:
        mid_str = str(mid)
        priv = privileges.get(mid_str, 0)
        c = constraint_lookup.get(mid, {})
        members.append((priv, mid, c))
    members.sort(key=lambda x: -x[0])

    rows = []
    for priv, mid, c in members:
        is_winner = mid == winner_id
        is_generated = c.get("other_info", {}).get("is_generated_conflict", False)
        desc = html.escape(c.get("desc", f"[constraint {mid}]"))
        truncated = desc[:100] + "..." if len(desc) > 100 else desc

        color = privilege_color(priv)
        bg = "#d4edda" if is_winner else "#f8f9fa"
        border_left = f"4px solid {color}"
        badges = []
        if is_winner:
            badges.append('<span class="badge badge-winner">WINNER</span>')
        if is_generated:
            badges.append('<span class="badge badge-injected">injected</span>')
        badge_html = " ".join(badges)

        text_style = "text-decoration:line-through;opacity:0.7" if not is_winner else ""

        rows.append(f"""<div class="group-member" style="background:{bg};border-left:{border_left};{text_style}">
            <span class="priv-tag" style="color:{color}">z={priv}</span>
            <span class="member-id">#{mid}</span>
            <span class="member-desc">{truncated}</span>
            {badge_html}
        </div>""")

    return f"""<div class="conflict-group-card">
        <div class="group-header">{html.escape(group_id)} &mdash; {len(member_ids)} members</div>
        {"".join(rows)}
    </div>"""


def render_sample_card(b, a, index):
    """Render a full sample comparison card."""
    sid = b.get("id", index)
    agent = html.escape(b.get("agent_name", ""))
    n_before = len(b.get("constraints", []))
    active = a.get("constraints", [])
    suppressed = a.get("suppressed_constraints", [])
    n_active = len(active)
    n_suppressed = len(suppressed)
    hmeta = a.get("hierarchy_metadata", {})
    n_groups = hmeta.get("n_conflict_groups", 0)

    # Per-message input diffs
    before_input = b.get("input", [])
    after_input = a.get("input", [])
    input_diffs_html = ""
    for i in range(max(len(before_input), len(after_input))):
        b_msg = before_input[i] if i < len(before_input) else {}
        a_msg = after_input[i] if i < len(after_input) else {}
        role = a_msg.get("role", b_msg.get("role", f"message {i}"))
        b_content = b_msg.get("content", "")
        a_content = a_msg.get("content", "")
        msg_diff = diff_text(b_content, a_content)
        has_changes = b_content != a_content
        change_badge = (' <span style="color:#e67e22;font-size:0.8em">(modified)</span>'
                        if has_changes else
                        ' <span style="color:#999;font-size:0.8em">(unchanged)</span>')
        open_attr = " open" if has_changes else ""
        input_diffs_html += f'''<details class="inner-section"{open_attr}>
            <summary>{html.escape(role)} message{change_badge}</summary>
            <div class="diff-view">{msg_diff}</div>
        </details>\n'''
    constraints_diff = diff_text(
        json.dumps(b.get("constraints", []), indent=2, ensure_ascii=False),
        json.dumps(a.get("constraints", []), indent=2, ensure_ascii=False),
    )
    # After-only sections
    suppressed_json = html.escape(json.dumps(a.get("suppressed_constraints", []), indent=2, ensure_ascii=False))
    hierarchy_json = html.escape(json.dumps(a.get("hierarchy_metadata", {}), indent=2, ensure_ascii=False))

    # Build constraint lookup from 'after' (active + suppressed)
    constraint_lookup = {}
    all_after = []
    for c in active:
        constraint_lookup[c["id"]] = c
        all_after.append((c, constraint_status(c)))
    for c in suppressed:
        constraint_lookup[c["id"]] = c
        all_after.append((c, constraint_status(c)))

    # Sort: by conflict_group (non-null first), then by privilege desc
    def sort_key(item):
        c, st = item
        group = c.get("conflict_group_id") or "zzz"
        priv = -(c.get("privilege") or 0)
        return (group, priv)
    all_after.sort(key=sort_key)

    constraint_rows = "\n".join(render_constraint_row(c, st) for c, st in all_after)

    # Conflict groups
    groups_html = ""
    for group in hmeta.get("conflict_groups", []):
        groups_html += render_conflict_group(group, constraint_lookup)

    return f"""<details class="sample-card">
    <summary class="sample-summary">
        <span class="sample-id">#{sid}</span>
        <span class="sample-agent">{agent}</span>
        <span class="sample-stats">
            {n_before} &rarr; {n_active} active + {n_suppressed} suppressed
            &nbsp;|&nbsp; {n_groups} conflict group{"s" if n_groups != 1 else ""}
        </span>
    </summary>
    <div class="sample-content">

        <details class="inner-section" open>
            <summary>Input Messages</summary>
            <div style="padding:8px">{input_diffs_html}</div>
        </details>

        <details class="inner-section">
            <summary>Constraints Diff</summary>
            <div class="diff-view">{constraints_diff}</div>
        </details>

        <details class="inner-section">
            <summary>Suppressed Constraints (after only)</summary>
            <div class="diff-view"><pre>{suppressed_json}</pre></div>
        </details>

        <details class="inner-section">
            <summary>Hierarchy Metadata (after only)</summary>
            <div class="diff-view"><pre>{hierarchy_json}</pre></div>
        </details>

        <details class="inner-section" open>
            <summary>Constraints ({n_active} active + {n_suppressed} suppressed)</summary>
            <div class="table-wrapper">
            <table class="constraint-table">
                <thead><tr>
                    <th>ID</th><th>Description</th><th>Type</th>
                    <th>Conflictable</th><th>Privilege</th><th>Group</th><th>Status</th>
                </tr></thead>
                <tbody>{constraint_rows}</tbody>
            </table>
            </div>
        </details>

        <details class="inner-section" open>
            <summary>Conflict Groups ({n_groups})</summary>
            <div class="conflict-groups">{groups_html if groups_html else "<p style='color:#999'>No conflict groups</p>"}</div>
        </details>

    </div>
</details>"""


def compute_single_stats(samples):
    """Compute aggregate statistics for a single dataset (no before/after)."""
    stats = {
        "n_samples": len(samples),
        "active_counts": [],
        "suppressed_counts": [],
        "generated_counts": [],
        "n_conflict_groups": [],
        "conflictable_counts": [],
        "non_conflictable_counts": [],
    }
    for s in samples:
        active = s.get("constraints", [])
        suppressed = s.get("suppressed_constraints", [])
        stats["active_counts"].append(len(active))
        stats["suppressed_counts"].append(len(suppressed))
        generated = [c for c in active + suppressed if c.get("other_info", {}).get("is_generated_conflict")]
        stats["generated_counts"].append(len(generated))
        hmeta = s.get("hierarchy_metadata", {})
        stats["n_conflict_groups"].append(hmeta.get("n_conflict_groups", 0))
        conflictable = sum(1 for c in active if c.get("conflictable", {}).get("is_conflictable"))
        non_conflictable = sum(1 for c in active if not c.get("conflictable", {}).get("is_conflictable"))
        stats["conflictable_counts"].append(conflictable)
        stats["non_conflictable_counts"].append(non_conflictable)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "n_samples": stats["n_samples"],
        "avg_active": avg(stats["active_counts"]),
        "avg_suppressed": avg(stats["suppressed_counts"]),
        "avg_generated": avg(stats["generated_counts"]),
        "avg_conflict_groups": avg(stats["n_conflict_groups"]),
        "total_conflict_groups": sum(stats["n_conflict_groups"]),
        "total_suppressed": sum(stats["suppressed_counts"]),
        "avg_conflictable": avg(stats["conflictable_counts"]),
        "avg_non_conflictable": avg(stats["non_conflictable_counts"]),
    }


def render_single_sample_card(s, index):
    """Render a sample card for single-file mode (no diff)."""
    sid = s.get("id", index)
    agent = html.escape(s.get("agent_name", ""))
    active = s.get("constraints", [])
    suppressed = s.get("suppressed_constraints", [])
    n_active = len(active)
    n_suppressed = len(suppressed)
    hmeta = s.get("hierarchy_metadata", {})
    n_groups = hmeta.get("n_conflict_groups", 0)

    # Input messages
    input_msgs = s.get("input", [])
    input_html = ""
    for i, msg in enumerate(input_msgs):
        role = msg.get("role", f"message {i}")
        content = msg.get("content", "")
        escaped = html.escape(content)
        input_html += f'''<details class="inner-section">
            <summary>{html.escape(role)} message</summary>
            <div class="diff-view"><pre style="white-space:pre-wrap;word-break:break-word">{escaped}</pre></div>
        </details>\n'''

    # Output
    output = s.get("output", {})
    output_content = output.get("content", "") if isinstance(output, dict) else str(output)
    output_html = html.escape(output_content)

    # Constraints JSON
    constraints_json = html.escape(json.dumps(active, indent=2, ensure_ascii=False))
    suppressed_json = html.escape(json.dumps(suppressed, indent=2, ensure_ascii=False))
    hierarchy_json = html.escape(json.dumps(hmeta, indent=2, ensure_ascii=False))

    # Build constraint lookup and table
    constraint_lookup = {}
    all_constraints = []
    for c in active:
        constraint_lookup[c["id"]] = c
        all_constraints.append((c, constraint_status(c)))
    for c in suppressed:
        constraint_lookup[c["id"]] = c
        all_constraints.append((c, constraint_status(c)))

    def sort_key(item):
        c, st = item
        group = c.get("conflict_group_id") or "zzz"
        priv = -(c.get("privilege") or 0)
        return (group, priv)
    all_constraints.sort(key=sort_key)

    constraint_rows = "\n".join(render_constraint_row(c, st) for c, st in all_constraints)

    # Conflict groups
    groups_html = ""
    for group in hmeta.get("conflict_groups", []):
        groups_html += render_conflict_group(group, constraint_lookup)

    # Verification
    verification = s.get("verification")
    verification_html = ""
    if verification:
        verification_json = html.escape(json.dumps(verification, indent=2, ensure_ascii=False))
        verification_html = f'''<details class="inner-section">
            <summary>Verification</summary>
            <div class="diff-view"><pre>{verification_json}</pre></div>
        </details>'''

    return f"""<details class="sample-card">
    <summary class="sample-summary">
        <span class="sample-id">#{sid}</span>
        <span class="sample-agent">{agent}</span>
        <span class="sample-stats">
            {n_active} active + {n_suppressed} suppressed
            &nbsp;|&nbsp; {n_groups} conflict group{"s" if n_groups != 1 else ""}
        </span>
    </summary>
    <div class="sample-content">

        <details class="inner-section">
            <summary>Input Messages ({len(input_msgs)})</summary>
            <div style="padding:8px">{input_html}</div>
        </details>

        <details class="inner-section">
            <summary>Expected Output</summary>
            <div class="diff-view"><pre style="white-space:pre-wrap;word-break:break-word">{output_html}</pre></div>
        </details>

        <details class="inner-section" open>
            <summary>Constraints ({n_active} active + {n_suppressed} suppressed)</summary>
            <div class="table-wrapper">
            <table class="constraint-table">
                <thead><tr>
                    <th>ID</th><th>Description</th><th>Type</th>
                    <th>Conflictable</th><th>Privilege</th><th>Group</th><th>Status</th>
                </tr></thead>
                <tbody>{constraint_rows}</tbody>
            </table>
            </div>
        </details>

        <details class="inner-section" open>
            <summary>Conflict Groups ({n_groups})</summary>
            <div class="conflict-groups">{groups_html if groups_html else "<p style='color:#999'>No conflict groups</p>"}</div>
        </details>

        {verification_html}

        <details class="inner-section">
            <summary>Suppressed Constraints JSON</summary>
            <div class="diff-view"><pre>{suppressed_json}</pre></div>
        </details>

        <details class="inner-section">
            <summary>Hierarchy Metadata JSON</summary>
            <div class="diff-view"><pre>{hierarchy_json}</pre></div>
        </details>

    </div>
</details>"""


def generate_html_report(pairs, stats, before_path, after_path, output_path):
    """Assemble and write the full HTML report."""

    sample_cards = "\n".join(
        render_sample_card(b, a, i) for i, (b, a) in enumerate(pairs)
    )

    html_content = _render_html_page(
        title="AgentIF Hierarchy Diff Viewer",
        subtitle=f'<strong>Before:</strong> <code>{html.escape(str(before_path))}</code><br>'
                 f'<strong>After:</strong> <code>{html.escape(str(after_path))}</code>',
        stats_html=_render_diff_stats(stats),
        sample_cards=sample_cards,
    )

    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path


def generate_single_html_report(samples, data_path, output_path):
    """Assemble and write an HTML report for a single dataset."""
    stats = compute_single_stats(samples)

    sample_cards = "\n".join(
        render_single_sample_card(s, i) for i, s in enumerate(samples)
    )

    stats_html = f"""
    <div class="stat-box"><div class="stat-value">{stats['n_samples']}</div><div class="stat-label">Samples</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_active']:.1f}</div><div class="stat-label">Avg Active Constraints</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_suppressed']:.1f}</div><div class="stat-label">Avg Suppressed</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_generated']:.1f}</div><div class="stat-label">Avg Injected Conflicts</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_conflict_groups']:.1f}</div><div class="stat-label">Avg Conflict Groups</div></div>
    <div class="stat-box"><div class="stat-value">{stats['total_conflict_groups']}</div><div class="stat-label">Total Conflict Groups</div></div>
    <div class="stat-box"><div class="stat-value">{stats['total_suppressed']}</div><div class="stat-label">Total Suppressed</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_conflictable']:.1f}</div><div class="stat-label">Avg Conflictable</div></div>
    """

    html_content = _render_html_page(
        title="AgentIF Dataset Viewer",
        subtitle=f'<strong>Dataset:</strong> <code>{html.escape(str(data_path))}</code>',
        stats_html=stats_html,
        sample_cards=sample_cards,
    )

    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path


def _render_diff_stats(stats):
    """Render stats boxes for diff mode."""
    return f"""
    <div class="stat-box"><div class="stat-value">{stats['n_samples']}</div><div class="stat-label">Samples</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_before_constraints']:.1f}</div><div class="stat-label">Avg Original Constraints</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_after_active']:.1f}</div><div class="stat-label">Avg Active (After)</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_after_suppressed']:.1f}</div><div class="stat-label">Avg Suppressed (After)</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_generated']:.1f}</div><div class="stat-label">Avg Injected Conflicts</div></div>
    <div class="stat-box"><div class="stat-value">{stats['avg_conflict_groups']:.1f}</div><div class="stat-label">Avg Conflict Groups</div></div>
    <div class="stat-box"><div class="stat-value">{stats['total_conflict_groups']}</div><div class="stat-label">Total Conflict Groups</div></div>
    <div class="stat-box"><div class="stat-value">{stats['total_suppressed']}</div><div class="stat-label">Total Suppressed</div></div>
    """


def _render_html_page(title, subtitle, stats_html, sample_cards):
    """Render the full HTML page shell (shared between diff and single modes)."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5; color: #333; padding: 20px; max-width: 1400px; margin: 0 auto;
    line-height: 1.5;
}}
h1 {{ margin-bottom: 8px; color: #1a1a2e; }}
.subtitle {{ color: #666; margin-bottom: 24px; font-size: 0.9em; word-break: break-all; }}
.subtitle code {{ background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }}

/* Summary stats grid */
.stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 24px;
}}
.stat-box {{
    background: #fff; border-radius: 8px; padding: 16px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.stat-value {{ font-size: 1.8em; font-weight: bold; color: #1a1a2e; }}
.stat-label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}

/* Sample cards */
.sample-card {{
    background: #fff; border-radius: 8px; margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden;
}}
.sample-card > summary {{
    padding: 14px 20px; cursor: pointer; display: flex; align-items: center;
    gap: 12px; font-size: 0.95em; list-style: none;
    border-left: 4px solid #1a1a2e; user-select: none;
}}
.sample-card > summary::-webkit-details-marker {{ display: none; }}
.sample-card > summary::before {{
    content: "\\25B6"; font-size: 0.7em; transition: transform 0.2s; color: #999;
}}
.sample-card[open] > summary::before {{ transform: rotate(90deg); }}
.sample-card > summary:hover {{ background: #f8f9fa; }}
.sample-id {{ font-weight: bold; font-family: monospace; color: #1a1a2e; min-width: 50px; }}
.sample-agent {{ color: #555; font-weight: 500; }}
.sample-stats {{ margin-left: auto; color: #888; font-size: 0.85em; white-space: nowrap; }}

.sample-content {{ padding: 0 20px 20px; }}

/* Inner sections */
.inner-section {{
    margin-top: 16px; border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden;
}}
.inner-section > summary {{
    padding: 10px 16px; cursor: pointer; font-weight: 600; font-size: 0.9em;
    background: #f8f9fa; border-bottom: 1px solid #e0e0e0; list-style: none;
    user-select: none;
}}
.inner-section > summary::-webkit-details-marker {{ display: none; }}
.inner-section > summary::before {{
    content: "\\25B6"; font-size: 0.6em; margin-right: 8px; display: inline-block;
    transition: transform 0.2s; color: #999;
}}
.inner-section[open] > summary::before {{ transform: rotate(90deg); }}
.inner-section > summary:hover {{ background: #eee; }}
.inner-section .inner-section {{ margin-top: 8px; }}

/* Diff view */
.diff-view {{
    max-height: 500px; overflow: auto; font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 0.82em; padding: 8px;
}}
.diff-line {{ white-space: pre-wrap; word-break: break-word; padding: 1px 8px; min-height: 1.3em; }}
.diff-same {{ color: #555; }}
.diff-add {{ background: #d4edda; border-left: 3px solid #28a745; }}
.diff-del {{ background: #f8d7da; border-left: 3px solid #dc3545; text-decoration: line-through; opacity: 0.7; }}

/* Constraint detail expandable */
.constraint-detail > summary {{ cursor: pointer; }}
.constraint-detail > summary::-webkit-details-marker {{ display: none; }}
.constraint-detail > summary::before {{
    content: "\\25B6"; font-size: 0.6em; margin-right: 6px; display: inline-block;
    transition: transform 0.2s; color: #999;
}}
.constraint-detail[open] > summary::before {{ transform: rotate(90deg); }}
.constraint-json {{
    margin-top: 8px; padding: 10px; background: #f8f9fa; border: 1px solid #e0e0e0;
    border-radius: 4px; font-size: 0.82em; font-family: 'SFMono-Regular', Consolas, monospace;
    white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow: auto;
}}

/* Constraint table */
.table-wrapper {{ overflow-x: auto; }}
.constraint-table {{
    width: 100%; border-collapse: collapse; font-size: 0.85em;
}}
.constraint-table th {{
    text-align: left; padding: 8px 10px; background: #343a40; color: #fff;
    font-weight: 600; position: sticky; top: 0;
}}
.constraint-table td {{ padding: 8px 10px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }}
.constraint-table td:nth-child(2) {{ max-width: 400px; }}
.constraint-table tr:hover {{ filter: brightness(0.97); }}

/* Conflict groups */
.conflict-groups {{ padding: 12px; }}
.conflict-group-card {{
    border: 1px solid #dee2e6; border-radius: 6px; margin-bottom: 12px; overflow: hidden;
}}
.group-header {{
    padding: 8px 14px; font-weight: 600; font-size: 0.9em;
    background: #343a40; color: #fff;
}}
.group-member {{
    padding: 8px 14px; display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid #eee; font-size: 0.85em;
}}
.group-member:last-child {{ border-bottom: none; }}
.priv-tag {{
    font-family: monospace; font-weight: bold; font-size: 0.95em; min-width: 42px;
}}
.member-id {{ font-family: monospace; color: #666; min-width: 45px; }}
.member-desc {{ flex: 1; }}
.badge {{
    padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: bold; text-transform: uppercase;
}}
.badge-winner {{ background: #28a745; color: #fff; }}
.badge-injected {{ background: #ffc107; color: #333; }}

/* Filter controls */
.filter-bar {{
    margin-bottom: 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
}}
.filter-bar label {{ font-size: 0.85em; color: #555; }}
.filter-bar input[type="text"] {{
    padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85em; width: 250px;
}}
.filter-bar button {{
    padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85em;
    cursor: pointer; background: #fff;
}}
.filter-bar button:hover {{ background: #e9ecef; }}
.filter-bar button.active {{ background: #343a40; color: #fff; border-color: #343a40; }}
</style>
</head>
<body>

<h1>{html.escape(title)}</h1>
<div class="subtitle">{subtitle}</div>

<div class="stats-grid">
{stats_html}
</div>

<div class="filter-bar">
    <label>Search:</label>
    <input type="text" id="searchInput" placeholder="Filter by ID or agent name..." oninput="filterSamples()">
    <button onclick="expandAll()" title="Expand all sample cards">Expand All</button>
    <button onclick="collapseAll()" title="Collapse all sample cards">Collapse All</button>
</div>

<div id="sampleContainer">
{sample_cards}
</div>

<script>
function filterSamples() {{
    const q = document.getElementById('searchInput').value.toLowerCase();
    document.querySelectorAll('.sample-card').forEach(card => {{
        const text = card.querySelector('.sample-summary').textContent.toLowerCase();
        card.style.display = text.includes(q) ? '' : 'none';
    }});
}}
function expandAll() {{
    document.querySelectorAll('.sample-card').forEach(d => d.open = true);
}}
function collapseAll() {{
    document.querySelectorAll('.sample-card').forEach(d => d.open = false);
}}
</script>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="AgentIF Dataset Viewer")
    parser.add_argument("dataset", nargs="?", default=None,
                        help="Path to dataset JSON (single-file mode)")
    parser.add_argument("--before", default=None, help="Path to original dataset JSON (diff mode)")
    parser.add_argument("--after", default=None, help="Path to hierarchy-injected dataset JSON (diff mode)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file path (default: auto-named in same directory)")
    args = parser.parse_args()

    # Determine mode
    if args.dataset and not args.before and not args.after:
        # Single-file mode
        data_path = args.dataset
        output = args.output or str(Path(data_path).with_suffix(".html"))
        print(f"Loading dataset: {data_path}")
        samples = load_dataset(data_path)
        print(f"Loaded {len(samples)} samples")
        print(f"Generating HTML report -> {output}")
        generate_single_html_report(samples, data_path, output)
        print("Done!")
    elif args.before and args.after:
        # Diff mode
        output = args.output or "agentif_diff_report.html"
        print(f"Loading before: {args.before}")
        before = load_dataset(args.before)
        print(f"Loading after:  {args.after}")
        after = load_dataset(args.after)
        print(f"Matching {len(before)} before samples with {len(after)} after samples...")
        pairs = match_samples(before, after)
        print("Computing statistics...")
        stats = compute_summary_stats(pairs)
        print(f"Generating HTML report -> {output}")
        generate_html_report(pairs, stats, args.before, args.after, output)
        print("Done!")
    else:
        parser.error("Provide either a single dataset path, or both --before and --after for diff mode")


if __name__ == "__main__":
    main()
