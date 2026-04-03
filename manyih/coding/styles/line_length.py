"""
Line length style definitions.

Styles: line_79, line_120, line_unlimited
"""

from typing import Tuple

from .registry import StyleDefinition, REGISTRY


def validate_line_79(code: str) -> Tuple[bool, str]:
    """Validate max 79 character lines."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if len(line) > 79:
            return False, f"Line {i+1}: Length {len(line)} exceeds 79"
    return True, "All lines <= 79 chars"


def validate_line_120(code: str) -> Tuple[bool, str]:
    """Validate max 120 character lines."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if len(line) > 120:
            return False, f"Line {i+1}: Length {len(line)} exceeds 120"
    return True, "All lines <= 120 chars"


def validate_line_unlimited(code: str) -> Tuple[bool, str]:
    """No line length limit - always passes."""
    return True, "No line length limit"


# Register styles
REGISTRY.register(StyleDefinition(
    style_id="line_79",
    style_key="line_length",
    description="Limit all lines to a maximum of 79 characters.",
    validate=validate_line_79,
    category="formatting",
    pylint_code="C0301",
    check_description="No line exceeds 79 characters",
    example_pass="x = 1  # short line",
    example_fail="x = 'a' * 80  # " + "x" * 60,
))

REGISTRY.register(StyleDefinition(
    style_id="line_120",
    style_key="line_length",
    description="Limit all lines to a maximum of 120 characters.",
    validate=validate_line_120,
    category="formatting",
    pylint_code="C0301",
    check_description="No line exceeds 120 characters",
    example_pass="x = 'a' * 100  # under 120",
    example_fail="x = 'a' * 130  # over 120",
))

REGISTRY.register(StyleDefinition(
    style_id="line_unlimited",
    style_key="line_length",
    description="Do not limit line length; lines can be any length.",
    validate=validate_line_unlimited,
    category="formatting",
    pylint_code="C0301",
    check_description="No line length restriction",
    example_pass="Any line length is acceptable",
    example_fail="N/A - always passes",
))
