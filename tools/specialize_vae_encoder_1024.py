#!/usr/bin/env python3
"""Specialize only the two reviewed spatial reshapes for 1024x1024."""
from pathlib import Path


def specialized_lines(lines):
    expected = {"reshape_77": ["0=16", "1=512"], "reshape_78": ["0=4", "1=4", "2=512"]}
    replacements = {"reshape_77": ["0=16384", "1=512"],
                    "reshape_78": ["0=128", "1=128", "2=512"]}
    result, changes = list(lines), []
    for index, line in enumerate(result):
        parts = line.split()
        if parts and parts[0] == "Reshape":
            if parts[1] not in expected or parts[6:] != expected[parts[1]]:
                raise ValueError("Unreviewed encoder reshape")
            changed = " ".join(parts[:6] + replacements[parts[1]])
            changes.append((line, changed)); result[index] = changed
    if len(changes) != 2:
        raise ValueError("Expected exactly two encoder reshapes")
    return result, changes


def reviewed_dimensions(width, height):
    if (width, height) != (1024, 1024):
        raise ValueError("1024 encoder specialization is pinned to one development shape")
