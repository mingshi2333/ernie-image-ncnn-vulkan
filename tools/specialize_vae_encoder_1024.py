#!/usr/bin/env python3
"""Specialize only the two reviewed spatial reshapes for 1024x1024."""
try:
    from specialize_vae_encoder import specialize_lines
except ImportError:
    from tools.specialize_vae_encoder import specialize_lines


def specialized_lines(lines):
    result, changes = specialize_lines(lines, 1024, 1024)
    return result, [(item['before'], item['after']) for item in changes]


def reviewed_dimensions(width, height):
    if (width, height) != (1024, 1024):
        raise ValueError("1024 encoder specialization is pinned to one development shape")
