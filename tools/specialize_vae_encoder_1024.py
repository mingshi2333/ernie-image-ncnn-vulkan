#!/usr/bin/env python3
"""Specialize only the two reviewed spatial reshapes for 1024x1024."""
import argparse
import json
from pathlib import Path
try:
    from specialize_vae_encoder import specialize_candidate, specialize_lines
except ImportError:
    from tools.specialize_vae_encoder import specialize_candidate, specialize_lines


def specialized_lines(lines):
    result, changes = specialize_lines(lines, 1024, 1024)
    return result, [(item['before'], item['after']) for item in changes]


def reviewed_dimensions(width, height):
    if (width, height) != (1024, 1024):
        raise ValueError("1024 encoder specialization is pinned to one development shape")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((args.reference / "fixture.json").read_text())
    reviewed_dimensions(fixture.get("width"), fixture.get("height"))
    print(json.dumps(specialize_candidate(args.template, args.reference, args.output, 1024, 1024)))


if __name__ == "__main__":
    main()
