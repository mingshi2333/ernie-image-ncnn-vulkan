#!/usr/bin/env python3
"""Fail CI when CTest logged validation errors, including in skipped tests."""
import argparse
from pathlib import Path
import re


def check(path):
    # LastTest.log retains both output streams even for SKIP_RETURN_CODE=77.
    # CTest's skip code otherwise takes precedence over FAIL_REGULAR_EXPRESSION.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    failures = [(number, line) for number, line in enumerate(lines, 1)
                if re.search(r"[Vv]alidation [Ee]rror|VUID-", line)]
    for number, line in failures:
        print(f"{path}:{number}: {line}")
    if failures:
        print(f"Vulkan validation failed: {len(failures)} matching log lines")
        return 1
    print("No Vulkan validation errors in CTest output")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    raise SystemExit(check(args.log))
