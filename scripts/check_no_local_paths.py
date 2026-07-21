"""Reject developer-local filesystem paths from entering the repository.

Runs as a pre-commit hook over the staged files. A committed source tree must be
portable: absolute home-directory paths (``/Users/<name>/...`` on macOS,
``/home/<name>/...`` on Linux, ``C:\\Users\\<name>\\...`` on Windows) belong to
one machine and leak the author's identity. Configuration should be resolved at
runtime (env var, CLI flag, repo-relative default), never hard-coded.

The built-in patterns are generic on purpose: they match the *shape* of a home
path, not any specific username, so this script never needs to embed a private
identifier. To block extra machine-specific tokens (a username, a conda env
name) on your own checkout, drop them — one per line — into
``scripts/.local-denylist`` (git-ignored); they are matched as plain substrings.

Usage (also the pre-commit entry point):
    python scripts/check_no_local_paths.py FILE [FILE ...]
Exits non-zero and prints offending ``path:line`` locations if anything matches.
Bypass for a one-off false positive with ``git commit --no-verify``.
"""

import os
import re
import sys

# Generic home-directory shapes — no specific username baked in.
PATTERNS = [
    (re.compile(r"/Users/[^/\s'\"]+"), "macOS home path"),
    (re.compile(r"/home/[^/\s'\"]+"), "Linux home path"),
    (re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]"), "Windows home path"),
]

_HERE = os.path.dirname(os.path.abspath(__file__))
DENYLIST_FILE = os.path.join(_HERE, ".local-denylist")
# This script and the denylist describe the patterns, so they must not scan themselves.
SELF_SKIP = {os.path.abspath(__file__), os.path.abspath(DENYLIST_FILE)}


def load_local_denylist():
    """Optional per-checkout extra tokens (git-ignored); plain substring match."""
    terms = []
    if os.path.exists(DENYLIST_FILE):
        with open(DENYLIST_FILE, encoding="utf-8") as fh:
            for line in fh:
                term = line.strip()
                if term and not term.startswith("#"):
                    terms.append(term)
    return terms


def scan_file(path, extra_terms):
    violations = []
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except (UnicodeDecodeError, OSError):
        return violations  # binary or unreadable — nothing to check
    for lineno, line in enumerate(lines, 1):
        for rx, why in PATTERNS:
            m = rx.search(line)
            if m:
                violations.append((lineno, why, m.group(0)))
        for term in extra_terms:
            if term in line:
                violations.append((lineno, "local denylist", term))
    return violations


def main(argv):
    extra_terms = load_local_denylist()
    found = False
    for path in argv:
        if os.path.abspath(path) in SELF_SKIP:
            continue
        for lineno, why, snippet in scan_file(path, extra_terms):
            found = True
            print(f"{path}:{lineno}: developer-local reference ({why}): {snippet!r}")
    if found:
        print(
            "\nRefusing to commit developer-local paths. Resolve config at runtime "
            "(env var / flag / repo-relative default), or bypass with --no-verify."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
