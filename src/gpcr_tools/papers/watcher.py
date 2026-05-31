"""Filesystem watcher for manually-downloaded paywalled papers.

After the auto-download phase, watches ``papers/`` for new PDFs dropped
by the user, validates them, auto-renames to ``{pdb_id}.pdf``, and
updates the download log.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    WATCHER_MAX_INGEST_ATTEMPTS,
    WATCHER_POLL_INTERVAL,
    WATCHER_STABILITY_CHECKS,
    WATCHER_STABILITY_INTERVAL,
    get_config,
)
from gpcr_tools.papers.downloader import _update_download_log

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


def _is_valid_pdf(path: Path) -> bool:
    """Check if a file starts with the ``%PDF`` magic bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header == _PDF_MAGIC
    except OSError:
        return False


def _get_pending_paywalled(download_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return entries from the log that are still paywalled."""
    return {
        pdb_id: entry
        for pdb_id, entry in download_log.items()
        if isinstance(entry, dict) and entry.get("status") == "fallback_paywalled"
    }


def _match_pdf_to_pdb(
    pdf_path: Path,
    pending: dict[str, dict[str, Any]],
) -> str | None:
    """Try to match a PDF filename to a pending paywalled PDB ID.

    Strategy:
      1. If the stem matches a pending PDB ID (e.g., ``7W55.pdf``), use it.
      2. Otherwise return None (user will need to rename).
    """
    stem = pdf_path.stem.upper()
    if stem in pending:
        return stem
    return None


def _wait_for_stability(path: Path) -> bool:
    """Wait until the file size stops changing (download complete)."""
    prev_size = -1
    stable_count = 0
    for _ in range(WATCHER_STABILITY_CHECKS + 5):  # max ~7 iterations
        try:
            current_size = path.stat().st_size
        except OSError:
            return False
        if current_size == prev_size and current_size > 0:
            stable_count += 1
            if stable_count >= WATCHER_STABILITY_CHECKS:
                return True
        else:
            stable_count = 0
        prev_size = current_size
        time.sleep(WATCHER_STABILITY_INTERVAL)
    return False


def _ingest_if_ready(
    pdf_path: Path,
    pdb_id: str,
    pending: dict[str, dict[str, Any]],
    papers_dir: Path,
) -> bool:
    """If *pdf_path* is a stable, valid PDF, rename it to ``{pdb_id}.pdf`` and
    log it as manually provided.

    Returns True on success; False if the file is not yet stable or not a valid
    PDF — in which case the caller must NOT permanently blacklist it (it may
    still be downloading, or the user may re-drop a good copy), only retry later.
    """
    if not _wait_for_stability(pdf_path):
        return False
    if not _is_valid_pdf(pdf_path):
        return False

    canonical = papers_dir / f"{pdb_id}.pdf"
    if pdf_path != canonical:
        os.replace(str(pdf_path), str(canonical))

    _update_download_log(
        pdb_id,
        {
            "status": "manual_user_provided",
            "source": "user_manual",
            "file_path": str(canonical),
            "doi": pending[pdb_id].get("doi"),
            "pmid": pending[pdb_id].get("pmid"),
            "pmcid": pending[pdb_id].get("pmcid"),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    return True


def _scan_and_ingest(pending: dict[str, dict[str, Any]], papers_dir: Path) -> int:
    """Match every PDF currently in *papers_dir* against *pending* and ingest
    the ready ones, removing them from *pending* in place. Returns the count.

    Run at startup (to pick up papers dropped before the watch began, or left
    from a previous session) and again on a Ctrl-C exit, so a manually-provided
    paper is never silently abandoned as ``skipped_no_paper``.
    """
    matched = 0
    for pdf_path in sorted(papers_dir.iterdir()):
        if pdf_path.suffix.lower() != ".pdf":
            continue
        pdb_id = _match_pdf_to_pdb(pdf_path, pending)
        if pdb_id is None:
            continue
        if _ingest_if_ready(pdf_path, pdb_id, pending, papers_dir):
            del pending[pdb_id]
            matched += 1
            print(
                f"  ✅ {pdb_id}.pdf — matched and saved ({len(pending)} remaining)",
                file=sys.stderr,
            )
    return matched


def run_watcher(paywalled_entries: dict[str, dict[str, Any]]) -> int:
    """Watch ``papers/`` for new PDFs and match to paywalled entries.

    Return the number of papers successfully matched.
    """
    cfg = get_config()
    papers_dir = cfg.papers_dir
    papers_dir.mkdir(parents=True, exist_ok=True)

    pending = dict(paywalled_entries)
    if not pending:
        return 0

    # Pick up papers already sitting in papers_dir before we start watching.
    matched = _scan_and_ingest(pending, papers_dir)

    if pending:
        _print_instructions(pending, papers_dir)
        # Re-examine matching files every poll instead of blacklisting them.
        # attempts/last_size give a still-downloading or just-re-dropped file
        # repeated chances; the count resets on any size change so only a
        # genuinely stuck stable file is eventually given up on.
        attempts: dict[str, int] = {}
        last_size: dict[str, int] = {}
        try:
            while pending:
                time.sleep(WATCHER_POLL_INTERVAL)
                for pdf_path in sorted(papers_dir.iterdir()):
                    if pdf_path.suffix.lower() != ".pdf":
                        continue
                    pdb_id = _match_pdf_to_pdb(pdf_path, pending)
                    if pdb_id is None:
                        continue
                    try:
                        size = pdf_path.stat().st_size
                    except OSError:
                        continue
                    if last_size.get(pdf_path.name) != size:
                        last_size[pdf_path.name] = size
                        attempts[pdf_path.name] = 0
                    if attempts[pdf_path.name] >= WATCHER_MAX_INGEST_ATTEMPTS:
                        continue
                    attempts[pdf_path.name] += 1
                    if _ingest_if_ready(pdf_path, pdb_id, pending, papers_dir):
                        del pending[pdb_id]
                        matched += 1
                        print(
                            f"  ✅ {pdb_id}.pdf — matched and saved ({len(pending)} remaining)",
                            file=sys.stderr,
                        )

        except KeyboardInterrupt:
            # Before giving up, ingest anything that became ready at the last
            # moment, then record only the truly-absent ones as skipped.
            matched += _scan_and_ingest(pending, papers_dir)
            for pdb_id, entry in pending.items():
                _update_download_log(
                    pdb_id,
                    {
                        "status": "skipped_no_paper",
                        "source": None,
                        "file_path": None,
                        "doi": entry.get("doi"),
                        "pmid": entry.get("pmid"),
                        "pmcid": entry.get("pmcid"),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

    # Phase 3: Summary
    total = len(paywalled_entries)
    skipped = len(pending)
    skipped_ids = ", ".join(sorted(pending.keys()))
    if skipped:
        print(
            f"\nDone. {matched}/{total} paywalled papers provided. "
            f"{skipped} skipped ({skipped_ids}).",
            file=sys.stderr,
        )
    else:
        print(
            f"\nDone. {matched}/{total} paywalled papers provided. All resolved.",
            file=sys.stderr,
        )

    return matched


def _print_instructions(pending: dict[str, dict[str, Any]], papers_dir: Path) -> None:
    """Print the paywalled paper instructions box."""
    count = len(pending)
    print(
        "\n╭─ Papers Needing Manual Download ─────────────────────────────╮",
        file=sys.stderr,
    )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        f"│  {count} paper(s) could not be auto-downloaded (paywalled).   │",
        file=sys.stderr,
    )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        "│  Download them in your browser and save to:                  │",
        file=sys.stderr,
    )
    print(
        f"│    📂  {papers_dir!s:<50s}│",
        file=sys.stderr,
    )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        f"│  {'PDB':<6s} {'DOI':<42s} {'PMID':<10s}│",
        file=sys.stderr,
    )
    for pdb_id, entry in sorted(pending.items()):
        doi = entry.get("doi") or "(none)"
        if len(doi) > 40:
            doi = doi[:37] + "..."
        pmid = str(entry.get("pmid") or "(none)")
        print(
            f"│  {pdb_id:<6s} {doi:<42s} {pmid:<10s}│",
            file=sys.stderr,
        )
    print(
        "│                                                              │",
        file=sys.stderr,
    )
    print(
        "│  ⏳ Watching for new PDFs... (Ctrl+C to stop)                │",
        file=sys.stderr,
    )
    print(
        "╰──────────────────────────────────────────────────────────────╯\n",
        file=sys.stderr,
    )
