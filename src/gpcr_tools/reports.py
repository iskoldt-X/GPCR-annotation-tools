"""Read-only operational reports over pipeline outputs.

Each function returns the report as a string so it is easy to test; the CLI
prints the returned text.  No mutation, no external calls.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from gpcr_tools.config import get_config

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def _validation_log_files() -> list[Path]:
    vdir = get_config().aggregated_dir / "validation_logs"
    return sorted(vdir.glob("*_validation.json")) if vdir.is_dir() else []


def report_pdf_coverage() -> str:
    """Summarise paper-PDF coverage from the download log: how many PDB entries
    landed in each outcome (downloaded, paywalled, no DOI, ...)."""
    cfg = get_config()
    log = _read_json(cfg.download_log_file) or {}
    entries = [e for e in log.values() if isinstance(e, dict)]
    if not entries:
        return "PDF coverage: no download log found (run 'fetch-papers' first)."

    counts = Counter(e.get("status", "unknown") for e in entries)
    total = sum(counts.values())
    lines = [f"PDF coverage report ({total} PDB entr{'y' if total == 1 else 'ies'}):", ""]
    for status, n in counts.most_common():
        pct = 100 * n / total
        lines.append(f"  {n:4d} ({pct:5.1f}%)  {status}")
    return "\n".join(lines)


def report_full_audit() -> str:
    """Summarise validation warnings and chimera conflicts across all aggregated
    PDBs."""
    files = _validation_log_files()
    if not files:
        return "Full audit: no validation logs found (run 'aggregate' first)."

    with_warnings: list[str] = []
    with_conflicts: list[str] = []
    chimera_status: Counter[str] = Counter()
    for f in files:
        pdb = f.name.removesuffix("_validation.json")
        data = _read_json(f) or {}
        chimera_status[data.get("chimera_status") or "unknown"] += 1
        if data.get("critical_warnings"):
            with_warnings.append(pdb)
        if data.get("algo_conflicts"):
            with_conflicts.append(pdb)

    lines = [f"Full validation audit ({len(files)} PDB(s)):", ""]
    lines.append(f"  PDBs with critical warnings: {len(with_warnings)}")
    if with_warnings:
        lines.append(f"    {', '.join(sorted(with_warnings))}")
    lines.append(f"  PDBs with algo conflicts:    {len(with_conflicts)}")
    if with_conflicts:
        lines.append(f"    {', '.join(sorted(with_conflicts))}")
    lines.append("  Chimera status:")
    for status, n in chimera_status.most_common():
        lines.append(f"    {n:4d}  {status}")
    return "\n".join(lines)


def report_tail_analysis() -> str:
    """Summarise the G-protein chimera ('4-residue tail') analysis: the score
    distribution, status breakdown, and which structures to review.

    (The historical report also catalogued tail sequences and candidate pools;
    those need per-run data not kept in the validation logs and are out of scope
    here.)"""
    files = _validation_log_files()
    if not files:
        return "Tail analysis: no validation logs found (run 'aggregate' first)."

    score_dist: Counter[Any] = Counter()
    status_dist: Counter[str] = Counter()
    flagged: list[tuple[str, Any]] = []
    for f in files:
        pdb = f.name.removesuffix("_validation.json")
        data = _read_json(f) or {}
        score = data.get("chimera_score")
        status = data.get("chimera_status") or "unknown"
        score_dist[score] += 1
        status_dist[status] += 1
        # A non-success status or an imperfect score is worth a curator's eye.
        if status != "success" or (isinstance(score, int) and score < 4):
            flagged.append((pdb, score))

    lines = [
        f"G-protein tail (chimera) analysis ({len(files)} PDB(s)):",
        "",
        "  Score distribution:",
    ]
    for score in sorted((s for s in score_dist if isinstance(s, int)), reverse=True):
        lines.append(f"    score {score}: {score_dist[score]}")
    if score_dist.get(None):
        lines.append(f"    score n/a: {score_dist[None]}")
    lines.append("  Status:")
    for status, n in status_dist.most_common():
        lines.append(f"    {n:4d}  {status}")
    lines.append(f"  Flagged for review (non-success or score < 4): {len(flagged)}")
    for pdb, score in sorted(flagged):
        lines.append(f"    {pdb}: score={score}")
    return "\n".join(lines)
