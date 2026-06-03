"""The pre-annotation detect stage.

Reads a PDB's enriched metadata, runs the detectors, and persists the resulting
signals to ``detect/{pdb}.json`` for the annotate and curate stages to consume.
No AI and no paper are used here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from gpcr_tools.config import get_config
from gpcr_tools.detector.geometry import detect_dual_role_ligands
from gpcr_tools.detector.gprotein import detect_g_protein_identity
from gpcr_tools.detector.ligands import detect_disputed_ligands, detect_excluded_real_ligands
from gpcr_tools.detector.signals import DetectSignal
from gpcr_tools.validator.cache import SequenceCache

logger = logging.getLogger(__name__)

_SEQUENCE_CACHE_NAME = "uniprot_sequence_cache.json"


def _enriched_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the ``data.entry`` envelope if present, else use the object."""
    return (raw.get("data") or {}).get("entry") or raw


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path* atomically (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_detect(
    pdb_id: str,
    *,
    skip_api_checks: bool = False,
    cache: SequenceCache | None = None,
) -> list[DetectSignal]:
    """Run the detectors on one PDB, persist ``detect/{pdb}.json``, return signals.

    Sequence-based detectors are skipped when *skip_api_checks* is set (they need
    UniProt reference sequences). The detect file is written regardless, so the
    stage's output is always present and inspectable.

    A *cache* may be supplied by a caller running many PDBs so fetched sequences
    persist across them; the caller then owns saving it. When none is given (a
    single-PDB run) one is created and saved here.
    """
    cfg = get_config()
    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"
    if not enriched_path.is_file():
        logger.warning("[detect] %s: no enriched data found; skipping.", pdb_id)
        return []

    try:
        raw = json.loads(enriched_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[detect] %s: could not read enriched data: %s", pdb_id, exc)
        return []

    entry = _enriched_entry(raw)
    signals: list[DetectSignal] = []

    # Metadata-only detectors (no external fetch) always run.
    signals.extend(detect_excluded_real_ligands(pdb_id, entry))
    signals.extend(detect_disputed_ligands(pdb_id, entry))

    # Detectors needing a network fetch (UniProt references, coordinate files).
    if not skip_api_checks:
        owns_cache = cache is None
        if cache is None:
            cache = SequenceCache(cfg.cache_dir / _SEQUENCE_CACHE_NAME)
        signals.extend(detect_g_protein_identity(pdb_id, entry, cache))
        signals.extend(detect_dual_role_ligands(pdb_id, entry, cfg.cache_dir))
        if owns_cache:
            cache.save()
    # Future detectors (entity reconciliation, ...) append here.

    out_path = cfg.detect_dir / f"{pdb_id}.json"
    _atomic_write_json(
        out_path,
        {"pdb_id": pdb_id, "signals": [s.to_dict() for s in signals]},
    )
    logger.info("[detect] %s: %d signal(s) -> %s", pdb_id, len(signals), out_path)
    return signals


def run_detect_stage(pdb_id: str | None = None, *, skip_api_checks: bool = False) -> dict[str, int]:
    """Run detect on one PDB or every enriched PDB. Returns {pdb_id: signal count}."""
    cfg = get_config()
    if pdb_id:
        targets = [pdb_id]
    elif cfg.enriched_dir.exists():
        targets = sorted(p.stem for p in cfg.enriched_dir.glob("*.json"))
    else:
        targets = []

    # One shared cache across all PDBs so UniProt references fetched for an early
    # PDB are reused (and persisted) rather than refetched for each one.
    cache = None if skip_api_checks else SequenceCache(cfg.cache_dir / _SEQUENCE_CACHE_NAME)
    summary = {
        pid: len(run_detect(pid, skip_api_checks=skip_api_checks, cache=cache)) for pid in targets
    }
    if cache is not None:
        cache.save()
    logger.info("[detect] processed %d PDB(s).", len(targets))
    return summary


def load_detect_signals(pdb_id: str) -> list[DetectSignal]:
    """Read back the persisted detect signals for *pdb_id* (empty if none)."""
    cfg = get_config()
    path = cfg.detect_dir / f"{pdb_id}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [DetectSignal.from_dict(s) for s in (data.get("signals") or [])]
