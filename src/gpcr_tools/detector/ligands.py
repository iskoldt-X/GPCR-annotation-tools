"""Pre-annotation detector for real ligands hidden by the buffer exclude list.

Some components on ``LIGAND_EXCLUDE_LIST`` are stripped from the metadata before
the model sees it (so common buffers/ions/sugars do not pollute the annotation).
A few of those can nonetheless be a genuine functional ligand (palmitate, for
example). When one is present, the model is blind to it, so this detector
surfaces it for human review.

This is metadata-only (no sequence/UniProt fetch), so it always runs.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.config import EXCLUDED_REAL_LIGAND_INTEREST, LIGAND_EXCLUDE_LIST
from gpcr_tools.detector.signals import (
    SEVERITY_REVIEW,
    SIGNAL_EXCLUDED_REAL_LIGAND,
    DetectSignal,
)

LIGANDS_LOCUS = "ligands"


def _nonpolymer_comp_ids(enriched_entry: dict[str, Any]) -> list[str]:
    """Collect the chem_comp ids of every non-polymer entity (None-safe)."""
    entities = enriched_entry.get("nonpolymer_entities")
    if not isinstance(entities, list):
        return []
    ids: list[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        comp_id = ((entity.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id")
        if comp_id:
            ids.append(comp_id)
    return ids


def detect_excluded_real_ligands(
    pdb_id: str,
    enriched_entry: dict[str, Any],
) -> list[DetectSignal]:
    """One review signal per high-interest ligand that is present but excluded.

    Only components that are BOTH high-interest AND actually on the exclude list
    are flagged — so a high-interest code that is not excluded (it already
    reaches the model) is never mis-reported here. One signal per component
    keeps each anchored to a single id and avoids plural-grammar pitfalls.
    """
    present = set(_nonpolymer_comp_ids(enriched_entry))
    hidden = sorted(present & EXCLUDED_REAL_LIGAND_INTEREST & LIGAND_EXCLUDE_LIST)
    return [
        DetectSignal(
            kind=SIGNAL_EXCLUDED_REAL_LIGAND,
            target_ref=LIGANDS_LOCUS,
            summary=(
                f"{code} is present in the structure but on the buffer exclude "
                f"list, so it is stripped before the model sees it; confirm "
                f"whether it is a functional ligand."
            ),
            payload={"comp_id": code},
            severity=SEVERITY_REVIEW,
        )
        for code in hidden
    ]
