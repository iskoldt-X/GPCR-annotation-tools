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

from gpcr_tools.config import (
    DISPUTED_MOLECULES,
    EXCLUDED_REAL_LIGAND_INTEREST,
    LIGAND_EXCLUDE_LIST,
    LOCUS_LIGANDS,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_DISPUTED_LIGAND,
    SIGNAL_EXCLUDED_REAL_LIGAND,
    DetectSignal,
)


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

    Disputed molecules are subtracted: the disputed fork un-strips them and
    guides the model directly (accommodate + guide), so they must NOT also fire a
    "stripped before the model sees it" review -- that claim would be false and
    the two pathways would contradict.
    """
    present = set(_nonpolymer_comp_ids(enriched_entry))
    hidden = sorted(
        present & EXCLUDED_REAL_LIGAND_INTEREST & LIGAND_EXCLUDE_LIST - DISPUTED_MOLECULES
    )
    return [
        DetectSignal(
            kind=SIGNAL_EXCLUDED_REAL_LIGAND,
            target_ref=LOCUS_LIGANDS,
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


def detect_disputed_ligands(
    pdb_id: str,
    enriched_entry: dict[str, Any],
) -> list[DetectSignal]:
    """One advisory signal per disputed molecule (cholesterol / palmitate) present.

    A disputed molecule can be EITHER a functional ligand OR an incidental
    structural lipid. The signal is advisory: it routes evidence into the prompt
    so the model judges the role itself (and any disputed member stripped by the
    exclude list is un-stripped so the model can see it) -- it does not silently
    send the case to human review.
    """
    present = sorted(set(_nonpolymer_comp_ids(enriched_entry)) & DISPUTED_MOLECULES)
    return [
        DetectSignal(
            kind=SIGNAL_DISPUTED_LIGAND,
            target_ref=LOCUS_LIGANDS,
            summary=(
                f"{code} is present and is a disputed molecule: it can be a "
                f"functional ligand or an incidental structural lipid. Judge its "
                f"role from the paper and record a disputed_assessment."
            ),
            payload={"comp_id": code},
            severity=SEVERITY_ADVISORY,
        )
        for code in present
    ]
