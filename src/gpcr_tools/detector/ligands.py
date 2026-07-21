"""Pre-annotation detector for incidental-candidate molecules.

A molecule on the curated ``INCIDENTAL_CANDIDATES`` set -- a biological lipid such
as cholesterol / palmitate, or a counter-ion metal (Ca / Na / Mg / Zn / Mn) -- can
be EITHER a functional ligand OR an incidental structural additive. When one is
present this surfaces it as advisory evidence so the model judges the role
itself -- and the exclude-list strip is bypassed so the model can see it.

This is metadata-only (no sequence/UniProt fetch), so it always runs.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    LIGAND_EXCLUDE_LIST,
    LOCUS_LIGANDS,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_INCIDENTAL_CANDIDATE,
    SIGNAL_TRANSDUCER_COPY,
    DetectSignal,
)
from gpcr_tools.validator.oligomer import (
    build_chain_identity_index,
    build_nonpolymer_instance_index,
    is_transducer_chain,
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


def detect_incidental_candidates(
    pdb_id: str,
    enriched_entry: dict[str, Any],
) -> list[DetectSignal]:
    """One advisory signal per incidental-candidate molecule (a biological lipid or a counter-ion metal) present.

    An incidental-candidate molecule can be EITHER a functional ligand OR an incidental
    structural lipid. The signal is advisory: it routes evidence into the prompt
    so the model judges the role itself (and any incidental-candidate member stripped by the
    exclude list is un-stripped so the model can see it) -- it does not silently
    send the case to human review.
    """
    present = sorted(set(_nonpolymer_comp_ids(enriched_entry)) & INCIDENTAL_CANDIDATES)
    return [
        DetectSignal(
            kind=SIGNAL_INCIDENTAL_CANDIDATE,
            target_ref=LOCUS_LIGANDS,
            summary=(
                f"{code} is present; it can be a functional ligand or an incidental "
                f"structural component. Judge its role from the paper and record a "
                f"pharmacological_role_check."
            ),
            payload={"comp_id": code},
            severity=SEVERITY_ADVISORY,
        )
        for code in present
    ]


def detect_transducer_copies(
    pdb_id: str,
    enriched_entry: dict[str, Any],
) -> list[DetectSignal]:
    """One advisory signal naming the non-polymer copies that sit on a G protein /
    transducer chain rather than on the receptor.

    Such a copy -- the transducer's own nucleotide / cofactor (a GTP / GDP or a
    structural Mg on the G-alpha), never a receptor ligand -- is surfaced so the
    model sets its role to Cofactor and it is catalogued as auxiliary. The chain is
    identified from the STRUCTURE (:func:`is_transducer_chain`), never the model's
    name, and copies are named per author chain + residue so the same component
    split across a transducer chain and the receptor is told apart. Metadata-only,
    so it always runs.
    """
    chains = build_chain_identity_index(enriched_entry)
    transducer_chains = {auth for auth, info in chains.items() if is_transducer_chain(info)}
    if not transducer_chains:
        return []
    copies: list[str] = []
    for comp_id, instances in build_nonpolymer_instance_index(enriched_entry).items():
        if comp_id in LIGAND_EXCLUDE_LIST:
            # Stripped before the model ever saw it (a buffer / mechanical ion / glycan
            # that happens to sit on the transducer chain): naming it would be advisory
            # noise the model cannot act on. Only model-visible copies are surfaced.
            continue
        for inst in instances or []:
            if not isinstance(inst, dict):
                continue
            auth = str(inst.get("auth_asym_id") or "").strip()
            seq = str(inst.get("auth_seq_id") or "").strip()
            if auth and auth in transducer_chains:
                copies.append(f"{comp_id} {auth}:{seq}")
    if not copies:
        return []
    copies.sort()
    return [
        DetectSignal(
            kind=SIGNAL_TRANSDUCER_COPY,
            target_ref=LOCUS_LIGANDS,
            summary=(
                f"Copies on a G protein / transducer chain (not the receptor): "
                f"{', '.join(copies)}. Judge each as the transducer's cofactor (role Cofactor)."
            ),
            payload={"copies": copies},
            severity=SEVERITY_ADVISORY,
        )
    ]
