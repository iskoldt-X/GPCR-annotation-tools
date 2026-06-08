"""Pre-annotation ligand binding-site EVIDENCE (site_ref), from structure geometry.

For every ligand the model will annotate, this computes the objective geometric
FACTS about where each modelled copy sits -- contact residues (gemmi) -> UniProt
positions (RCSB alignment) -> GPCRdb generic numbers + segments (shipped table),
plus enclosure, lipid-vs-pocket facing, and membrane depth -- and routes them into
the prompt as evidence. It deliberately does NOT compute a site verdict: the model
infers ``site_ref`` from these facts plus the paper. The old deterministic
classifier was retired -- a confidently-wrong label dragged the model off the
right answer; a missing fact is better than a wrong conclusion.

Needs the coordinate file and the RCSB alignment, so it runs only when API checks
are enabled. Sparse or unmatched contacts yield no facts for that copy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    LIGAND_EXCLUDE_LIST,
    LOCUS_LIGANDS,
    ORTHOSTERIC_CORE_GENERIC,
    SITE_REF_MIN_MAPPED_CONTACTS,
)
from gpcr_tools.detector.signals import SEVERITY_ADVISORY, SIGNAL_SITE_REF, DetectSignal
from gpcr_tools.validator.api_clients import fetch_polymer_alignment
from gpcr_tools.validator.generic_numbering import load_numbering_table, map_contacts
from gpcr_tools.validator.geometry import ligand_contact_residues, load_structure
from gpcr_tools.validator.membrane import (
    ligand_facing_fractions,
    ligand_membrane_depth,
    membrane_frame,
)
from gpcr_tools.validator.oligomer import build_nonpolymer_instance_index, is_gpcr_slug

logger = logging.getLogger(__name__)


def _gpcr_chain_accessions(enriched_entry: dict[str, Any]) -> dict[str, str]:
    """Map each GPCR receptor author chain to its UniProt accession (None-safe).

    ``is_gpcr_slug`` is a permissive denylist, so a crystallization fusion partner
    (rubredoxin, flavodoxin, ...) on the same entity can also pass it. Among the
    candidates, prefer the accession that is actually in the numbering table (the
    real receptor) over a fusion partner that merely survives the denylist.
    """
    table = load_numbering_table()
    chains: dict[str, str] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        candidates: list[str] = [
            u["rcsb_id"]
            for u in entity.get("uniprots") or []
            if isinstance(u, dict)
            and is_gpcr_slug(u.get("gpcrdb_entry_name_slug") or "")
            and u.get("rcsb_id")
        ]
        if not candidates:
            continue
        accession = next((acc for acc in candidates if acc in table), candidates[0])
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth:
                chains[auth] = accession
    return chains


def _annotated_ligands(enriched_entry: dict[str, Any]) -> set[str]:
    """Component ids the model annotates: present non-polymers minus the stripped
    buffers (incidental-candidate molecules are un-stripped, so kept). Broad on purpose -- a
    site fact helps every real ligand, not only the studied one."""
    present = set(build_nonpolymer_instance_index(enriched_entry).keys())
    return present - (LIGAND_EXCLUDE_LIST - INCIDENTAL_CANDIDATES)


def _copy_evidence(
    contacts: list[tuple[str, int, str]],
    chain_accessions: dict[str, str],
    alignment: dict[str, dict[str, list[tuple[int, int, int]]]],
) -> dict[str, Any] | None:
    """Objective contact facts for one ligand copy, or ``None`` if too sparse.

    Maps the copy's receptor contacts to GPCRdb generic numbers + segments and
    counts how many fall in the canonical orthosteric core (a FACT, not a verdict).
    """
    by_chain: dict[str, list[tuple[int, str]]] = {}
    for auth_chain, label_seq, amino_acid in contacts:
        by_chain.setdefault(auth_chain, []).append((label_seq, amino_acid))
    if not by_chain:
        return None
    generic_numbers: set[str] = set()
    segments: set[str] = set()
    mapped = 0
    for auth_chain, chain_contacts in by_chain.items():
        accession = chain_accessions.get(auth_chain)
        regions = (alignment.get(auth_chain) or {}).get(accession) if accession else None
        if not accession or not regions:
            continue
        g, s, m, _ = map_contacts(chain_contacts, regions, accession)
        generic_numbers |= g
        segments |= s
        mapped += m
    if mapped < SITE_REF_MIN_MAPPED_CONTACTS:
        return None
    return {
        "generic_numbers": sorted(generic_numbers),
        "segments": sorted(segments),
        "core_hits": len(generic_numbers & ORTHOSTERIC_CORE_GENERIC),
        "mapped": mapped,
    }


def _build_signal(comp_id: str, copies: list[dict[str, Any]]) -> DetectSignal:
    summary = (
        f"{comp_id}: geometry facts for {len(copies)} modelled "
        f"cop{'y' if len(copies) == 1 else 'ies'} (infer the binding site)."
    )
    return DetectSignal(
        kind=SIGNAL_SITE_REF,
        target_ref=LOCUS_LIGANDS,
        summary=summary,
        payload={"comp_id": comp_id, "copies": copies},
        severity=SEVERITY_ADVISORY,
    )


def detect_site_refs(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache_dir: Path,
) -> list[DetectSignal]:
    """One advisory signal per annotated ligand carrying per-copy geometry facts.

    Each copy contributes its contact generic numbers + segments + canonical-core
    count + enclosure + lipid-vs-pocket facing + membrane depth. The model reads
    these facts (plus the paper) to assign ``site_ref`` itself. Missing coordinates
    / alignment, or only sparse/unmatched contacts, yield no facts rather than a
    guess. A ligand modelled at distinct sites simply shows distinct per-copy facts;
    the model decides whether to emit one entry per site.
    """
    chain_accessions = _gpcr_chain_accessions(enriched_entry)
    if not chain_accessions:
        return []
    comp_ids = _annotated_ligands(enriched_entry)
    if not comp_ids:
        return []
    structure = load_structure(pdb_id, cache_dir)
    if structure is None:
        return []
    alignment = fetch_polymer_alignment(pdb_id)
    if not alignment:
        return []

    receptor_chains = set(chain_accessions)
    frame = membrane_frame(structure)
    # model + per-copy atom lists are only needed for the membrane-depth fact, so
    # they are resolved only when a frame was fitted.
    model = structure[0] if frame is not None else None
    signals: list[DetectSignal] = []
    for comp_id in sorted(comp_ids):
        contact_copies = ligand_contact_residues(structure, comp_id, receptor_chains)
        if not contact_copies:
            continue
        # facing + per-copy atom lists are in model order, the same order as
        # ligand_contact_residues, so they align by index.
        facings = (
            ligand_facing_fractions(structure, comp_id, frame)
            if frame is not None
            else [None] * len(contact_copies)
        )
        atom_lists = (
            [list(res) for chain in model for res in chain if res.name == comp_id]
            if model is not None
            else []
        )
        copies: list[dict[str, Any]] = []
        for i, (burial, contacts) in enumerate(contact_copies):
            evidence = _copy_evidence(contacts, chain_accessions, alignment)
            if evidence is None:
                continue
            evidence["enclosure"] = round(burial, 2)
            evidence["facing"] = facings[i] if i < len(facings) else None
            if frame is not None and i < len(atom_lists):
                depth_band = ligand_membrane_depth(frame, atom_lists[i])
                if depth_band is not None:
                    evidence["depth"], evidence["in_band"] = depth_band
            copies.append(evidence)
        if copies:
            signals.append(_build_signal(comp_id, copies))
    return signals
