"""Pre-annotation ligand binding-site detector (site_ref), structure geometry.

Computes, for every ligand the model will annotate, which receptor site it binds
(``orthosteric`` / ``allosteric_7tm`` / ``extracellular_vestibule`` /
``intracellular`` / ``extracellular_domain`` / ``unknown``) and routes it into the
prompt as evidence. The value is objective and upstream: ligand contact residues
(gemmi) -> UniProt positions (RCSB alignment) -> GPCRdb generic numbers + segments
(shipped table) -> a class-aware signature rule. No GPCRdb runtime dependency.

``classify_site`` is the pure rule (signature + segment based, deliberately not
depth-from-number: TM2/TM4/TM6 are numbered in reverse). ``detect_site_refs`` is
the detector that wires geometry + alignment + table + rule into advisory signals.
It needs the coordinate file and the RCSB alignment, so it runs only when API
checks are enabled. Sparse or unmatched contacts yield no signal -- missing beats
confidently wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gpcr_tools.config import (
    DISPUTED_MOLECULES,
    EXTRACELLULAR_DOMAIN_SEGMENT,
    GPCR_CLASS_C,
    GPCR_CLASS_T2,
    GPCR_CLASSES_LARGE_ECD,
    INTRACELLULAR_SEGMENTS,
    LIGAND_EXCLUDE_LIST,
    LOCUS_LIGANDS,
    ORTHOSTERIC_CORE_GENERIC,
    ORTHOSTERIC_CORE_GENERIC_T2,
    RCSB_SUBJECT_OF_INVESTIGATION,
    SITE_REF_ALLOSTERIC_7TM,
    SITE_REF_EXTRACELLULAR_DOMAIN,
    SITE_REF_EXTRACELLULAR_VESTIBULE,
    SITE_REF_INTRACELLULAR,
    SITE_REF_MIN_MAPPED_CONTACTS,
    SITE_REF_ORTHOSTERIC,
    SITE_REF_UNKNOWN,
    VESTIBULE_SEGMENTS,
)
from gpcr_tools.detector.signals import SEVERITY_ADVISORY, SIGNAL_SITE_REF, DetectSignal
from gpcr_tools.validator.api_clients import fetch_polymer_alignment
from gpcr_tools.validator.generic_numbering import map_contacts, receptor_class
from gpcr_tools.validator.geometry import ligand_contact_residues, load_structure
from gpcr_tools.validator.oligomer import build_nonpolymer_instance_index, is_gpcr_slug

logger = logging.getLogger(__name__)


def classify_site(
    gpcr_class: str | None,
    generic_numbers: set[str],
    segments: set[str],
) -> str:
    """Return a ``site_ref`` value for a ligand from its contact signature."""
    if not generic_numbers and not segments:
        return SITE_REF_UNKNOWN

    in_tm = any(s.startswith("TM") for s in segments)
    on_ecd = EXTRACELLULAR_DOMAIN_SEGMENT in segments
    orthosteric_core = (
        ORTHOSTERIC_CORE_GENERIC_T2 if gpcr_class == GPCR_CLASS_T2 else ORTHOSTERIC_CORE_GENERIC
    )
    hits_core = bool(generic_numbers & orthosteric_core)

    # Class C: the orthosteric site is the extracellular Venus flytrap; the 7TM
    # bundle hosts only allosteric modulators. ECD is checked before the 7TM core
    # by design -- the VFT and the bundle are far apart, so a ligand with N-term
    # contacts is in the VFT, not a 7TM modulator that happens to graze a loop.
    if gpcr_class == GPCR_CLASS_C:
        if on_ecd:
            return SITE_REF_EXTRACELLULAR_DOMAIN
        if in_tm:
            return SITE_REF_ALLOSTERIC_7TM
        return SITE_REF_UNKNOWN

    # Note: generic numbers + the segments above cover the 7TM core and its loops
    # plus N-term; adhesion-receptor GAIN-subdomain segments (A.* / B.*) are not in
    # these sets, so a ligand confined to the GAIN domain falls through to unknown
    # -- a safe miss rather than a wrong label.

    # Large-ECD classes (B1 secretin, B2 adhesion, F Frizzled CRD): a ligand on
    # the extracellular domain without reaching the 7TM core binds the ECD.
    if gpcr_class in GPCR_CLASSES_LARGE_ECD and on_ecd and not hits_core:
        return SITE_REF_EXTRACELLULAR_DOMAIN

    if hits_core:
        return SITE_REF_ORTHOSTERIC
    if segments & INTRACELLULAR_SEGMENTS:
        return SITE_REF_INTRACELLULAR
    if segments & VESTIBULE_SEGMENTS:
        return SITE_REF_EXTRACELLULAR_VESTIBULE
    if in_tm:
        return SITE_REF_ALLOSTERIC_7TM
    return SITE_REF_UNKNOWN


def _gpcr_chain_accessions(enriched_entry: dict[str, Any]) -> dict[str, str]:
    """Map each GPCR receptor author chain to its UniProt accession (None-safe)."""
    chains: dict[str, str] = {}
    for entity in enriched_entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        accession: str | None = None
        for u in entity.get("uniprots") or []:
            if isinstance(u, dict) and is_gpcr_slug(u.get("gpcrdb_entry_name_slug") or ""):
                accession = u.get("rcsb_id")
                break
        if not accession:
            continue
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
    buffers (disputed molecules are un-stripped, so kept). Broad on purpose -- a
    site label helps every real ligand, not only the studied one."""
    present = set(build_nonpolymer_instance_index(enriched_entry).keys())
    return present - (LIGAND_EXCLUDE_LIST - DISPUTED_MOLECULES)


def _studied_ligands(enriched_entry: dict[str, Any]) -> set[str]:
    """Studied ligands (RCSB subject of investigation) plus disputed molecules.

    Only these earn a multi-site "emit one entry per site" nudge: an incidental
    additive scattered across grooves would otherwise be told to split into
    several entries, which would mislead the model.
    """
    studied: set[str] = set()
    for entity in enriched_entry.get("nonpolymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        comp_id = (entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get(
            "nonpolymer_comp_id"
        )
        if not comp_id:
            continue
        if comp_id in DISPUTED_MOLECULES:
            studied.add(comp_id)
            continue
        annotations = entity.get("rcsb_nonpolymer_entity_annotation")
        if isinstance(annotations, list) and any(
            isinstance(a, dict) and a.get("type") == RCSB_SUBJECT_OF_INVESTIGATION
            for a in annotations
        ):
            studied.add(comp_id)
    return studied


def _classify_copy(
    contacts: list[tuple[str, int, str]],
    chain_accessions: dict[str, str],
    alignment: dict[str, dict[str, list[tuple[int, int, int]]]],
) -> tuple[str, int]:
    """Classify one ligand copy from its receptor contacts. Returns (site, mapped)."""
    by_chain: dict[str, list[tuple[int, str]]] = {}
    for auth_chain, label_seq, amino_acid in contacts:
        by_chain.setdefault(auth_chain, []).append((label_seq, amino_acid))
    if not by_chain:
        return SITE_REF_UNKNOWN, 0
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
        return SITE_REF_UNKNOWN, mapped
    primary_chain = max(by_chain, key=lambda c: len(by_chain[c]))
    gpcr_class = receptor_class(chain_accessions.get(primary_chain) or "")
    return classify_site(gpcr_class, generic_numbers, segments), mapped


def _build_signal(comp_id: str, sites: list[str]) -> DetectSignal:
    if len(sites) == 1:
        summary = f"{comp_id} is computed to bind the {sites[0]} site."
    else:
        summary = (
            f"{comp_id} is modelled at {len(sites)} distinct sites "
            f"({', '.join(sites)}); record one ligand entry per site."
        )
    return DetectSignal(
        kind=SIGNAL_SITE_REF,
        target_ref=LOCUS_LIGANDS,
        summary=summary,
        payload={"comp_id": comp_id, "sites": sites},
        severity=SEVERITY_ADVISORY,
    )


def detect_site_refs(
    pdb_id: str,
    enriched_entry: dict[str, Any],
    cache_dir: Path,
) -> list[DetectSignal]:
    """One advisory site_ref signal per annotated ligand with a resolved site.

    A ligand at two distinct sites yields a multi-site signal (so the model emits
    one entry per site). Missing coordinates / alignment, or only sparse/unmatched
    contacts, yield no signal rather than a guess.
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

    studied = _studied_ligands(enriched_entry)
    receptor_chains = set(chain_accessions)
    signals: list[DetectSignal] = []
    for comp_id in sorted(comp_ids):
        sites: dict[str, int] = {}
        for contacts in ligand_contact_residues(structure, comp_id, receptor_chains):
            site, mapped = _classify_copy(contacts, chain_accessions, alignment)
            if site != SITE_REF_UNKNOWN and mapped > sites.get(site, 0):
                sites[site] = mapped
        if not sites:
            continue
        site_list = sorted(sites)
        # Only a studied/disputed ligand earns the multi-site "one entry per site"
        # nudge; an incidental additive reports just its dominant site.
        if len(site_list) > 1 and comp_id not in studied:
            site_list = [max(sites, key=lambda s: sites[s])]
        signals.append(_build_signal(comp_id, site_list))
    return signals
