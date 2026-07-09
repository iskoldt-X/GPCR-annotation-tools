"""Orchestration layer — wire all aggregation + validation components.

``aggregate_pdb()`` runs the full pipeline for a single PDB ID:
    AI runs → voting → best run → deepcopy → ground truth → validators →
    discrepancies → integrity → chimera → validation report → atomic writes.

``aggregate_all()`` iterates pending PDBs with per-PDB error isolation.

Conventions:
    - Atomic writes: all output files are written to temp files first, then
      ``os.replace``-d together after every write succeeds (``try...finally`` cleanup).
    - Truthiness: ``if enriched is None:`` — NOT ``if not enriched:``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpcr_tools.aggregator.ai_results_loader import (
    get_pending_pdb_ids,
    load_ai_runs,
    pdb_has_runs,
)
from gpcr_tools.aggregator.enriched_loader import enriched_is_incomplete, load_enriched_data
from gpcr_tools.aggregator.ground_truth import inject_ground_truth
from gpcr_tools.aggregator.voting import (
    extract_ai_g_protein,
    find_discrepancies,
    flag_low_confidence_consensus,
    get_majority_votes,
    select_best_run,
)
from gpcr_tools.config import (
    A5_SUBTYPE_FAMILY,
    AGG_STATUS_COMPLETED,
    AGG_STATUS_FAILED,
    ALERT_MULTI_COPY_LIGAND,
    ALERT_PREFIX_ALGO_WARNING,
    ALERT_PREFIX_ALPHA5_GRAFT,
    ALERT_PREFIX_API_UNAVAILABLE,
    ALERT_PREFIX_CHIMERIC_REVIEW,
    ALERT_PREFIX_HALLUCINATION,
    ALERT_PREFIX_TIE_BREAKER_ALIGNED,
    ALERT_PREFIX_TIE_BREAKER_OVERRIDE,
    ALERT_PREFIX_UNRECOGNISED_G_ALPHA,
    CHIMERA_BACKBONE_UNKNOWN,
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_SKIPPED,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_SUBTYPE_LOW_CONFIDENCE,
    EMPTY_VALUES,
    FULL_G_ALPHA_CANDIDATES,
    LOW_CONFIDENCE_LEVELS,
    POLYMER_FEATURES_CACHE_NAME,
    SITE_REF_UNKNOWN,
    SUBTYPE_BASIS_CONSTRUCT_NAME,
    SUBTYPE_BASIS_FAMILY_VERIFIED,
    SUBTYPE_BASIS_RESOLVED,
    VALIDATION_EXCLUDED_BUFFER,
    get_config,
    is_empty_key,
    list_item_identity,
)
from gpcr_tools.detector.signals import (
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_COUPLING_PROTOMER,
    to_critical_warnings,
)
from gpcr_tools.detector.stage import load_detect_signals
from gpcr_tools.fetcher.cache import JsonCache
from gpcr_tools.validator.api_clients import SynonymCache
from gpcr_tools.validator.cache import (
    PolymerFeaturesCache,
    SequenceCache,
    ValidationCache,
)
from gpcr_tools.validator.chimera import get_chimera_analysis
from gpcr_tools.validator.consistency import state_ligand_consistency_warnings
from gpcr_tools.validator.integrity_checker import validate_all
from gpcr_tools.validator.ligand_validator import validate_and_enrich_ligands
from gpcr_tools.validator.oligomer import (
    analyze_oligomer,
    correct_binder_names,
    detect_crystallization_fusions,
    reconcile_missed_polymers,
    relocate_misfiled_g_protein_fragments,
)
from gpcr_tools.validator.receptor_validator import validate_receptor_identity

logger = logging.getLogger(__name__)

# Detect REVIEW signals of these kinds are NOT re-surfaced as critical warnings
# here: the aggregator re-derives the G protein review from its own alpha5
# analysis below, with finer severity tuning (low-confidence -> note, not a
# blocker). Routing the detect copy too would both duplicate the warning and
# override that tuning. (The deferred chimera-logic consolidation will collapse
# the two into a single source.)
_AGGREGATOR_OWNED_REVIEW_KINDS = frozenset({SIGNAL_CHIMERIC_GPROTEIN})


def _coupling_protomer(pdb_id: str) -> str | None:
    """The geometric G protein-coupling protomer chain from the detect sidecar, if any.

    Returns ``None`` when the detect stage did not run, found no G protein, or could
    not resolve a single protomer -- in which case primary selection falls back to the
    lower ranks.
    """
    for signal in load_detect_signals(pdb_id):
        if signal.kind == SIGNAL_COUPLING_PROTOMER:
            chain = signal.payload.get("coupling_chain")
            return chain if isinstance(chain, str) else None
    return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AggregateResult:
    """Container for a single PDB aggregation result."""

    pdb_id: str
    success: bool
    aggregated_path: Path | None = None
    voting_log_path: Path | None = None
    validation_path: Path | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation report assembly
# ---------------------------------------------------------------------------

# The recognised G-alpha subunit slugs (the values of the curated human G-alpha
# roster). A specific alpha-subunit slug outside this set is not a known G-alpha
# candidate and must reach a human. Built once at import.
_RECOGNISED_G_ALPHA_SLUGS = frozenset(FULL_G_ALPHA_CANDIDATES.values())


def _warn_on_unrecognised_g_alpha(best_run_data: dict[str, Any]) -> list[str]:
    """Flag (for the curator) a G protein alpha subunit named with a specific slug
    that is NOT in the curated G-alpha candidate set.

    The candidate roster is alpha-specific, so this checks the alpha subunit only;
    beta/gamma carry their own slugs and are out of scope. An honest abstention
    (missing / empty / 'unknown' name) is never flagged -- only a specific,
    off-roster slug, which is the signature of an invented subtype/species. The
    warning is a critical warning so it disables one-click accept-all for the PDB.
    """
    slug = extract_ai_g_protein(best_run_data)
    if not isinstance(slug, str):
        return []
    normalised = slug.strip().lower()
    if normalised in EMPTY_VALUES or normalised == "unknown":
        return []
    if normalised in _RECOGNISED_G_ALPHA_SLUGS:
        return []
    return [
        f"{ALERT_PREFIX_UNRECOGNISED_G_ALPHA} at "
        f"'signaling_partners.g_protein.alpha_subunit': G protein alpha subunit "
        f"'{slug}' is not a recognised G-alpha candidate (off the curated human "
        f"G-alpha set); verify the subtype/species against the paper."
    ]


def _prune_excluded_buffer_ligands(best_run_data: dict[str, Any]) -> None:
    """Drop excluded-buffer ligands from the aggregated record, in place.

    A ligand the validator tagged ``EXCLUDED_BUFFER`` (a crystallization
    detergent / cryo-additive / matrix lipid such as BOG or NAG) is not a
    functional GPCR ligand and must not reach the final aggregated record,
    the curator, or the CSV export. It is removed here, at the aggregation
    layer, so every downstream consumer sees one consistent ligand list.

    A single, narrow rescue keeps a genuinely-functional incidental molecule:
    a dual-use lipid (e.g. palmitate) the model explicitly judged a real ligand
    carries ``pharmacological_role_check.is_functional_ligand == True`` and is
    kept. The rescue uses an ``is True`` identity test on purpose -- a null /
    missing / ``False`` verdict means "not assessed" or "not functional" and
    does NOT rescue.

    The predicate is the validation status ALONE. A molecule that actually
    matched a real component is tagged ``MATCHED_SMALL_MOLECULE`` (not
    ``EXCLUDED_BUFFER``), so a matched lipid is never dropped here -- testing
    component-id membership instead would wrongly drop it.

    When a dropped component had a ``MULTI_COPY_LIGAND`` oligomer alert, that
    alert is pruned too, so the record stays self-consistent (no alert points
    at a ligand that no longer exists).
    """
    ligands = best_run_data.get("ligands")
    if not isinstance(ligands, list):
        return

    kept: list[Any] = []
    dropped_comp_ids: set[str] = set()
    for lig in ligands:
        if isinstance(lig, dict) and _is_excluded_buffer_drop(lig):
            comp_id = lig.get("chem_comp_id")
            if isinstance(comp_id, str) and comp_id.strip():
                dropped_comp_ids.add(comp_id.strip())
            continue
        kept.append(lig)
    best_run_data["ligands"] = kept

    if not dropped_comp_ids:
        return

    # Keep the oligomer record self-consistent: a MULTI_COPY_LIGAND alert names
    # its component in the path 'ligands[{comp_id}]'; once that component is
    # dropped the alert dangles, so remove it. Other alert types are untouched.
    oligomer = best_run_data.get("oligomer_analysis")
    if not isinstance(oligomer, dict):
        return
    alerts = oligomer.get("alerts")
    if not isinstance(alerts, list):
        return
    dropped_paths = {f"ligands[{comp_id}]" for comp_id in dropped_comp_ids}
    oligomer["alerts"] = [
        alert
        for alert in alerts
        if not (
            isinstance(alert, dict)
            and alert.get("type") == ALERT_MULTI_COPY_LIGAND
            and any(path in str(alert.get("message", "")) for path in dropped_paths)
        )
    ]


def _is_excluded_buffer_drop(lig: dict[str, Any]) -> bool:
    """True if *lig* is an excluded buffer that is NOT rescued as functional.

    Drop condition: ``validation_status == EXCLUDED_BUFFER`` AND the model did
    not explicitly judge it a functional ligand (``is True`` identity rescue).
    """
    if lig.get("validation_status") != VALIDATION_EXCLUDED_BUFFER:
        return False
    prc = lig.get("pharmacological_role_check")
    rescued = isinstance(prc, dict) and prc.get("is_functional_ligand") is True
    return not rescued


def _copy_token(value: Any) -> str:
    """Normalise a copy-identifier fragment to the spelling the CSV writer uses.

    A copy identifier is ``"<auth_asym_id>:<auth_seq_id>"`` built with ``str().strip()``
    on each side, so the reverse index below keys on exactly the tokens the per-copy
    ``copy_id`` votes and the CSV residue tokens carry.
    """
    return "" if value is None else str(value).strip()


def _copy_chain(copy_id: Any) -> str:
    """Author chain id from a per-copy identifier ``"<auth_asym_id>:<auth_seq_id>"``."""
    token = _copy_token(copy_id)
    return token.split(":", 1)[0] if ":" in token else token


def _stamp_is_functional(row: dict[str, Any], value: Any) -> None:
    """Set ``pharmacological_role_check.is_functional_ligand`` on *row* in place."""
    prc = row.get("pharmacological_role_check")
    if not isinstance(prc, dict):
        prc = {}
        row["pharmacological_role_check"] = prc
    prc["is_functional_ligand"] = value


# Per-site decision + narrative fields on a ligand row. Everything else a ligand
# row carries (name / SMILES / InChIKey / pubchem / type / is_endogenous /
# validation_status / synonyms) is component-intrinsic chemistry that travels with
# the compound regardless of which site the copy sits at. These three keys are the
# ones that describe a SPECIFIC binding site: the voted decision (``role.value`` and
# ``pharmacological_role_check.is_functional_ligand``) and the free-text prose that
# explains that decision (``role.evidence`` / ``pharmacological_role_check.evidence``
# / ``site_ref_justification`` -- all in ``config.SOFT_FIELD_KEYS`` as explanatory,
# never-voted writing). When a row is rebuilt by borrowing another site's chemistry
# template, these must NOT ride along from that sibling site.
_PER_SITE_DECISION_KEYS: tuple[str, ...] = (
    "role",
    "pharmacological_role_check",
    "site_ref_justification",
)


def _apply_per_site_from_majority(row: dict[str, Any], mv_entry: Any) -> None:
    """Replace *row*'s per-site decision + narrative fields with the majority vote.

    A rebuilt/revived row borrows its chemistry from a surviving row of the SAME
    compound at ANOTHER site (the only place the enriched chemistry lives), so that
    template's ``role`` / ``pharmacological_role_check`` / ``site_ref_justification``
    describe the wrong site. Overwrite them with the majority-voted values for THIS
    ``component:site`` identity, which already carry the voted decision (role.value,
    is_functional_ligand) with the un-votable soft fields (evidence / confidence /
    justification) nulled out by the voting stage. When no majority entry exists for
    this identity, clear them to ``None`` -- an honest "not assessed" rather than the
    sibling site's prose. The caller stamps the final ``is_functional_ligand`` after
    this (True to revive, False for a dropped follow-out marker).
    """
    for key in _PER_SITE_DECISION_KEYS:
        row[key] = copy.deepcopy(mv_entry.get(key)) if isinstance(mv_entry, dict) else None


def _rebuild_small_molecule_rows_from_per_copy(
    best_run_data: dict[str, Any],
    majority_votes: Any,
) -> None:
    """Re-derive small-molecule ligand rows from the aggregated per-copy site votes.

    The shipped ligand list is otherwise taken verbatim from the single selected
    best run, whose per-copy site attribution can be an outlier: a physical copy
    then lands on the wrong binding-site row -- one row inflated with copies that
    belong elsewhere, or a whole row dropped so its copies have nowhere to go. This
    step re-anchors each keyed small molecule on its post-prune rows and uses the
    aggregated (majority-voted) per-copy attribution only to move copies onto the
    site the runs agreed on, revive a row an outlier best run wrongly dropped, and
    let a copy follow a dropped row out.

    Runs as the final aggregation step, AFTER discrepancy detection (so the
    best-run-vs-majority review gate is already computed against the original best
    run and preserved) and AFTER the excluded-buffer prune (so the post-prune list
    is the chemistry-and-existence source). Only the shipped / CSV ligand list is
    rebuilt; nothing the discrepancy pass compared is rewritten.

    The rule, per ``(component, binding-site)`` of a keyed small molecule that has
    at least one mapped per-copy group and a surviving post-prune chemistry
    template (everything else -- keyless peptide/glycan/apo rows, and components the
    buffer prune removed entirely -- passes through untouched):

    * ``is_functional_ligand`` follows the in-memory majority vote for that exact
      ``component:site`` identity (never the on-disk voting log, which records only
      disagreements, nor the best run's own outlier verdict): a majority **True**
      keeps/revives the row, **False** makes the CSV writer drop it and follow its
      copies out, and a majority **null** ("not assessed" / no such vote) does NOT
      overwrite -- the row keeps whatever verdict the post-prune row itself carried.
    * The row set is anchored on the post-prune rows. A post-prune row is kept
      (revived if the majority says True over an outlier drop). A per-copy group at
      a site with **no** post-prune row of that component is only turned into a
      shipped row when the majority explicitly says True; a group whose majority is
      null/False builds only a dropped follow-out marker (never a shipped row), so
      structural-lipid copies leave with it instead of flooding a surviving sibling.
      A post-prune row of the component whose site no per-copy group covers is kept
      as-is unless the majority says False -- so a real functional row the per-copy
      votes simply did not reach is never silently deleted.
    * ``chain_id`` is re-derived from the author chains of the copies actually
      grouped onto that site (empty when none), so it reflects the physical copies
      rather than an inherited template chain set.

    Finally the record's ``ligand_copies`` is set to the majority-voted list, so the
    CSV writer re-partitions each row's copies from the same voted attribution.
    """
    ligands = best_run_data.get("ligands")
    if not isinstance(ligands, list):
        return
    if not isinstance(majority_votes, dict):
        return
    voted_copies = majority_votes.get("ligand_copies")
    if not isinstance(voted_copies, list) or not voted_copies:
        return  # no per-copy attribution to rebuild from -> ship best run as-is

    instance_index = (best_run_data.get("oligomer_analysis") or {}).get("nonpolymer_instance_index")
    if not isinstance(instance_index, dict):
        return

    # Reverse index: copy identifier -> component id, from the structure's roster.
    token_to_comp: dict[str, str] = {}
    for comp_id, instances in instance_index.items():
        if not isinstance(comp_id, str) or not isinstance(instances, list):
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            token = (
                f"{_copy_token(inst.get('auth_asym_id'))}:{_copy_token(inst.get('auth_seq_id'))}"
            )
            token_to_comp[token] = comp_id

    # Post-prune chemistry: a representative row per component (chemistry is
    # component-intrinsic, used only when a per-copy group has no exact per-site
    # row), the exact per-site row keyed by its component:site identity, and the
    # component's own set of post-prune site identities (its anchor rows). Keyed
    # small molecules only; keyless rows (peptide / glycan / apo) are left out so
    # they pass through.
    template_by_comp: dict[str, dict[str, Any]] = {}
    row_by_ident: dict[str, dict[str, Any]] = {}
    idents_by_comp: dict[str, set[str]] = {}
    for lig in ligands:
        if not isinstance(lig, dict):
            continue
        comp_id = _copy_token(lig.get("chem_comp_id"))
        if is_empty_key(comp_id):
            continue
        template_by_comp.setdefault(comp_id, lig)
        ident = list_item_identity(lig, "chem_comp_id", 0)
        row_by_ident.setdefault(ident, lig)
        idents_by_comp.setdefault(comp_id, set()).add(ident)

    # Per component:site ligand identity, from the in-memory vote result: the
    # majority is_functional (None "not assessed" / no such group keeps; only False
    # drops) and the full voted entry, so a rebuilt row's per-site decision +
    # narrative fields come from the majority vote for its OWN site rather than the
    # sibling-site chemistry template it borrows.
    isfunc_by_ident: dict[str, Any] = {}
    mv_ligand_by_ident: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(majority_votes.get("ligands") or []):
        if not isinstance(entry, dict):
            continue
        ident = list_item_identity(entry, "chem_comp_id", idx)
        prc = entry.get("pharmacological_role_check")
        isfunc_by_ident[ident] = prc.get("is_functional_ligand") if isinstance(prc, dict) else None
        mv_ligand_by_ident[ident] = entry

    # Group each physical copy by (component, majority-voted site). 'unknown'/absent
    # sites are not grouped into a row (they surface via the CSV homeless bucket),
    # but their component is still recorded so an all-unknown component falls back
    # to pass-through rather than losing its best-run row.
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for copy_row in voted_copies:
        if not isinstance(copy_row, dict):
            continue
        comp_id = token_to_comp.get(_copy_token(copy_row.get("copy_id")))
        if comp_id is None:
            continue  # unmappable copy: leave the component's rows to pass-through
        # Group by the site verbatim (the schema pins it to the shared, all-lowercase
        # site_ref enum), so the component:site identity built from this site below
        # matches the one list_item_identity derives from the ligands / majority-vote
        # entries -- both use the raw enum value, with a case-insensitive unknown test.
        site = _copy_token(copy_row.get("site_ref"))
        if not site or site.lower() == SITE_REF_UNKNOWN:
            continue
        groups.setdefault(comp_id, {}).setdefault(site, []).append(copy_row)

    # A component is rebuilt only when it has a real-site per-copy group AND a
    # surviving post-prune chemistry template. Everything else passes through.
    rebuilt_comps = {c for c in groups if c in template_by_comp}
    if not rebuilt_comps:
        # deepcopy so the aggregated record owns its per-copy list rather than
        # aliasing the in-memory vote structure (which best_run_data is otherwise
        # fully independent of, being a deepcopy of the selected run).
        best_run_data["ligand_copies"] = copy.deepcopy(voted_copies)
        return

    def _chains(copies: list[dict[str, Any]]) -> str:
        return ", ".join(sorted({c for c in (_copy_chain(r.get("copy_id")) for r in copies) if c}))

    new_ligands: list[Any] = []
    emitted: set[str] = set()
    for lig in ligands:
        comp_id = _copy_token(lig.get("chem_comp_id")) if isinstance(lig, dict) else ""
        if comp_id not in rebuilt_comps:
            new_ligands.append(lig)  # keyless / no per-copy / all-unknown: unchanged
            continue
        if comp_id in emitted:
            continue  # this component's rows are all emitted at its first occurrence
        emitted.add(comp_id)

        voted_sites = groups[comp_id]
        # Site identity <-> the component:site path, for both the component's own
        # post-prune rows (its anchors) and its per-copy voted sites.
        voted_ident_to_site = {
            list_item_identity({"chem_comp_id": comp_id, "site_ref": site}, "chem_comp_id", 0): site
            for site in voted_sites
        }
        all_idents = sorted(idents_by_comp.get(comp_id, set()) | set(voted_ident_to_site))
        for ident in all_idents:
            base = row_by_ident.get(ident)
            majority = isfunc_by_ident.get(ident)
            voted_site = voted_ident_to_site.get(ident)
            copies = voted_sites.get(voted_site, []) if voted_site is not None else []
            if base is not None:
                # Anchor row: keep it, re-stamping only a definitive majority verdict
                # (True revives an outlier drop; False drops it and follows its copies
                # out; null leaves the post-prune verdict untouched). An anchor row no
                # per-copy group reached (copies == []) is still kept -- a real
                # functional row the votes did not cover is never silently deleted.
                row = copy.deepcopy(base)
                if majority is True or majority is False:
                    _stamp_is_functional(row, majority)
                row["chain_id"] = _chains(copies)
                new_ligands.append(row)
            elif majority is True:
                # No post-prune row here, but the majority explicitly calls it a
                # functional ligand -> revive it, borrowing only the component's
                # CHEMISTRY from another site's template. The per-site decision +
                # narrative (role.value / evidence / justification) come from THIS
                # site's majority vote, never the borrowed sibling-site template.
                row = copy.deepcopy(template_by_comp[comp_id])
                row["chem_comp_id"] = comp_id
                row["site_ref"] = voted_site
                _apply_per_site_from_majority(row, mv_ligand_by_ident.get(ident))
                _stamp_is_functional(row, True)
                row["chain_id"] = _chains(copies)
                new_ligands.append(row)
            elif copies:
                # No post-prune row and no majority-functional call, but copies vote
                # here (a structural lipid the prune removed, or a null-verdict group):
                # emit a dropped follow-out marker -- never shipped -- so those copies
                # leave with it rather than flooding a surviving sibling as homeless.
                # Same rule: only chemistry is borrowed; the marker's own per-site
                # decision + narrative come from the majority vote (empty when none),
                # so the curator panel never shows another site's prose on it.
                row = copy.deepcopy(template_by_comp[comp_id])
                row["chem_comp_id"] = comp_id
                row["site_ref"] = voted_site
                _apply_per_site_from_majority(row, mv_ligand_by_ident.get(ident))
                _stamp_is_functional(row, False)
                row["chain_id"] = _chains(copies)
                new_ligands.append(row)
            # else: no post-prune row, no majority-True, no copies -> nothing to build.

    best_run_data["ligands"] = new_ligands
    # deepcopy so the aggregated record owns its per-copy list (see note above).
    best_run_data["ligand_copies"] = copy.deepcopy(voted_copies)


def _build_validation_report(
    pdb_id: str,
    best_run_data: dict[str, Any],
    enriched_entry: dict[str, Any],
    all_warnings: list[str],
    chimera_result: dict[str, Any],
    validation_cache: ValidationCache | None,
) -> dict[str, Any]:
    """Assemble the validation report from all warning sources.

    None-safe reads throughout (``chimera_result.get("score") or 0``) and all
    status comparisons go through the shared constants. The G-alpha sequence
    finding is classified against the model's claim: family agreement, subtype
    resolution, and routing of an indistinguishable subtype to human review.
    """
    report: dict[str, Any] = {
        "critical_warnings": list(all_warnings),
        "algo_conflicts": [],
        "detector_notes": [],
        "chimera_score": chimera_result.get("score") or 0,
        "chimera_status": chimera_result.get("status") or CHIMERA_STATUS_SKIPPED,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    # G protein subunit fragments the model misfiled under auxiliary_proteins /
    # ligands (a GaCT / alpha5 peptide, or a tag-named beta/gamma subunit). An
    # unambiguous single-subunit fragment is MOVED into the G protein record (a
    # gating warning routes the move to a curator); a no-slug or cross-role fragment
    # is gated in place. This mutates the ligands / auxiliary_proteins lists, so it
    # must run BEFORE the integrity check's positional list-index recursion
    # (validate_all emits 'ligands[N]' paths that curate parses into index cleanups),
    # exactly as the excluded-buffer prune (step 10b) precedes this report for the
    # same reason.
    report["critical_warnings"].extend(
        relocate_misfiled_g_protein_fragments(enriched_entry, best_run_data)
    )

    # Integrity checks (ghost chain, fake UniProt/PubChem, ghost ligand, method)
    integrity_warnings = validate_all(pdb_id, best_run_data, enriched_entry, cache=validation_cache)
    report["critical_warnings"].extend(integrity_warnings)

    # Candidate-membership backstop: a specific G-alpha slug outside the curated
    # roster reaches a human. Deterministic, independent of the alpha5 API check,
    # so it fires even under --skip-api-checks.
    report["critical_warnings"].extend(_warn_on_unrecognised_g_alpha(best_run_data))

    # Non-GPCR polymer chains present in the structure but never annotated by the
    # model (the oligomer missed-protomer check covers GPCR chains only).
    report["critical_warnings"].extend(reconcile_missed_polymers(enriched_entry, best_run_data))

    # Coupling-aware state/ligand advisory: an active-state call carrying an
    # inactive-stabilising ligand with no transducer modelled asks a curator to
    # confirm the state.
    report["critical_warnings"].extend(state_ligand_consistency_warnings(best_run_data))

    # Detect-stage REVIEW signals -> curator critical warnings. This is the
    # production consumer of the detect review route (advisory signals already
    # went into the annotation prompt upstream); without it, a detector's review
    # signal -- and any signal whose severity failed safe to review -- never
    # reaches a human. Kinds the aggregator re-derives itself are excluded above.
    detect_reviews = [
        s for s in load_detect_signals(pdb_id) if s.kind not in _AGGREGATOR_OWNED_REVIEW_KINDS
    ]
    report["critical_warnings"].extend(to_critical_warnings(detect_reviews))

    # Receptor-side crystallization fusions (BRIL / T4 lysozyme) -- advisory,
    # non-blocking: recorded for the curator, does not gate accept-all.
    report["detector_notes"].extend(detect_crystallization_fusions(enriched_entry))

    # Auxiliary binders (Fab / nanobody / scFv / DARPin) the model named after the
    # antigen they bind -- rewritten to a canonical name from the RCSB chain
    # description. Deterministic and safe -> advisory note, mutates the name in place.
    report["detector_notes"].extend(
        correct_binder_names(enriched_entry, best_run_data.get("auxiliary_proteins"))
    )

    # Chimeric G protein review is driven by the deterministic alpha5 analysis,
    # NOT the model's optional is_chimeric flag (which the model can silently
    # omit -> a false negative that skips review). The model flag is kept only as
    # a fallback for when the alpha5 was INCONCLUSIVE -- it never ran
    # (--skip-api-checks), or ran but could not decide (too short / no reference
    # comparisons / error). In those cases there is no deterministic ruling, so a
    # self-declared chimera must still reach a human. When the alpha5 reached a
    # conclusion the fallback is suppressed: SUCCESS -> the alpha5 routing below
    # owns the review (it forces review only when it genuinely cannot resolve the
    # subtype, and stays silent once the identity is settled); NO_G_PROTEIN -> the
    # hallucination branch below owns it (the algorithm positively found none).
    status = chimera_result.get("status") or CHIMERA_STATUS_SKIPPED
    ai_uniprot = extract_ai_g_protein(best_run_data)
    g_protein = (best_run_data.get("signaling_partners") or {}).get("g_protein") or {}
    alpha5_inconclusive = status not in (CHIMERA_STATUS_SUCCESS, CHIMERA_STATUS_NO_G_PROTEIN)
    if alpha5_inconclusive and g_protein.get("is_chimeric") is True:
        report["critical_warnings"].append(
            f"{ALERT_PREFIX_CHIMERIC_REVIEW} at "
            f"'signaling_partners.g_protein.alpha_subunit': chimeric G protein — "
            f"confirm the alpha-subunit identity manually."
        )

    # Compare the alpha5 sequence finding against the model's G-alpha claim.

    if status == CHIMERA_STATUS_SUCCESS:
        family = chimera_result.get("family")
        subtype = chimera_result.get("subtype")
        resolution = chimera_result.get("subtype_resolution")
        candidate_set = chimera_result.get("candidate_set") or []
        a5_tail = chimera_result.get("a5_tail") or "N/A"
        ai_family = A5_SUBTYPE_FAMILY.get(ai_uniprot) if ai_uniprot else None

        # The functional coupling identity and the modelled backbone scaffold are
        # now two distinct fields. The alpha5 helix is the receptor-coupling
        # determinant, so it defines the FUNCTIONAL identity; the scaffold the
        # construct was built on is recorded separately and never substitutes for
        # it. The functional_coupling slug follows the most reliable source per
        # branch below:
        #   - alpha5 RESOLVED a single subtype -> the detector's resolved slug
        #     (reliable; it stands even when the model voted differently, in which
        #     case the [TIE-BREAKER OVERRIDE] still fires to gate the disagreement).
        #   - alpha5 reached only FAMILY (inseparable set the alpha5 cannot split)
        #     AND the model's slug is family-consistent -> the model's slug, since
        #     the detector cannot pin the member here (family-correct is
        #     functionally correct).
        #   - otherwise (family mismatch / absent / off-roster model slug) -> left
        #     unset so the accompanying [TIE-BREAKER OVERRIDE] / [UNRECOGNISED
        #     G-ALPHA] conflict drives the manual review.
        family_verified = ai_family is not None and family is not None and ai_family == family
        functional_coupling: str | None = None

        if subtype is not None:
            # The alpha5 resolves to a single subtype: store the detector's
            # resolved slug (the reliable source), independent of the model's vote.
            functional_coupling = subtype
            subtype_basis = SUBTYPE_BASIS_RESOLVED
            if ai_uniprot and ai_uniprot != subtype:
                report["algo_conflicts"].append(
                    f"{ALERT_PREFIX_TIE_BREAKER_OVERRIDE} at 'chimera_analysis': "
                    f"alpha5 '{a5_tail}' resolves G-alpha to '{subtype}', but the "
                    f"model chose '{ai_uniprot}'. Confirm the identity."
                )
            else:
                report["detector_notes"].append(
                    f"{ALERT_PREFIX_TIE_BREAKER_ALIGNED} at 'chimera_analysis': "
                    f"alpha5 '{a5_tail}' resolves G-alpha to '{subtype}'."
                )
        elif resolution == CHIMERA_SUBTYPE_LOW_CONFIDENCE:
            subtype_basis = SUBTYPE_BASIS_CONSTRUCT_NAME
            report["detector_notes"].append(
                f"{ALERT_PREFIX_ALGO_WARNING} at 'chimera_analysis': "
                f"alpha5 match is weak (best window score "
                f"{chimera_result.get('score') or 0}); G-alpha identity unverified."
            )
        elif ai_family and family and ai_family != family:
            # The model's family disagrees with the alpha5 coupling family.
            subtype_basis = SUBTYPE_BASIS_CONSTRUCT_NAME
            report["algo_conflicts"].append(
                f"{ALERT_PREFIX_TIE_BREAKER_OVERRIDE} at 'chimera_analysis': "
                f"alpha5 '{a5_tail}' indicates the {family} family, but the model "
                f"chose '{ai_uniprot}' ({ai_family}). Confirm the G-alpha identity."
            )
        elif family:
            # Family is confident but the subtype cannot be told apart by the
            # alpha5. When the off-roster slugs are non-human orthologs the call is
            # NOT an inseparable-subtype problem -- it is a species-mapping one, so
            # say so honestly rather than implying the subtype is ambiguous. Either
            # way the subtype is routed to a human rather than forced to a member.
            # The detector could not pin the member, so when the model's slug is
            # family-consistent it carries the functional coupling here.
            if family_verified:
                functional_coupling = ai_uniprot
                subtype_basis = SUBTYPE_BASIS_FAMILY_VERIFIED
            else:
                subtype_basis = SUBTYPE_BASIS_CONSTRUCT_NAME
            members = ", ".join(candidate_set) or "indistinguishable subtypes"
            off_roster = [s for s in candidate_set if s not in _RECOGNISED_G_ALPHA_SLUGS]
            if off_roster:
                report["critical_warnings"].append(
                    f"{ALERT_PREFIX_CHIMERIC_REVIEW} at "
                    f"'signaling_partners.g_protein.alpha_subunit': alpha5 indicates a "
                    f"non-human ortholog of the {family} family ({members}); confirm "
                    f"the species / GPCRdb mapping."
                )
            else:
                report["critical_warnings"].append(
                    f"{ALERT_PREFIX_CHIMERIC_REVIEW} at "
                    f"'signaling_partners.g_protein.alpha_subunit': alpha5 confirms the "
                    f"{family} family but cannot distinguish the subtype ({members}); "
                    f"confirm manually."
                )
        else:
            # The best match spans more than one coupling family or an
            # unrecognised slug, so even the family is undetermined. Never leave
            # this silent: surface it as a conflict for manual resolution.
            subtype_basis = SUBTYPE_BASIS_CONSTRUCT_NAME
            members = ", ".join(candidate_set) or "no recognised subtype"
            report["algo_conflicts"].append(
                f"{ALERT_PREFIX_ALGO_WARNING} at 'chimera_analysis': "
                f"alpha5 '{a5_tail}' does not map to a single coupling family "
                f"({members}); G-alpha identity cannot be determined automatically."
            )

        # Record the two distinct identities (plus provenance) on the alpha
        # subunit. These are aggregator-OWNED outputs -- the model never fills
        # them; the functional slug was chosen per branch above (the detector's
        # resolved subtype when the alpha5 pinned one, else the model's
        # family-matching slug). The backbone is always recorded for provenance,
        # independent of whether it differs from the alpha5: when the entity
        # carries no attached UniProt (e.g. a G-alpha deposited without an
        # accession) it falls back to an explicit "unknown" rather than being
        # silently dropped.
        alpha_block = g_protein.get("alpha_subunit")
        if isinstance(alpha_block, dict):
            # functional_coupling may be set even while a [TIE-BREAKER OVERRIDE]
            # review conflict is active (by design): the stored value is the best
            # determination (the detector's resolved subtype), while the conflict
            # still drives the curator's confirmation of the model/detector split.
            if functional_coupling is not None:
                alpha_block["functional_coupling"] = functional_coupling
            alpha_block["backbone"] = (
                chimera_result.get("backbone_slug") or CHIMERA_BACKBONE_UNKNOWN
            )
            alpha_block["subtype_basis"] = subtype_basis

        # alpha5-graft: the engineered scaffold differs from the functional
        # alpha5 (~6% of G-alpha structures). Note it -- informational, not a
        # conflict: the alpha5 helix is the principal receptor-coupling
        # determinant, so it defines the G-alpha identity, while the scaffold the
        # construct was built on is recorded separately on the alpha subunit above.
        if chimera_result.get("is_alpha5_graft"):
            backbone_slug = chimera_result.get("backbone_slug")
            backbone_family = chimera_result.get("backbone_family")
            report["detector_notes"].append(
                f"{ALERT_PREFIX_ALPHA5_GRAFT} at "
                f"'signaling_partners.g_protein.alpha_subunit': alpha5-graft chimera "
                f"-- backbone {backbone_slug or CHIMERA_BACKBONE_UNKNOWN} "
                f"({backbone_family}), functional alpha5 = {family}; identity follows "
                f"the alpha5 per convention."
            )
    elif status == CHIMERA_STATUS_NO_G_PROTEIN:
        if ai_uniprot and str(ai_uniprot).lower() not in EMPTY_VALUES:
            report["algo_conflicts"].append(
                f"{ALERT_PREFIX_HALLUCINATION} at 'chimera_analysis': "
                f"AI found '{ai_uniprot}' but algorithm found NO G protein "
                f"in source PDB."
            )
    elif status != CHIMERA_STATUS_SKIPPED:
        error_msg = chimera_result.get("error")
        report["algo_conflicts"].append(
            f"{ALERT_PREFIX_ALGO_WARNING} at 'chimera_analysis': "
            f"Verification could not run. Status: '{status}'. "
            f"Details: {error_msg}"
        )

    return report


# ---------------------------------------------------------------------------
# Atomic write block
# ---------------------------------------------------------------------------


def _write_outputs(
    pdb_id: str,
    best_run_data: dict[str, Any],
    discrepancies: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> AggregateResult:
    """Write aggregated JSON, voting log, and validation report atomically.

    All temp files are written first, then ``os.replace``-d (atomic write).
    ``try...finally`` guarantees cleanup on failure.
    """
    cfg = get_config()
    aggregated_path = cfg.aggregated_dir / f"{pdb_id}.json"
    voting_log_dir = cfg.aggregated_dir / "logs"
    validation_dir = cfg.aggregated_dir / "validation_logs"

    aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    voting_log_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # The voting log is written for every PDB, not only when discrepancies exist.
    # A clean PDB's log is an empty list -- an explicit, audit-friendly record that
    # aggregation ran and found no disagreement, distinct from a missing file. Each
    # discrepancy record already carries its per-field vote tallies (all_votes), so
    # the log preserves the vote shape that drove every flagged field. The payload
    # stays a list of records to honour the curate loader's contract; an empty list
    # yields an empty controversy map downstream, so a clean PDB is not gated.
    voting_log_path = voting_log_dir / f"{pdb_id}_voting_log.json"
    validation_path = validation_dir / f"{pdb_id}_validation.json"

    tmp_paths: list[str] = []
    try:
        # Write all temp files
        tmp_agg = _write_temp_json(aggregated_path.parent, best_run_data)
        tmp_paths.append(tmp_agg)

        tmp_val = _write_temp_json(validation_path.parent, validation_report)
        tmp_paths.append(tmp_val)

        tmp_log = _write_temp_json(voting_log_path.parent, discrepancies)
        tmp_paths.append(tmp_log)

        # Commit all at once
        os.replace(tmp_agg, str(aggregated_path))
        os.replace(tmp_val, str(validation_path))
        os.replace(tmp_log, str(voting_log_path))

        # Clear committed paths from cleanup list
        tmp_paths.clear()

        return AggregateResult(
            pdb_id=pdb_id,
            success=True,
            aggregated_path=aggregated_path,
            voting_log_path=voting_log_path,
            validation_path=validation_path,
        )
    finally:
        for tmp in tmp_paths:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _write_temp_json(directory: Path, data: Any) -> str:
    """Write *data* to a temp file in *directory* and return the temp path."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(directory),
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as fd:
        json.dump(data, fd, indent=4)
        return fd.name


# ---------------------------------------------------------------------------
# Aggregate log
# ---------------------------------------------------------------------------


def _update_aggregate_log(
    pdb_id: str,
    status: str,
) -> None:
    """Record *pdb_id* processing status in ``aggregate_log.json``.

    Uses atomic write.
    Never swallows exceptions silently — logs warnings.
    """
    cfg = get_config()
    log_path = cfg.state_dir / "aggregate_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_data: dict[str, Any] = {}
    if log_path.is_file():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                log_data = raw
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read aggregate log: %s", exc)

    log_data[pdb_id] = {
        "status": status,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(log_path.parent),
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fd:
            tmp_path = fd.name
            json.dump(log_data, fd, indent=2)
        os.replace(tmp_path, str(log_path))
        tmp_path = None
    except OSError as exc:
        logger.warning("Failed to update aggregate log for %s: %s", pdb_id, exc)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def aggregate_pdb(
    pdb_id: str,
    *,
    skip_api_checks: bool = False,
    validation_cache: ValidationCache | None = None,
    sequence_cache: SequenceCache | None = None,
    synonym_cache: SynonymCache | None = None,
    polymer_features_cache: PolymerFeaturesCache | None = None,
) -> AggregateResult:
    """Run the full aggregation + validation pipeline for a single PDB.

    Steps:
        1. Load AI runs
        2. Majority voting
        3. Select best run + deepcopy (mutation boundary)
        4. Load enriched data
        5. Inject ground truth
        6. Ligand validation
        7. Receptor validation
        8. Oligomer analysis
        9. Compute discrepancies
        10. Chimera analysis
        10b. Prune excluded-buffer ligands
        10c. Rebuild small-molecule rows from per-copy site votes
        11. Assemble validation report
        12. Atomic write block

    The ``validation_cache`` / ``sequence_cache`` / ``synonym_cache`` are supplied by
    the batch ``aggregate_all`` path. Single-PDB CLI invocations pass none, so the
    cache-gated API steps (chimera analysis and the PubChem synonym gate) are skipped
    there, consistent with the existing offline behaviour of the single-PDB path.

    Use ``if enriched is None:`` — NOT ``if not enriched:`` (an empty dict is valid).
    """
    # Fail fast on a stale / missing storage contract before doing real work.
    from gpcr_tools.workspace import validate_contract

    cfg = get_config()
    validate_contract(cfg)

    # The single-PDB paths (CLI `aggregate <PDB>` and the pipeline) pass no
    # polymer-features cache. Lazily construct one from the same cache file the
    # batch path uses, so single-PDB runs get the same fail-open protection (a
    # genuine TM-fetch failure routes to a curator instead of silently counting a
    # peptide as a receptor) and warm the shared cache for later runs. A
    # locally-built cache is saved at the end of this call; a caller-supplied one
    # is owned and saved by that caller (aggregate_all).
    owns_polymer_cache = polymer_features_cache is None
    if owns_polymer_cache:
        polymer_features_cache = PolymerFeaturesCache(cfg.cache_dir / POLYMER_FEATURES_CACHE_NAME)

    # 1. Load AI runs
    runs = load_ai_runs(pdb_id)
    if not runs:
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="No valid AI runs found",
        )

    # 2. Majority voting
    majority_votes, all_votes = get_majority_votes(runs)

    # 3. Select best run + deepcopy
    _best_idx, best_run_original = select_best_run(runs, majority_votes)
    best_run_data = copy.deepcopy(best_run_original)

    # 4. Load enriched data
    enriched = load_enriched_data(pdb_id)
    # if enriched is None — empty dict {} is valid
    if enriched is None:
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="Enriched data not available",
        )

    # An enrichment written during a transient API outage must NOT be aggregated:
    # consuming it would turn an unresolved (but recoverable) field into an
    # affirmative answer (e.g. a missing receptor slug → "no GPCR"). Refuse and
    # signal re-enrichment; a plain re-run of fetch self-heals the record.
    if enriched_is_incomplete(pdb_id):
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error="Enrichment incomplete (transient API gap) — re-run fetch to complete it before aggregating",
        )

    try:
        # 5. Inject ground truth (mutates best_run_data)
        inject_ground_truth(pdb_id, best_run_data, enriched)

        # 6. Ligand validation (mutates best_run_data, returns warnings)
        all_warnings: list[str] = []
        ligand_warnings = validate_and_enrich_ligands(
            pdb_id,
            best_run_data,
            enriched,
            synonym_cache=synonym_cache if not skip_api_checks else None,
        )
        all_warnings.extend(ligand_warnings)

        # 7. Receptor validation (mutates best_run_data, returns warnings)
        receptor_warnings = validate_receptor_identity(pdb_id, best_run_data, enriched)
        all_warnings.extend(receptor_warnings)

        # 8. Oligomer analysis (mutates best_run_data — may override chain_id). The
        # detect stage's geometric coupling-protomer signal, when present, selects the
        # primary protomer (the G protein coupler) over the AI's chain guess.
        analyze_oligomer(
            pdb_id,
            best_run_data,
            enriched,
            coupling_chain=_coupling_protomer(pdb_id),
            polymer_features_cache=polymer_features_cache,
        )

        # Persist a locally-built cache (single-PDB path) so the warmed TM fetch
        # survives to the next run; a caller-supplied cache is saved by the caller.
        if owns_polymer_cache and polymer_features_cache is not None:
            try:
                polymer_features_cache.save()
            except OSError as exc:
                logger.warning("[%s] Failed to save polymer-features cache: %s", pdb_id, exc)

        # 9. Compute discrepancies
        discrepancies = find_discrepancies(best_run_data, majority_votes, all_votes)
        # Also surface unanimous-but-low-confidence decision units for review
        # (consensus is not correctness); dedupe by path so a field already
        # flagged as a real disagreement or near-tie is not duplicated.
        low_conf = flag_low_confidence_consensus(best_run_data, LOW_CONFIDENCE_LEVELS)
        seen_paths = {d["path"] for d in discrepancies}
        discrepancies.extend(d for d in low_conf if d["path"] not in seen_paths)

        # 10. Chimera analysis
        chimera_result: dict[str, Any] = {
            "status": CHIMERA_STATUS_SKIPPED,
            "score": 0,
        }
        if not skip_api_checks and sequence_cache is not None:
            chimera_result = get_chimera_analysis(pdb_id, enriched, sequence_cache)

        # 10b. Prune excluded-buffer ligands (BOG / NAG / detergents) from the
        # record so they never reach the aggregated JSON, the curator, or the CSV
        # -- a genuinely-functional incidental lipid the model judged real is
        # kept. Must run BEFORE the validation report (step 11). The only emitter
        # of numeric ligands[N] paths is the integrity checker's generic list
        # recursion (integrity_checker.validate_all, run inside
        # _build_validation_report); those positional indices, which curate parses
        # into index cleanups, are correct only if the aggregated ligand list is
        # already pruned. (The ghost-ligand, oligomer, and voting warnings instead
        # key on comp id -- ligands[<comp_id>] -- so they are position-stable and
        # not the reason for this ordering.)
        _prune_excluded_buffer_ligands(best_run_data)

        # 10c. Re-derive each small-molecule ligand row from the aggregated
        # per-copy site attribution, so a physical copy is placed on the
        # binding-site row the runs agreed on rather than inheriting one outlier
        # run's copy list (which can inflate one row or drop another). Runs AFTER
        # discrepancy detection (step 9) so the best-run-vs-majority review gate is
        # already computed and preserved, and AFTER the buffer prune (step 10b) so
        # the post-prune list is the chemistry-and-existence source. Runs BEFORE
        # the validation report (step 11) for the same reason the buffer prune
        # does: the integrity checker emits positional ligands[N] paths that curate
        # parses into index cleanups, so the ligand list must be in its final shape
        # before those paths are produced.
        _rebuild_small_molecule_rows_from_per_copy(best_run_data, majority_votes)

        # 11. Assemble validation report
        v_cache = validation_cache if not skip_api_checks else None
        report = _build_validation_report(
            pdb_id,
            best_run_data,
            enriched,
            all_warnings,
            chimera_result,
            v_cache,
        )

        # 12. Atomic write block
        result = _write_outputs(pdb_id, best_run_data, discrepancies, report)
        result.warnings = report["critical_warnings"]
        return result
    except Exception as exc:
        logger.error("[%s] Pipeline failure: %s", pdb_id, exc)
        return AggregateResult(
            pdb_id=pdb_id,
            success=False,
            error=str(exc),
        )


def _pdbs_with_api_unavailable(cfg: Any) -> list[str]:
    """PDB ids whose last validation report recorded a transient API abstention.

    A report carrying an ``[API_UNAVAILABLE]`` warning means a UniProt/PubChem
    lookup abstained (the result was not cached), so re-aggregating that PDB
    retries only those lookups -- the definitive results stay cached.
    """
    validation_dir = cfg.aggregated_dir / "validation_logs"
    if not validation_dir.is_dir():
        return []
    suffix = "_validation.json"
    pdb_ids: list[str] = []
    for path in sorted(validation_dir.glob(f"*{suffix}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        warnings = data.get("critical_warnings") or []
        if any(ALERT_PREFIX_API_UNAVAILABLE in str(w) for w in warnings):
            pdb_ids.append(path.name[: -len(suffix)])
    return pdb_ids


def aggregate_all(
    *,
    skip_api_checks: bool = False,
    force: bool = False,
    retry_unavailable: bool = False,
) -> list[AggregateResult]:
    """Aggregate all pending PDBs with per-PDB error isolation.

    Args:
        skip_api_checks: Skip UniProt/PubChem/chimera API calls.
        force: Re-process PDBs already in the aggregate log.
        retry_unavailable: Re-process only PDBs whose last validation report
            recorded a transient API abstention (``[API_UNAVAILABLE]``).
            Definitive results are cached, so only the failed lookups are retried.
            Cannot be combined with ``skip_api_checks`` (which would skip the very
            checks the retry exists to re-run). It also cannot recover a cache that
            an older build poisoned with a transient failure stored as a definitive
            result -- those read back as a verdict, not an abstention, and must be
            evicted manually.

    Returns list of :class:`AggregateResult` for each processed PDB.
    """
    if retry_unavailable and skip_api_checks:
        raise ValueError(
            "retry_unavailable cannot be combined with skip_api_checks: the retry "
            "exists to re-run the API checks that skip_api_checks disables."
        )

    try:
        cfg = get_config()
    except Exception as exc:
        logger.error("Failed to initialize workspace config: %s", exc)
        return []

    # Fail fast on a stale / missing storage contract before the batch begins.
    from gpcr_tools.workspace import validate_contract

    validate_contract(cfg)

    # Cache initialization
    try:
        validation_cache = ValidationCache(cfg.cache_dir / "id_validation_cache.json")
        sequence_cache = SequenceCache(cfg.cache_dir / "uniprot_sequence_cache.json")
        synonym_cache = JsonCache(cfg.cache_dir / "pubchem_synonym_cache.json")
        polymer_features_cache = PolymerFeaturesCache(cfg.cache_dir / POLYMER_FEATURES_CACHE_NAME)
    except Exception as exc:
        logger.error("Failed to initialize caches: %s", exc)
        return []

    if retry_unavailable:
        # Only re-run PDBs whose AI results are still present, so a stale
        # validation report never downgrades a PDB that can no longer be rebuilt.
        pending = [
            pdb_id
            for pdb_id in _pdbs_with_api_unavailable(cfg)
            if pdb_has_runs(cfg.ai_results_dir / pdb_id)
        ]
        if not pending:
            logger.info(
                "[aggregate] --retry-unavailable: no PDB with AI results recorded a "
                "transient API failure; nothing to retry."
            )
            return []
    elif force:
        # Get ALL PDB IDs with AI results (bypass aggregate log)
        ai_dir = cfg.ai_results_dir
        if not ai_dir.is_dir():
            return []
        pending = sorted(d.name for d in ai_dir.iterdir() if pdb_has_runs(d))
    else:
        pending = get_pending_pdb_ids()

    from tqdm import tqdm

    results: list[AggregateResult] = []
    for pdb_id in tqdm(pending, desc="Progress"):
        try:
            result = aggregate_pdb(
                pdb_id,
                skip_api_checks=skip_api_checks,
                validation_cache=validation_cache,
                sequence_cache=sequence_cache,
                synonym_cache=synonym_cache,
                polymer_features_cache=polymer_features_cache,
            )
            results.append(result)
            status = AGG_STATUS_COMPLETED if result.success else AGG_STATUS_FAILED
            _update_aggregate_log(pdb_id, status)
        except Exception as exc:
            logger.error("[%s] Critical failure: %s", pdb_id, exc)
            results.append(AggregateResult(pdb_id=pdb_id, success=False, error=str(exc)))
            _update_aggregate_log(pdb_id, AGG_STATUS_FAILED)

    # Surface transient API abstentions so they can be retried in isolation.
    unavailable = [
        r.pdb_id for r in results if any(ALERT_PREFIX_API_UNAVAILABLE in str(w) for w in r.warnings)
    ]
    if unavailable:
        logger.warning(
            "[aggregate] %d PDB(s) hit a transient API failure (not cached): %s. "
            "Re-run 'gpcr-tools aggregate --retry-unavailable' to retry only these.",
            len(unavailable),
            ", ".join(unavailable),
        )

    # Save caches (best-effort, after output commit)
    try:
        validation_cache.save()
    except OSError as exc:
        logger.warning("Failed to save validation cache: %s", exc)
    try:
        sequence_cache.save()
    except OSError as exc:
        logger.warning("Failed to save sequence cache: %s", exc)
    try:
        synonym_cache.save()
    except OSError as exc:
        logger.warning("Failed to save synonym cache: %s", exc)
    try:
        polymer_features_cache.save()
    except OSError as exc:
        logger.warning("Failed to save polymer-features cache: %s", exc)

    return results
