"""Single source of truth for whether a PDB needs human review (is "gated").

A PDB is gated when any algorithmic source disagrees with, or cannot vouch for,
the model's annotation: a validation-log critical warning / algo conflict, an
oligomer finding worth a curator's eye, or a vote-aggregation controversy that
is not a mere advisory.

This module is UI-free and depends only on ``config`` so the interactive curator,
the auto-accept pass, and the read-only run manifest all reach the same verdict
from one implementation instead of three drifting copies. Every input is treated
as optional: a missing source (e.g. no aggregated file on disk) simply
contributes nothing rather than raising.
"""

from __future__ import annotations

import re

from gpcr_tools.config import (
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    ALERT_MULTI_COPY_LIGAND,
    ALERT_NO_GPCR,
    ALERT_NON_RECEPTOR_PARTNER,
    ALERT_OLIGOMER_DISAGREEMENT,
    ALERT_SUSPICIOUS_7TM,
    ALERT_TM_DATA_UNAVAILABLE,
    TM_STATUS_INCOMPLETE,
    ensure_alert_prefix,
)

# Oligomer alert types that should stop the curator at ``receptor_info``. A
# CONFIRMED_OLIGOMER (the roster matched) or an ASSEMBLY_MISMATCH (an advisory
# "confirm" note) is informational, not gating, so neither is in this set.
_GATING_OLIGOMER_ALERTS: frozenset[str] = frozenset(
    {
        ALERT_HALLUCINATION,
        ALERT_MISSED_PROTOMER,
        ALERT_SUSPICIOUS_7TM,
        ALERT_NO_GPCR,
        ALERT_TM_DATA_UNAVAILABLE,
        ALERT_OLIGOMER_DISAGREEMENT,
        ALERT_NON_RECEPTOR_PARTNER,
    }
)

# Block key of the per-copy ligand table (one row per modelled copy, each
# recording the binding site that copy occupies).
_LIGAND_COPIES_BLOCK: str = "ligand_copies"

# The compound-level routing anchor a multi-copy ligand alert is built with,
# e.g. "at 'ligands[CLR]':". Matched so the mirror can re-anchor the same text at
# the per-copy table while keeping the component id visible to the curator.
_LIGANDS_ANCHOR_RE = re.compile(r"at '\.?ligands\[(?P<component>[^\]]*)\]'\s*:?\s*")


def _mirror_at_ligand_copies(alert_type: str, message: str | None) -> str:
    """Re-anchor a compound-level ligand alert at the per-copy ligand table.

    A multi-copy ligand alert is raised against the compound (``ligands[<comp>]``),
    but the compound-level row cannot say which copy sits at which site -- only the
    per-copy table can, and that is where a wrong site assignment is corrected.
    Rewriting the anchor to ``ligand_copies`` makes the alert reachable from that
    table, and the component id is carried over in words so the curator still knows
    which compound is meant without re-introducing a compound-level anchor (which
    would route the mirror straight back to the compound table).

    The ``[TYPE]`` label is preserved exactly once via :func:`ensure_alert_prefix`,
    so a back-catalogue message stored without its label still gets one, and a
    current message keeps the single label it already carries. A message with no
    recognizable compound anchor is simply anchored at the per-copy table as-is.
    """
    label = f"[{alert_type}]"
    body = ensure_alert_prefix(alert_type, message)[len(label) :].strip()

    anchor = f"at '{_LIGAND_COPIES_BLOCK}':"
    match = _LIGANDS_ANCHOR_RE.search(body)
    if match:
        anchor = f"at '{_LIGAND_COPIES_BLOCK}' (component {match.group('component')}):"
        body = (body[: match.start()] + body[match.end() :]).strip()

    return f"{label} {anchor} {body}".strip()


def oligomer_gating_warnings(oligo: dict | None) -> list[str]:
    """Oligomer findings that should gate review, as curator-facing strings.

    Produces the same strings, in the same order, the interactive curator
    surfaces under ``receptor_info`` (a corrected chain id, a gating oligomer
    alert, a multi-copy ligand, an incomplete 7TM domain). Returns ``[]`` for an
    absent or empty oligomer analysis.

    A gating multi-copy ligand alert yields TWO strings: the compound-anchored
    original plus a copy re-anchored at ``ligand_copies``, because the per-copy
    table is the only block whose fields can record which copy sits at which
    binding site (see :func:`_mirror_at_ligand_copies`).
    """
    if not oligo:
        return []

    warnings: list[str] = []

    override = oligo.get("chain_id_override") or {}
    if override.get("applied"):
        warnings.append(
            f"CHAIN_ID CORRECTED at 'receptor_info': "
            f"{override.get('original_chain_id')} -> {override.get('corrected_chain_id')} "
            f"({override.get('trigger')}). Human confirmation required."
        )

    for alert in oligo.get("alerts") or []:
        atype = alert.get("type") or ""
        if atype in _GATING_OLIGOMER_ALERTS:
            if atype == ALERT_OLIGOMER_DISAGREEMENT and not alert.get("gating", True):
                # An OD alert the aggregator downgraded to advisory: the AI released
                # monomer while the classifier counted >=2 same-slug chains, but
                # RCSB's own global biological assembly records the receptor as a
                # single copy (a Monomer assembly, or an all-single stoichiometry),
                # so the released monomer agrees with both the AI and RCSB. Still
                # surfaced to the curator via the alert list, but not gating -- the
                # same non-gating policy the parallel ASSEMBLY_MISMATCH advisory
                # already carries. The flag defaults to True so a back-catalogue OD
                # alert recorded before the flag existed still gates rather than
                # being silently waved through.
                continue
            # The "at 'receptor_info'" prefix is a routing anchor so this alert
            # buckets under the receptor block during review. ensure_alert_prefix
            # keeps the message's own "[TYPE]" label present exactly once --
            # current validator messages already carry it (re-prepending would
            # duplicate it), while older recorded data needs it added.
            message = ensure_alert_prefix(atype, alert.get("message"))
            warnings.append(f"OLIGOMER ALERT at 'receptor_info': {message}")
        elif atype == ALERT_MULTI_COPY_LIGAND and alert.get("gating", True):
            # A multi-copy ligand gates only when its copies sit at distinct binding
            # sites; the aggregator stamps that decision on the alert's ``gating``
            # flag (copies sharing one site are advisory, still surfaced elsewhere).
            # The flag defaults to True so a back-catalogue alert recorded before the
            # flag existed still gates rather than being silently waved through.
            # Already carries its own 'ligands[...]' path, so it buckets with the
            # ligand block during review rather than under receptor_info.
            warnings.append(alert.get("message") or "")
            # ... but the compound-level ligand row has no field for a per-copy
            # site: the copies are only separable in the per-copy ligand table, so
            # that is the one block where a curator can act on this alert. Emit an
            # additional copy of the same finding re-anchored there, so the per-copy
            # table is opened for review instead of passing through unseen. Only a
            # gating alert is mirrored -- copies that share one binding site need no
            # per-copy decision, so mirroring those would be pure prompt noise.
            warnings.append(_mirror_at_ligand_copies(atype, alert.get("message")))

    if any(c.get("7tm_status") == TM_STATUS_INCOMPLETE for c in oligo.get("all_gpcr_chains") or []):
        warnings.append(
            "STRUCTURAL QUALITY at 'receptor_info': "
            "One or more GPCR chains have INCOMPLETE 7TM domains."
        )

    return warnings


def has_gating_controversy(controversies: dict | None) -> bool:
    """Whether any vote-aggregation controversy should gate review.

    Advisory-only controversies (records tagged ``gating=False``) are surfaced
    to the curator but do not gate. These include a minority omission (an entity
    some runs reported but the chosen run omitted) and any near-tie / genuine
    disagreement on a field that cannot encode a real error once identity is
    settled -- a lexical ``name`` wording variant, or a ``pubchem_id`` split
    with a blank shipped value or an authoritative ``api_pubchem_cid`` backstop.
    Every other controversy, and any record without an explicit ``gating`` flag
    (default ``True``), gates. Tolerant of ``None`` / an empty map (contributes
    nothing).
    """
    if not controversies:
        return False
    return any(c.get("gating", True) for c in controversies.values())


def is_pdb_gated(
    validation_data: dict | None,
    oligo: dict | None,
    controversies: dict | None,
) -> bool:
    """Whether a PDB needs human review, from all three algorithmic sources.

    Gated iff ANY of:
    - the validation log carries a critical warning or an algo conflict;
    - the oligomer analysis carries a gating finding
      (:func:`oligomer_gating_warnings`);
    - a vote-aggregation controversy gates (:func:`has_gating_controversy`).

    Each source is optional: an absent one (missing file, ``None``, empty)
    contributes nothing, so a clean PDB with no sibling artifacts is not gated.
    """
    if validation_data and (
        validation_data.get("critical_warnings") or validation_data.get("algo_conflicts")
    ):
        return True
    if oligomer_gating_warnings(oligo):
        return True
    return has_gating_controversy(controversies)
