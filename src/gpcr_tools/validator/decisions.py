"""Enumerate a structure's review decisions as an ordered, per-signal list.

Where :mod:`gpcr_tools.validator.gating` answers the single yes/no question
"does this structure need a human?", this module answers the richer one the
curator actually acts on: "*which* signals ask for a decision, and why?". It
returns one :class:`DecisionItem` per signal -- every validation finding, every
gating oligomer finding, and **every fork in the raw voting log** -- ranked
most-severe first.

The vote forks are read straight from the raw voting log, INDEPENDENT of the
whole-structure gated boolean: a structure that looks clean by its validation
and oligomer reasons can still carry vote forks, and they are all enumerated so
the curator sees the complete picture rather than only the findings that happen
to trip the boolean gate.

This module is UI-free (no Rich, no I/O) and depends only on ``config`` and the
sibling ``gating`` module, so the interactive curator and any offline caller
reach the same enumeration from one implementation. Every input is optional: a
missing source contributes nothing rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gpcr_tools.config import VOTE_NEAR_TIE_MARGIN
from gpcr_tools.validator.gating import oligomer_gating_warnings

# ── Category labels (domain language, stable enough to group/tally on) ──────
CATEGORY_CRITICAL_WARNING = "CRITICAL VALIDATION"
CATEGORY_ALGORITHM_CONFLICT = "ALGORITHM CONFLICT"
CATEGORY_OLIGOMER = "OLIGOMER"
CATEGORY_VOTE_DISAGREEMENT = "VOTE DISAGREEMENT"
CATEGORY_VOTE_NEAR_TIE = "VOTE NEAR-TIE"
CATEGORY_VOTE_LOW_CONFIDENCE = "VOTE LOW-CONFIDENCE"
CATEGORY_VOTE_OMISSION = "MINORITY OMISSION"
CATEGORY_VOTE_ADVISORY = "VOTE (advisory)"

# ── Severity ranks (higher sorts earlier; ordering only, not field values) ──
SEVERITY_CRITICAL_WARNING = 60
SEVERITY_ALGORITHM_CONFLICT = 50
SEVERITY_OLIGOMER = 40
SEVERITY_VOTE_DISAGREEMENT = 30
SEVERITY_VOTE_NEAR_TIE = 20
SEVERITY_VOTE_LOW_CONFIDENCE = 15
SEVERITY_VOTE_OMISSION = 12
SEVERITY_VOTE_ADVISORY = 10


@dataclass(frozen=True)
class DecisionItem:
    """One thing a curator is being asked to decide about a structure."""

    category: str  # one of the CATEGORY_* labels
    path: str  # where in the annotation this decision lives ("" = structure-level)
    summary: str  # one-line human-facing description of the decision
    gating: bool  # whether this signal routes the structure to a human
    severity: int  # sort key; higher = more severe (see SEVERITY_* above)
    best_value: Any = None  # best-run value for a vote fork (else None)
    majority_value: Any = None  # majority-vote value for a vote fork (else None)
    votes: dict[str, Any] | None = None  # per-candidate vote counts for a fork
    margin: int | None = None  # top-two vote margin for a fork (None if N/A)
    evidence: str | None = None  # human-relevant evidence already in the data


def _marker_path(message: Any) -> str:
    """Pull the ``at '<path>'`` anchor out of a curator-facing message, or ''.

    Validation and oligomer messages carry their annotation path inline as
    ``... at 'receptor_info' ...``; surface it as the item's path so the brief
    can point the curator at the right block.
    """
    text = str(message)
    marker = "at '"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = text.find("'", start)
    return text[start:end] if end != -1 else ""


def _field_from_path(path: str) -> str:
    """The human-meaningful terminal field of a controversy path.

    Drops a trailing ``.value`` (a decision-unit wrapper) and any list-index
    brackets so ``ligands[ZMA].role.value`` reads as ``role`` and
    ``signaling_partners.g_protein.is_chimeric`` reads as ``is_chimeric``.
    """
    if not path:
        return ""
    trimmed = path[: -len(".value")] if path.endswith(".value") else path
    tail = trimmed.split(".")[-1]
    if "[" in tail:
        tail = tail.split("[")[0]
    return tail or path


def _vote_margin(vote_counts: Any) -> int | None:
    """Votes separating the top two candidates, or ``None`` if fewer than two.

    A lone candidate is a consensus, never a near-tie -- it has no runner-up to
    tie with. A 5:5 split yields 0. Non-integer counts are skipped defensively.
    (Mirrors the aggregator's own near-tie guard; kept local so this lower-layer
    validator module never has to import the curator package.)
    """
    numeric: list[int] = []
    for count in vote_counts:
        try:
            numeric.append(int(count))
        except (TypeError, ValueError):
            continue
    numeric.sort(reverse=True)
    if len(numeric) < 2:
        return None
    return numeric[0] - numeric[1]


def _record_gates(record: dict) -> bool:
    """Whether one voting-log fork routes the structure to a human.

    Pinned semantics for the ``gating`` flag, reconciling a known divergence: the
    boolean gate (:func:`gpcr_tools.validator.gating.has_gating_controversy`)
    read an EXPLICIT ``gating=None`` as non-gating -- because ``dict.get(key,
    default)`` returns the stored ``None`` rather than the default -- while the
    offline digest read the same ``None`` as gating. We PIN the conservative
    reading here: a fork gates unless its flag is *explicitly* ``False``. An
    absent flag OR an explicit ``None`` both mean "the aggregator did not clear
    this fork", so both route to a human.
    """
    return record.get("gating") is not False


def _normalize_voting_records(voting_log: Any) -> list[dict]:
    """The raw voting-log forks as a list of records, from a list or a map.

    Accepts either the on-disk list of fork records or the path-keyed map the
    curator already builds from it, so callers can pass whichever they hold.
    Drops anything that is not a dict carrying a ``path``.
    """
    if isinstance(voting_log, dict):
        candidates: Any = voting_log.values()
    elif isinstance(voting_log, list):
        candidates = voting_log
    else:
        return []
    return [r for r in candidates if isinstance(r, dict) and r.get("path")]


def _unwrap_decision_value(node: Any) -> Any:
    """Read a decision-unit's chosen value: ``{"value": X, ...}`` -> ``X``, else *node*.

    Several entity fields (e.g. a ligand ``role``) are decision units -- a dict
    wrapping the chosen ``value`` next to its confidence/evidence -- so unwrap to
    the value before showing it.
    """
    if isinstance(node, dict) and "value" in node:
        return node.get("value")
    return node


def _omitted_entity_summary(entity: dict) -> tuple[str, str | None]:
    """A one-line description plus evidence for a whole-entity minority omission.

    *entity* is the full ligand or auxiliary-protein dict the majority of runs
    reported but the selected run left out. Describe it from the fields present
    -- name, chem_comp_id, role, type, chain -- so the brief names the omitted
    entity (which the tree walk never reaches, as it is absent from the selected
    annotation) instead of dumping the entity dict's Python repr as one line.
    """
    name = entity.get("name") or entity.get("chem_comp_id") or "entity"
    label = str(name)
    comp_id = entity.get("chem_comp_id")
    if comp_id and str(comp_id) != label:
        label = f"{label} ({comp_id})"
    summary = f"{label}: reported by other runs, omitted by the selected run"

    details: list[str] = []
    role = _unwrap_decision_value(entity.get("role"))
    if role:
        details.append(f"role {role}")
    etype = entity.get("type")
    if etype:
        details.append(f"type {etype}")
    chain = entity.get("chain_id")
    if chain:
        details.append(f"chain {chain}")
    return summary, (", ".join(details) or None)


def _vote_item(record: dict) -> DecisionItem:
    """Build the decision item for one raw voting-log fork."""
    path = str(record.get("path") or "")
    best = record.get("best_run_value")
    majority = record.get("majority_vote_value")
    gates = _record_gates(record)

    # A whole-entity minority omission: some runs reported a ligand or auxiliary
    # protein the selected run left out, so the record's majority value is the
    # FULL entity dict (with a nested per-field all_votes) rather than a scalar
    # leaf. Describe the omitted entity from its own fields instead of rendering
    # the dict's repr as one line; the nested per-field breakdown is not a flat
    # {value: count} candidate tally, so vote counts and a top-two margin do not
    # apply and are left unset.
    if isinstance(majority, dict):
        summary, evidence = _omitted_entity_summary(majority)
        return DecisionItem(
            category=CATEGORY_VOTE_OMISSION,
            path=path,
            summary=summary,
            gating=gates,
            severity=SEVERITY_VOTE_OMISSION,
            best_value=best,
            majority_value=majority,
            votes=None,
            margin=None,
            evidence=evidence,
        )

    raw_votes = record.get("all_votes")
    votes = raw_votes if isinstance(raw_votes, dict) and raw_votes else None
    margin = _vote_margin(votes.values()) if votes else record.get("vote_margin")

    disagreement = best is not None and majority is not None and best != majority

    near_tie = margin is not None and margin <= VOTE_NEAR_TIE_MARGIN

    if not gates:
        # Advisory-only fork (e.g. a wording variant of a display name): surfaced
        # for completeness but it does not route the structure to a human.
        category = CATEGORY_VOTE_ADVISORY
        severity = SEVERITY_VOTE_ADVISORY
    elif disagreement:
        category = CATEGORY_VOTE_DISAGREEMENT
        severity = SEVERITY_VOTE_DISAGREEMENT
    elif near_tie:
        # The runs land on the same value, but by a fragile margin the aggregator
        # judged too close to present as settled.
        category = CATEGORY_VOTE_NEAR_TIE
        severity = SEVERITY_VOTE_NEAR_TIE
    else:
        # Gating with the runs in agreement and no competitive runner-up: a
        # unanimous low-confidence call the aggregator still wants confirmed.
        category = CATEGORY_VOTE_LOW_CONFIDENCE
        severity = SEVERITY_VOTE_LOW_CONFIDENCE

    field = _field_from_path(path) or path or "?"
    if disagreement:
        summary = f"{field}: best run {best!r} vs majority {majority!r}"
    elif near_tie:
        summary = f"{field}: {majority!r} by a {margin}-vote margin"
    else:
        summary = f"{field}: {majority!r} (low-confidence)"

    return DecisionItem(
        category=category,
        path=path,
        summary=summary,
        gating=gates,
        severity=severity,
        best_value=best,
        majority_value=majority,
        votes=votes,
        margin=margin,
    )


def enumerate_decisions(
    main_data: dict | None,
    validation_data: dict | None,
    voting_log: list | dict | None,
) -> list[DecisionItem]:
    """Every review decision for a structure, ranked most-severe first.

    Unions the three algorithmic sources into one ordered list:

    - validation-log critical warnings and algorithm conflicts (a cross-field
      contradiction -- e.g. a declared identity that disagrees with the detected
      one -- arrives here as an ``ALGORITHM CONFLICT`` item carrying the message);
    - gating oligomer findings (:func:`oligomer_gating_warnings`);
    - **every fork in the raw voting log**, read independently of the
      whole-structure gated boolean, so forks are enumerated even for a structure
      that would look clean by its validation and oligomer reasons.

    Each source is optional; an absent one contributes nothing. Items are sorted
    by descending severity (stable, so within a severity the input order is
    preserved).
    """
    items: list[DecisionItem] = []
    validation = validation_data or {}

    for warning in validation.get("critical_warnings") or []:
        message = str(warning)
        items.append(
            DecisionItem(
                category=CATEGORY_CRITICAL_WARNING,
                path=_marker_path(message),
                summary=message,
                gating=True,
                severity=SEVERITY_CRITICAL_WARNING,
                evidence=message,
            )
        )

    for conflict in validation.get("algo_conflicts") or []:
        message = str(conflict)
        items.append(
            DecisionItem(
                category=CATEGORY_ALGORITHM_CONFLICT,
                path=_marker_path(message),
                summary=message,
                gating=True,
                severity=SEVERITY_ALGORITHM_CONFLICT,
                evidence=message,
            )
        )

    oligo = (main_data or {}).get("oligomer_analysis")
    for message in oligomer_gating_warnings(oligo):
        items.append(
            DecisionItem(
                category=CATEGORY_OLIGOMER,
                path=_marker_path(message),
                summary=message,
                gating=True,
                severity=SEVERITY_OLIGOMER,
                evidence=message,
            )
        )

    for record in _normalize_voting_records(voting_log):
        items.append(_vote_item(record))

    items.sort(key=lambda item: -item.severity)
    return items
