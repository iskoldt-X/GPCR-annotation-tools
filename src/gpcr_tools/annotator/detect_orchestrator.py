"""Pure orchestration that routes detect signals into the annotation prompt/tool.

ADVISORY detect signals become evidence in the prompt (the model weighs them
against the paper); REVIEW signals are not handled here -- they route silently to
human review. An incidental-candidate advisory additionally augments the tool schema
with an optional ``pharmacological_role_check`` field, and a dual-role advisory with an
optional ``site_ref`` field. No I/O, no AI calls.

When there are no advisory signals the prompt block is ``None`` and the tool /
config are returned by identity, so an ordinary structure is byte-for-byte
unchanged.

NOTE: the user-facing wording in the prompt block is a DRAFT pending review.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from gpcr_tools.annotator.schema import (
    ANNOTATION_TOOL,
    PHARMACOLOGICAL_ROLE_CHECK_SCHEMA,
    TOOL_CONFIG,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_COUPLING_PROTOMER,
    SIGNAL_DUAL_ROLE_LIGAND,
    SIGNAL_INCIDENTAL_CANDIDATE,
    SIGNAL_SITE_REF,
    DetectSignal,
)

# A pocket-residue list is truncated to this many numbers in the prompt evidence.
_MAX_POCKET_RESIDUES_SHOWN = 12

# The advisory kinds that have a reviewed, model-facing formatter below. ONLY
# these reach the prompt. Any other advisory kind -- e.g. a future detector kind
# with no reviewed formatter yet -- is dropped here rather than leaked verbatim
# into the model prompt. Add a kind to this set only together with a reviewed
# formatter branch in _format_signal.
_MODEL_FACING_KINDS = frozenset(
    {
        SIGNAL_CHIMERIC_GPROTEIN,
        SIGNAL_COUPLING_PROTOMER,
        SIGNAL_INCIDENTAL_CANDIDATE,
        SIGNAL_DUAL_ROLE_LIGAND,
        SIGNAL_SITE_REF,
    }
)

# DRAFT wording -- pending Binghan's word-by-word review.
_DETECT_BLOCK_HEADER = (
    "=== DETECTOR EVIDENCE (computed before annotation) ===\n"
    "Treat each item below as evidence to weigh against the paper, not as a "
    "settled conclusion:"
)


def _format_signal(signal: DetectSignal) -> str | None:
    """Render one advisory signal as a prompt evidence line (DRAFT wording).

    Returns ``None`` for any kind without a reviewed model-facing formatter, so
    an unreviewed summary (e.g. a future detector kind's note) is never leaked
    into the prompt. Only kinds in ``_MODEL_FACING_KINDS`` produce a line.
    """
    if signal.kind not in _MODEL_FACING_KINDS:
        return None
    payload = signal.payload or {}
    if signal.kind == SIGNAL_CHIMERIC_GPROTEIN:
        tail = payload.get("a5_tail") or "?"
        family = payload.get("family") or "?"
        subtype = payload.get("subtype") or "an indistinguishable subtype"
        return (
            f"G-protein alpha5 analysis: the modelled alpha5 tail '{tail}' matches the "
            f"{family} family (subtype {subtype}). Weigh this against the paper before "
            f"assigning the G-alpha identity."
        )
    if signal.kind == SIGNAL_COUPLING_PROTOMER:
        chain = payload.get("coupling_chain") or "?"
        slug = payload.get("coupling_slug") or "?"
        return (
            f"Structure geometry shows the G protein engages receptor chain {chain} "
            f"({slug}); that protomer is the active, G-protein-coupling one — in a "
            f"heterodimer not necessarily the agonist-binding protomer. Weigh this "
            f"against the paper."
        )
    if signal.kind == SIGNAL_INCIDENTAL_CANDIDATE:
        comp = payload.get("comp_id") or "?"
        return (
            f"{comp} is present; it can be a functional ligand in some structures and an "
            f"incidental structural component in others. Judge its role from the paper "
            f"and record a pharmacological_role_check."
        )
    if signal.kind == SIGNAL_DUAL_ROLE_LIGAND:
        return _format_dual_role(payload)
    if signal.kind == SIGNAL_SITE_REF:
        comp = payload.get("comp_id") or "?"
        sites = payload.get("sites") or []
        if len(sites) == 1:
            return (
                f"{comp}: structure geometry places it at the {sites[0]} site — record "
                f"site_ref='{sites[0]}' unless the paper clearly says otherwise."
            )
        return (
            f"{comp}: structure geometry shows it at {len(sites)} distinct sites "
            f"({', '.join(sites)}) — emit one ligand entry per site, each with its site_ref."
        )
    # A whitelisted kind with no branch above (should not happen): never leak a
    # raw summary -- the _MODEL_FACING_KINDS guard and this fall-through agree.
    return None


def _format_dual_role(payload: dict[str, Any]) -> str:
    """Render the dual-role signal as burial evidence, one line per buried copy.

    This provides geometric evidence that the ligand sits in more than one pocket
    (so it may play more than one role); it deliberately does NOT command a split
    into one entry per site -- the site_ref signal owns that instruction, since it
    names each site. When site_ref resolves the pockets to distinct site classes
    it carries the split nudge; if both pockets are the same class it does not, and
    this burial evidence still flags the possible multiple roles for the model.
    """
    comp = payload.get("comp_id") or "?"
    chain = payload.get("gpcr_chain") or "?"
    copies = payload.get("copies") or []
    copy_lines = []
    for copy in copies:
        residues = copy.get("pocket_residues") or []
        shown = ", ".join(str(r) for r in residues[:_MAX_POCKET_RESIDUES_SHOWN])
        if len(residues) > _MAX_POCKET_RESIDUES_SHOWN:
            shown += ", ..."
        partner = (
            " and also contacts a non-receptor protein partner (possible active-state pocket)"
            if copy.get("contacts_partner")
            else ""
        )
        copy_lines.append(
            f"  copy {copy.get('chain')}/{copy.get('seq_id')}: buried "
            f"(enclosure {copy.get('burial')}), lines {copy.get('n_pocket_residues')} "
            f"receptor residues [{shown}]{partner}"
        )
    body = "\n".join(copy_lines)
    return (
        f"{comp} is buried in {len(copies)} distinct receptor pockets on chain {chain} "
        f"(geometry below), so weigh whether it plays more than one role; the computed "
        f"site_ref names each site:\n{body}"
    )


def _advisory_signals(signals: list[DetectSignal]) -> list[DetectSignal]:
    return [s for s in signals if s.severity == SEVERITY_ADVISORY]


def assemble_detect_block(signals: list[DetectSignal]) -> str | None:
    """Build the prompt evidence block from advisory signals, or ``None`` if none.

    Deterministic order (by kind, target_ref, comp_id) so the prompt is stable.
    """
    advisory = _advisory_signals(signals)
    if not advisory:
        return None
    ordered = sorted(
        advisory,
        key=lambda s: (s.kind, s.target_ref, str((s.payload or {}).get("comp_id") or "")),
    )
    # Drop kinds with no reviewed model-facing formatter (None) -- never leak a
    # raw summary. If nothing renders, there is no block.
    rendered = [line for s in ordered if (line := _format_signal(s)) is not None]
    if not rendered:
        return None
    lines = "\n".join(f"- {line}" for line in rendered)
    return f"{_DETECT_BLOCK_HEADER}\n{lines}"


def build_tool_for_signals(base_tool: types.Tool, signals: list[DetectSignal]) -> types.Tool:
    """Return *base_tool* augmented for an incidental-candidate advisory, else itself.

    An incidental-candidate advisory adds the optional ``pharmacological_role_check`` field to each
    ligand item. (``site_ref`` is a permanent base-schema field for every ligand,
    so it is not injected here; the dual-role advisory only adds prompt evidence.)
    With no incidental-candidate signal the base tool is returned by identity, guaranteeing
    zero schema perturbation. The base tool is never mutated (deep copy first).
    """
    has_incidental = any(
        s.kind == SIGNAL_INCIDENTAL_CANDIDATE and s.severity == SEVERITY_ADVISORY for s in signals
    )
    if not has_incidental:
        return base_tool
    declarations = base_tool.function_declarations or []
    if not declarations:
        return base_tool
    tool = base_tool.model_copy(deep=True)
    params = (tool.function_declarations or [])[0].parameters
    ligands = (params.properties or {}).get("ligands") if params else None
    items = ligands.items if ligands is not None else None
    if items is None or items.properties is None:
        return base_tool
    # Guard against a future SDK making deep model_copy shallow: mutating a nested
    # dict still shared with the base would corrupt every subsequent structure.
    base_decls = base_tool.function_declarations or []
    base_params = base_decls[0].parameters if base_decls else None
    base_ligands = (base_params.properties or {}).get("ligands") if base_params else None
    base_items = base_ligands.items if base_ligands is not None else None
    if base_items is not None and items.properties is base_items.properties:
        raise RuntimeError(
            "Tool.model_copy(deep=True) did not deep-copy nested Schema properties; "
            "refusing to mutate the shared base tool (check the google-genai version)."
        )
    items.properties["pharmacological_role_check"] = PHARMACOLOGICAL_ROLE_CHECK_SCHEMA
    return tool


def build_tool_config(signals: list[DetectSignal]) -> types.GenerateContentConfig:
    """Return the generation config for *signals* (identity ``TOOL_CONFIG`` if no mutation)."""
    tool = build_tool_for_signals(ANNOTATION_TOOL, signals)
    if tool is ANNOTATION_TOOL:
        return TOOL_CONFIG
    config = TOOL_CONFIG.model_copy(deep=True)
    config.tools = [tool]
    return config
