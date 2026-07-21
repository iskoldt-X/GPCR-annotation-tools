"""Pure orchestration that routes detect signals into the annotation prompt/tool.

ADVISORY detect signals become evidence in the prompt (the model weighs them
against the paper); REVIEW signals are not handled here -- they route silently to
human review. An incidental-candidate advisory additionally augments the tool schema
with an optional ``pharmacological_role_check`` field, and a dual-role advisory with an
optional ``site_ref`` field. No I/O, no AI calls.

When there are no advisory signals the prompt block is ``None`` and the tool /
config are returned by identity, so an ordinary structure is byte-for-byte
unchanged.

The model-facing wording of the evidence block is locked by a snapshot test.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google.genai import types

from gpcr_tools.annotator.schema import (
    ANNOTATION_TOOL,
    PHARMACOLOGICAL_ROLE_CHECK_SCHEMA,
    TOOL_CONFIG,
    build_ligand_copies_schema,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_CLASS_C_MULTI_PROTOMER,
    SIGNAL_COUPLING_PROTOMER,
    SIGNAL_DUAL_ROLE_LIGAND,
    SIGNAL_INCIDENTAL_CANDIDATE,
    SIGNAL_SITE_REF,
    SIGNAL_TRANSDUCER_COPY,
    DetectSignal,
)
from gpcr_tools.detector.site_ref import _annotated_ligands
from gpcr_tools.validator.oligomer import build_nonpolymer_instance_index

logger = logging.getLogger(__name__)

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
        SIGNAL_CLASS_C_MULTI_PROTOMER,
        SIGNAL_COUPLING_PROTOMER,
        SIGNAL_INCIDENTAL_CANDIDATE,
        SIGNAL_TRANSDUCER_COPY,
        SIGNAL_DUAL_ROLE_LIGAND,
        SIGNAL_SITE_REF,
    }
)

# The Class C multi-protomer advisory text is owner-locked: a single, general
# statement of the structural fact (Class C + more than one GPCR protomer), so the
# model treats both protomers as receptors. Rendered verbatim; do not embellish.
_CLASS_C_MULTI_PROTOMER_ADVISORY = (
    "This is a Class C receptor structure with more than one GPCR protomer."
)

# Header for the model-facing detector-evidence block (wording locked by a snapshot test).
_DETECT_BLOCK_HEADER = (
    "=== DETECTOR EVIDENCE (computed before annotation) ===\n"
    "Treat each item below as evidence to weigh against the paper, not as a "
    "settled conclusion:"
)


def _format_signal(signal: DetectSignal) -> str | None:
    """Render one advisory signal as a prompt evidence line.

    Returns ``None`` for any kind without a reviewed model-facing formatter, so
    an unreviewed summary (e.g. a future detector kind's note) is never leaked
    into the prompt. Only kinds in ``_MODEL_FACING_KINDS`` produce a line.
    """
    if signal.kind not in _MODEL_FACING_KINDS:
        return None
    payload = signal.payload or {}
    if signal.kind == SIGNAL_CLASS_C_MULTI_PROTOMER:
        # Owner-locked verbatim one-liner: a simple, general statement of the fact.
        return _CLASS_C_MULTI_PROTOMER_ADVISORY
    if signal.kind == SIGNAL_CHIMERIC_GPROTEIN:
        tail = payload.get("a5_tail") or "?"
        family = payload.get("family") or "?"
        subtype = payload.get("subtype") or "an indistinguishable subtype"
        return (
            f"G protein alpha5 analysis: the modelled alpha5 tail '{tail}' matches the "
            f"{family} family (subtype {subtype}). Weigh this against the paper before "
            f"assigning the G-alpha identity."
        )
    if signal.kind == SIGNAL_COUPLING_PROTOMER:
        chain = payload.get("coupling_chain") or "?"
        slug = payload.get("coupling_slug") or "?"
        return (
            f"Structure geometry shows the G protein engages receptor chain {chain} "
            f"({slug}); that protomer is the active, G protein-coupling one — in a "
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
    if signal.kind == SIGNAL_TRANSDUCER_COPY:
        listed = ", ".join(str(c) for c in (payload.get("copies") or [])) or "?"
        return (
            f"The following copies sit on a G protein / transducer chain, not the "
            f"receptor -- the transducer's own nucleotide / cofactor, not receptor "
            f"ligands: {listed}. Set that copy's role = Cofactor, unless the paper "
            f"specifically shows a functional role at THIS receptor."
        )
    if signal.kind == SIGNAL_DUAL_ROLE_LIGAND:
        return _format_dual_role(payload)
    if signal.kind == SIGNAL_SITE_REF:
        return _format_site_ref(payload)
    # A whitelisted kind with no branch above (should not happen): never leak a
    # raw summary -- the _MODEL_FACING_KINDS guard and this fall-through agree.
    return None


def _format_dual_role(payload: dict[str, Any]) -> str:
    """Render the dual-role signal as burial evidence, one line per buried copy.

    This provides geometric evidence that the ligand sits in more than one pocket
    (so it may play more than one role and may be more than one binding site). The
    model decides whether to split into one entry per site from the per-copy facts
    (the site_ref signal carries each copy's contact/segment facts); this dual-role
    block adds the buried-pocket count + residue context.
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
        f"(geometry below), so weigh whether it plays more than one role and whether each "
        f"pocket is a distinct binding site:\n{body}"
    )


def _format_site_ref(payload: dict[str, Any]) -> str | None:
    """Render the site_ref signal as per-copy geometry FACTS (no site verdict).

    The model infers site_ref from these facts plus the paper; distinct per-copy
    facts let it decide whether a ligand modelled at more than one site needs one
    entry per site.
    """
    comp = payload.get("comp_id") or "?"
    copies = payload.get("copies") or []
    if not copies:
        return None
    copy_lines = []
    for copy in copies:
        generic = ", ".join(copy.get("generic_numbers") or []) or "none mapped"
        segments = ", ".join(copy.get("segments") or []) or "?"
        core = copy.get("core_hits") or 0
        facing = copy.get("facing")
        facing_txt = (
            f"{facing:.2f} pocket-facing (1=buried in pocket, 0=lipid-facing)"
            if facing is not None
            else "facing n/a"
        )
        depth = copy.get("depth")
        if depth is not None:
            band = (
                "within the membrane band"
                if copy.get("in_band")
                else f"outside the membrane band (depth {depth} Å)"
            )
            # Append the oriented side fact when the structure could be oriented;
            # the outside-band wording keeps its signed depth number, and
            # unoriented copies keep the old no-side wording.
            side = copy.get("side")
            if side is not None:
                band = f"{band}, {side}"
        else:
            band = "membrane depth n/a"
        copy_id = copy.get("copy_id")
        label = f"copy {copy_id}" if copy_id else "a copy"
        copy_lines.append(
            f"  {label}: enclosure {copy.get('enclosure')}; contacts generic numbers "
            f"[{generic}] in segments [{segments}] ({core} Class A orthosteric-core); "
            f"{facing_txt}; {band}"
        )
    body = "\n".join(copy_lines)
    return (
        f"{comp}: geometry facts per modelled copy below — infer site_ref from these plus "
        f"the paper, use 'unknown' if neither settles it; if copies sit at distinct sites, "
        f"emit one entry per site:\n{body}"
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


# Header + fill-in instruction for the per-copy ligand roster block. Domain
# language only, kept in the detector-evidence voice.
_LIGAND_COPY_BLOCK_HEADER = (
    "=== LIGAND COPIES (assign a site and role to every copy) ===\n"
    "Each modelled copy of a candidate ligand is listed below by its copy "
    "identifier (author chain:residue) and component id, with its geometry facts:"
)

# A copy the geometry channel dropped as too sparse to map still gets a line:
# sparse contact is itself a clue that the copy is surface / membrane-facing.
_SPARSE_COPY_NOTE = (
    "few receptor contacts / surface-exposed (often a structural or "
    "membrane-facing copy) -- judge from the paper"
)

_LIGAND_COPY_INSTRUCTION = (
    "Fill the ligand_copies array with exactly one entry per copy listed above, "
    "reusing its copy_id verbatim. Assign each copy's site_ref and role from that "
    "copy's own geometry facts (keyed by the same identifier, both here and in the "
    "DETECTOR EVIDENCE block) plus the paper. Answers may repeat across copies -- "
    "many structural-lipid copies all at 'membrane_facing' is expected, so do not "
    "invent distinct sites to force them apart. Use 'unknown' when a copy's position "
    "is genuinely undetermined rather than guessing. Fill exactly the copies listed: "
    "do not add, omit, or alter any identifier."
)


def _entry(enriched_data: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the ``data.entry`` envelope if present, else use the object as-is."""
    return (enriched_data.get("data") or {}).get("entry") or enriched_data


def ligand_copy_identifiers(enriched_data: dict[str, Any]) -> list[tuple[str, str]]:
    """Ordered ``(comp_id, copy_id)`` for every functional-candidate ligand copy.

    The exam roster: every modelled copy -- from ``build_nonpolymer_instance_index``,
    the true physical copy set in the RCSB metadata -- of every functional-candidate
    component (``_annotated_ligands``: present non-polymers minus stripped buffers,
    with incidental membrane lipids such as cholesterol kept on the roster). Each
    ``copy_id`` is "<auth_asym_id>:<auth_seq_id>". Accepts the enriched envelope or a
    bare entry, and is empty when the structure has no functional-candidate copies.
    """
    entry = _entry(enriched_data)
    index = build_nonpolymer_instance_index(entry)
    candidates = _annotated_ligands(entry)
    roster: list[tuple[str, str]] = []
    # Dedup key is copy_id ALONE (not (comp_id, copy_id)): copy_id is the schema
    # enum value the model binds each answer to, so a repeated value would be
    # ambiguous. Track which comp_id first claimed each copy_id so a genuine
    # cross-component collision can be surfaced rather than silently dropped.
    claimed_by: dict[str, str] = {}
    for comp_id in sorted(candidates):
        for inst in index.get(comp_id, []):
            auth_asym = inst.get("auth_asym_id") or ""
            auth_seq = inst.get("auth_seq_id") or ""
            if not auth_asym or not auth_seq:
                # A copy with no usable author chain/residue cannot form an
                # identifier the model can bind an answer to -- skip it.
                continue
            copy_id = f"{auth_asym}:{auth_seq}"
            existing = claimed_by.get(copy_id)
            if existing is not None:
                # Two DIFFERENT components sharing one author chain:residue is a
                # rare collision; keep the first and warn rather than drop silently.
                if existing != comp_id:
                    logger.warning(
                        "[%s] ligand copy identifier %s is shared by components %s and %s; "
                        "keeping %s and dropping %s (copy_id must be a unique roster key).",
                        entry.get("rcsb_id") or "UNKNOWN",
                        copy_id,
                        existing,
                        comp_id,
                        existing,
                        comp_id,
                    )
                continue
            claimed_by[copy_id] = comp_id
            roster.append((comp_id, copy_id))
    return roster


def ligand_copy_id_enum(copy_roster: list[tuple[str, str]]) -> list[str]:
    """The ``copy_id`` enum (order-preserving, already unique) from a copy roster."""
    return [copy_id for _comp_id, copy_id in copy_roster]


@dataclass(frozen=True)
class LigandCopyCoverage:
    """Whether a run's ``ligand_copies`` covers the expected copy roster exactly.

    ``missing`` / ``duplicated`` are expected copy identifiers absent from, or
    repeated in, the returned rows; ``unexpected`` are returned identifiers
    outside the expected set. ``ok`` is True only when all three are empty --
    exact coverage: every expected ``copy_id`` present exactly once, nothing extra.
    """

    missing: tuple[str, ...] = ()
    duplicated: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicated or self.unexpected)

    def describe(self) -> str:
        """One-line human summary for a log message (``exact coverage`` when ok)."""
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing {list(self.missing)}")
        if self.duplicated:
            parts.append(f"duplicated {list(self.duplicated)}")
        if self.unexpected:
            parts.append(f"unexpected {list(self.unexpected)}")
        return "; ".join(parts) if parts else "exact coverage"


def check_ligand_copy_coverage(
    ligand_copies: Any, expected_copy_ids: Sequence[str]
) -> LigandCopyCoverage:
    """Check that a run's ``ligand_copies`` covers *expected_copy_ids* exactly.

    Exact coverage = every expected identifier present exactly once, with no
    duplicate and no identifier outside the expected set. The per-PDB schema enum
    already constrains ``copy_id`` values, but a returned response is validated
    defensively here -- deterministic post-processing, no AI schema/prompt change.

    An empty *expected_copy_ids* (a structure with no candidate copies) always
    reports exact coverage: there is nothing to validate, so this is a no-op.
    Tolerant of a non-list *ligand_copies* or rows without a string ``copy_id`` --
    such rows contribute no coverage, so a malformed response surfaces as missing
    copies rather than raising.
    """
    expected = list(expected_copy_ids)
    if not expected:
        return LigandCopyCoverage()
    counts: Counter[str] = Counter()
    if isinstance(ligand_copies, list):
        for row in ligand_copies:
            if isinstance(row, dict):
                copy_id = row.get("copy_id")
                if isinstance(copy_id, str):
                    counts[copy_id] += 1
    expected_set = set(expected)
    missing = tuple(cid for cid in expected if counts[cid] == 0)
    duplicated = tuple(cid for cid in expected if counts[cid] > 1)
    unexpected = tuple(sorted(cid for cid in counts if cid not in expected_set))
    return LigandCopyCoverage(missing=missing, duplicated=duplicated, unexpected=unexpected)


def _geometry_by_copy_id(signals: list[DetectSignal]) -> dict[str, dict[str, Any]]:
    """Index each site_ref advisory copy's geometry facts by its copy identifier."""
    by_copy: dict[str, dict[str, Any]] = {}
    for signal in signals:
        if signal.kind != SIGNAL_SITE_REF or signal.severity != SEVERITY_ADVISORY:
            continue
        for copy in (signal.payload or {}).get("copies") or []:
            copy_id = copy.get("copy_id")
            if copy_id:
                by_copy[copy_id] = copy
    return by_copy


def _format_copy_facts(copy: dict[str, Any]) -> str:
    """Compact one-line geometry summary for a single ligand copy."""
    generic = ", ".join(copy.get("generic_numbers") or []) or "none mapped"
    segments = ", ".join(copy.get("segments") or []) or "?"
    core = copy.get("core_hits") or 0
    parts = [
        f"contacts generic numbers [{generic}] in segments [{segments}] "
        f"({core} Class A orthosteric-core)"
    ]
    enclosure = copy.get("enclosure")
    if enclosure is not None:
        parts.append(f"enclosure {enclosure}")
    facing = copy.get("facing")
    if facing is not None:
        parts.append(f"{facing:.2f} pocket-facing")
    side = copy.get("side")
    if side is not None:
        parts.append(str(side))
    return "; ".join(parts)


def assemble_ligand_copy_block(
    copy_roster: list[tuple[str, str]],
    signals: list[DetectSignal],
) -> str | None:
    """Build the LIGAND COPIES prompt block, or ``None`` when the roster is empty.

    *copy_roster* is the ordered ``(comp_id, copy_id)`` list of every
    functional-candidate ligand copy (from RCSB metadata, so it includes copies
    whose geometry was too sparse to map). Every listed copy gets one line carrying
    its identifier, component id, and geometry facts looked up from the site_ref
    signals by ``copy_id`` -- or a sparse-contact note when no geometry mapped. The
    block then instructs the model to assign a site/role/confidence to every listed
    copy. Enumeration comes from the metadata copy set, not the geometry channel, so
    a sparse copy dropped from DETECTOR EVIDENCE is still represented here.
    """
    if not copy_roster:
        return None
    geometry = _geometry_by_copy_id(signals)
    lines: list[str] = []
    for comp_id, copy_id in copy_roster:
        facts = geometry.get(copy_id)
        summary = _format_copy_facts(facts) if facts else _SPARSE_COPY_NOTE
        lines.append(f"  copy {copy_id} ({comp_id}): {summary}")
    body = "\n".join(lines)
    return f"{_LIGAND_COPY_BLOCK_HEADER}\n{body}\n{_LIGAND_COPY_INSTRUCTION}"


def build_tool_for_signals(
    base_tool: types.Tool,
    signals: list[DetectSignal],
    ligand_copy_ids: list[str] | None = None,
) -> types.Tool:
    """Return *base_tool* augmented for this structure, or *base_tool* itself.

    Two independent, additive augmentations, both applied to a single deep copy so
    the base tool is never mutated:

    * an incidental-candidate advisory adds the optional
      ``pharmacological_role_check`` field to each ligand item; and
    * *ligand_copy_ids* (this structure's ligand copy identifiers) adds the new
      top-level ``ligand_copies`` array, whose ``copy_id`` enum is pinned to exactly
      those identifiers. The original ``ligands`` array is left untouched -- the
      per-copy array is a purely additive sidecar.

    With neither present the base tool is returned by identity, guaranteeing zero
    schema perturbation for an ordinary structure. (``site_ref`` is a permanent
    base-schema field for every ligand, so it is not injected here; the dual-role
    advisory only adds prompt evidence.)
    """
    has_incidental = any(
        s.kind == SIGNAL_INCIDENTAL_CANDIDATE and s.severity == SEVERITY_ADVISORY for s in signals
    )
    copy_ids = list(ligand_copy_ids or [])
    if not has_incidental and not copy_ids:
        return base_tool
    declarations = base_tool.function_declarations or []
    if not declarations:
        return base_tool
    tool = base_tool.model_copy(deep=True)
    params = (tool.function_declarations or [])[0].parameters
    if params is None or params.properties is None:
        return base_tool

    base_decls = base_tool.function_declarations or []
    base_params = base_decls[0].parameters if base_decls else None
    base_props = base_params.properties if base_params is not None else None

    if has_incidental:
        ligands = params.properties.get("ligands")
        items = ligands.items if ligands is not None else None
        if items is None or items.properties is None:
            return base_tool
        # Guard against a future SDK making deep model_copy shallow: mutating a
        # nested dict still shared with the base would corrupt every subsequent
        # structure.
        base_ligands = (base_props or {}).get("ligands") if base_props else None
        base_items = base_ligands.items if base_ligands is not None else None
        if base_items is not None and items.properties is base_items.properties:
            raise RuntimeError(
                "Tool.model_copy(deep=True) did not deep-copy nested Schema properties; "
                "refusing to mutate the shared base tool (check the google-genai version)."
            )
        items.properties["pharmacological_role_check"] = PHARMACOLOGICAL_ROLE_CHECK_SCHEMA

    if copy_ids:
        # Same deep-copy guard for the top-level properties dict we extend: a
        # shallow copy would still share it with the base tool.
        if base_props is not None and params.properties is base_props:
            raise RuntimeError(
                "Tool.model_copy(deep=True) did not deep-copy nested Schema properties; "
                "refusing to mutate the shared base tool (check the google-genai version)."
            )
        params.properties["ligand_copies"] = build_ligand_copies_schema(copy_ids)

    return tool


def build_tool_config(
    signals: list[DetectSignal],
    temperature: float | None = None,
    thinking_level: str | None = None,
    ligand_copy_ids: list[str] | None = None,
) -> types.GenerateContentConfig:
    """Return the generation config for *signals* (identity ``TOOL_CONFIG`` if no mutation).

    *temperature* sets the sampling temperature when given; ``None`` leaves it
    unset so the model's own default applies (``TOOL_CONFIG`` pins no temperature).
    *thinking_level* (one of ``minimal``/``low``/``medium``/``high``) sets the
    reasoning depth when given; ``None`` leaves it unset so the model's own
    default (``high``) applies. *ligand_copy_ids* (this structure's ligand copy
    identifiers) pins the per-PDB ``ligand_copies`` array; empty/``None`` leaves
    the schema unchanged.
    """
    tool = build_tool_for_signals(ANNOTATION_TOOL, signals, ligand_copy_ids=ligand_copy_ids)
    if tool is ANNOTATION_TOOL and temperature is None and thinking_level is None:
        return TOOL_CONFIG
    config = TOOL_CONFIG.model_copy(deep=True)
    config.tools = [tool]
    if temperature is not None:
        config.temperature = temperature
    if thinking_level is not None:
        config.thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(thinking_level.upper())
        )
    return config
