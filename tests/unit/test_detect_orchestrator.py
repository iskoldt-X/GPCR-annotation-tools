"""Tests for detect_orchestrator: prompt evidence block + tool/config routing.

Key invariants: advisory signals produce a block / augmented tool; review-only
or no signals leave the prompt block None and the tool/config returned by
identity (zero perturbation); the base ANNOTATION_TOOL / TOOL_CONFIG are never
mutated.
"""

from __future__ import annotations

import re

from google.genai import types

from gpcr_tools.annotator.detect_orchestrator import (
    assemble_detect_block,
    assemble_ligand_copy_block,
    build_tool_config,
    build_tool_for_signals,
    check_ligand_copy_coverage,
    ligand_copy_id_enum,
    ligand_copy_identifiers,
)
from gpcr_tools.annotator.schema import ANNOTATION_TOOL, TOOL_CONFIG
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_CLASS_C_MULTI_PROTOMER,
    SIGNAL_COUPLING_PROTOMER,
    SIGNAL_DUAL_ROLE_LIGAND,
    SIGNAL_INCIDENTAL_CANDIDATE,
    SIGNAL_SITE_REF,
    SIGNAL_TRANSDUCER_COPY,
    DetectSignal,
)


def _copy(
    generic: list[str],
    segments: list[str],
    core_hits: int,
    enclosure: float,
    facing: float | None = None,
    depth: float | None = None,
    in_band: bool | None = None,
    side: str | None = None,
    copy_id: str | None = None,
) -> dict:
    copy: dict = {
        "generic_numbers": generic,
        "segments": segments,
        "core_hits": core_hits,
        "enclosure": enclosure,
        "facing": facing,
    }
    if depth is not None:
        copy["depth"] = depth
        copy["in_band"] = in_band
    if side is not None:
        copy["side"] = side
    if copy_id is not None:
        copy["copy_id"] = copy_id
    return copy


def _site_ref(comp: str, copies: list[dict]) -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_SITE_REF,
        target_ref="ligands",
        summary=f"{comp} facts",
        payload={"comp_id": comp, "copies": copies},
        severity=SEVERITY_ADVISORY,
    )


def _chimeric_advisory() -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_CHIMERIC_GPROTEIN,
        target_ref="signaling_partners.g_protein.alpha_subunit",
        summary="alpha5 resolves to gnai1",
        payload={"family": "Gi/o", "subtype": "gnai1_human", "a5_tail": "IKENLKDCGLF", "score": 11},
        severity=SEVERITY_ADVISORY,
    )


def _incidental_candidate(comp: str = "PLM") -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_INCIDENTAL_CANDIDATE,
        target_ref="ligands",
        summary=f"{comp} incidental_candidate",
        payload={"comp_id": comp},
        severity=SEVERITY_ADVISORY,
    )


def _transducer_copy(copies: list[str] | None = None) -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_TRANSDUCER_COPY,
        target_ref="ligands",
        summary="transducer copies",
        payload={"copies": copies if copies is not None else ["GDP A:401"]},
        severity=SEVERITY_ADVISORY,
    )


def _dual_role(comp: str = "A1AEI") -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_DUAL_ROLE_LIGAND,
        target_ref="ligands",
        summary=f"{comp} in two pockets",
        payload={
            "comp_id": comp,
            "gpcr_chain": "R",
            "copies": [
                {
                    "chain": "R",
                    "seq_id": 601,
                    "burial": 0.99,
                    "n_pocket_residues": 17,
                    "pocket_residues": [104, 107, 108, 111, 194, 197, 198],
                    "contacts_partner": True,
                },
                {
                    "chain": "R",
                    "seq_id": 602,
                    "burial": 0.99,
                    "n_pocket_residues": 14,
                    "pocket_residues": [62, 65, 76, 82, 85],
                    "contacts_partner": False,
                },
            ],
        },
        severity=SEVERITY_ADVISORY,
    )


def _review() -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_CHIMERIC_GPROTEIN,
        target_ref="x",
        summary="indistinguishable subtype",
        payload={},
        severity=SEVERITY_REVIEW,
    )


class TestAssembleDetectBlock:
    def test_no_signals_returns_none(self) -> None:
        assert assemble_detect_block([]) is None

    def test_only_review_signals_returns_none(self) -> None:
        assert assemble_detect_block([_review()]) is None

    def test_chimeric_advisory_renders_alpha5_evidence(self) -> None:
        block = assemble_detect_block([_chimeric_advisory()])
        assert block is not None
        assert "IKENLKDCGLF" in block and "Gi/o" in block

    def test_incidental_candidate_renders_comp_and_guidance(self) -> None:
        block = assemble_detect_block([_incidental_candidate("PLM")])
        assert block is not None
        assert "PLM" in block and "pharmacological_role_check" in block

    def test_transducer_copy_renders_copies_and_cofactor_guidance(self) -> None:
        block = assemble_detect_block([_transducer_copy(["GDP A:401", "MG A:402"])])
        assert block is not None
        assert "GDP A:401" in block and "MG A:402" in block
        assert "transducer" in block.lower() and "Cofactor" in block

    def test_deterministic_order(self) -> None:
        a = assemble_detect_block([_incidental_candidate("PLM"), _chimeric_advisory()])
        b = assemble_detect_block([_chimeric_advisory(), _incidental_candidate("PLM")])
        assert a == b

    def test_site_ref_single_copy_facts(self) -> None:
        block = assemble_detect_block(
            [_site_ref("ADN", [_copy(["3x33", "6x51"], ["TM3", "TM6"], 2, 0.88, facing=0.9)])]
        )
        assert block is not None
        # Facts are rendered; no site verdict / "places it at the X site" label.
        assert "ADN" in block and "3x33" in block and "geometry facts" in block
        assert "places it at" not in block

    def test_site_ref_multi_copy_facts(self) -> None:
        block = assemble_detect_block(
            [
                _site_ref(
                    "A1AEI",
                    [
                        _copy(["3x33", "6x51"], ["TM3", "TM6"], 2, 0.95, facing=0.9),
                        _copy(["45x52"], ["ECL2"], 0, 0.6, facing=0.2),
                    ],
                )
            ]
        )
        assert block is not None
        # Both copies' facts present; the conditional per-site split instruction shown.
        assert "3x33" in block and "ECL2" in block
        assert "one entry per site" in block

    def test_site_ref_copy_line_labelled_with_copy_id(self) -> None:
        # A copy carrying its copy identifier labels its evidence line with it, so the
        # model can bind its answer to that physical copy.
        block = assemble_detect_block(
            [_site_ref("ADN", [_copy(["3x33"], ["TM3"], 1, 0.9, copy_id="R:602")])]
        )
        assert block is not None
        assert "copy R:602: enclosure 0.9;" in block
        assert "a copy:" not in block

    def test_site_ref_copy_line_falls_back_when_no_copy_id(self) -> None:
        # Without a copy identifier the line keeps the neutral "a copy" wording --
        # defensive rendering for a copy lacking an identifier, not a metadata-join
        # failure (identity is read from coordinates, so this path is unreachable today).
        block = assemble_detect_block([_site_ref("ADN", [_copy(["3x33"], ["TM3"], 1, 0.9)])])
        assert block is not None
        assert "  a copy: enclosure 0.9;" in block

    def test_split_instruction_owned_by_site_ref_not_dual_role(self) -> None:
        # The dual-role signal gives burial evidence but must NOT command a split;
        # only the site_ref facts carry the "one entry per site" instruction.
        dual = assemble_detect_block([_dual_role("A1AEI")])
        assert dual is not None
        assert "more than one role" in dual
        assert "entry per" not in dual  # no split command from dual-role
        site = assemble_detect_block(
            [
                _site_ref(
                    "A1AEI", [_copy(["3x33"], ["TM3"], 1, 0.95), _copy(["45x52"], ["ECL2"], 0, 0.6)]
                )
            ]
        )
        assert "one entry per site" in site

    def test_site_ref_renders_intracellular_side(self) -> None:
        # An oriented copy outside the band on the cytoplasmic side appends the
        # qualitative side fact while keeping the signed depth number.
        block = assemble_detect_block(
            [
                _site_ref(
                    "GTP",
                    [
                        _copy(
                            ["3x50", "7x53"],
                            ["H8", "ICL3"],
                            0,
                            0.7,
                            depth=-24.0,
                            in_band=False,
                            side="on the intracellular side",
                        )
                    ],
                )
            ]
        )
        assert block is not None
        assert "outside the membrane band (depth -24.0 Å)" in block  # depth number kept
        assert "on the intracellular side" in block

    def test_site_ref_renders_mid_membrane_side(self) -> None:
        # A mid-bilayer inter-helical copy inside the band is reported mid-membrane.
        block = assemble_detect_block(
            [
                _site_ref(
                    "OLA",
                    [
                        _copy(
                            ["3x40", "4x56", "5x46"],
                            ["TM3", "TM4", "TM5"],
                            0,
                            0.8,
                            depth=1.0,
                            in_band=True,
                            side="mid-membrane",
                        )
                    ],
                )
            ]
        )
        assert block is not None
        assert "within the membrane band, mid-membrane" in block

    def test_site_ref_unoriented_copy_keeps_old_wording(self) -> None:
        # When the structure could not be oriented, no side fact is added: the copy
        # keeps the existing no-side band wording (honest abstain).
        block = assemble_detect_block(
            [_site_ref("ADN", [_copy(["3x33"], ["TM3"], 1, 0.85, depth=3.0, in_band=True)])]
        )
        assert block is not None
        assert "within the membrane band" in block
        for side in ("intracellular side", "extracellular side", "mid-membrane"):
            assert side not in block

    def test_dual_role_renders_per_copy_pocket_evidence(self) -> None:
        block = assemble_detect_block([_dual_role("A1AEI")])
        assert block is not None
        assert "A1AEI" in block and "distinct binding site" in block
        # one line per buried copy, with its enclosure and pocket residues
        assert "R/601" in block and "R/602" in block
        assert "104" in block  # a pocket residue of the first copy
        # the partner-contacting copy is flagged as the possible active-state pocket
        assert "active-state pocket" in block


def _coupling_advisory() -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_COUPLING_PROTOMER,
        target_ref="receptor_info",
        summary="curator-facing summary (not what reaches the prompt)",
        payload={"coupling_chain": "B", "coupling_slug": "gabbr2_human"},
        severity=SEVERITY_ADVISORY,
    )


def _class_c_multi_protomer() -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_CLASS_C_MULTI_PROTOMER,
        target_ref="receptor_info",
        summary="Class C receptor structure with more than one GPCR protomer.",
        payload={"gpcr_chains": ["A", "B"], "accessions": ["O75899", "P47869"]},
        severity=SEVERITY_ADVISORY,
    )


class TestClassCMultiProtomerEvidence:
    """The Class C multi-protomer advisory renders the owner-locked verbatim
    one-liner exactly, with no chain names or embellishment."""

    _VERBATIM = "This is a Class C receptor structure with more than one GPCR protomer."

    def test_renders_verbatim_one_liner(self) -> None:
        block = assemble_detect_block([_class_c_multi_protomer()])
        assert block is not None
        assert self._VERBATIM in block

    def test_no_chain_names_or_embellishment(self) -> None:
        block = assemble_detect_block([_class_c_multi_protomer()])
        assert block is not None
        # The advisory line must be exactly the verbatim text (no chain ids, no
        # "weigh against the paper" tail, no accession leak).
        line = next(line for line in block.splitlines() if line.startswith("- "))
        assert line == f"- {self._VERBATIM}"


class TestNoUnreviewedLeakIntoPrompt:
    """Only kinds with a reviewed model-facing formatter reach the prompt; a kind
    without one is dropped, never leaked verbatim as evidence."""

    def _unformatted(self) -> DetectSignal:
        return DetectSignal(
            kind="some_future_kind",
            target_ref="x",
            summary="raw internal summary that must not reach the model",
            payload={},
            severity=SEVERITY_ADVISORY,
        )

    def test_unformatted_kind_alone_yields_no_block(self) -> None:
        assert assemble_detect_block([self._unformatted()]) is None

    def test_unformatted_kind_summary_absent_when_mixed_with_real_signal(self) -> None:
        block = assemble_detect_block([self._unformatted(), _incidental_candidate("PLM")])
        assert block is not None
        assert "PLM" in block
        assert "raw internal summary" not in block


class TestCouplingProtomerEvidence:
    """The coupling-protomer signal renders via its own reviewed formatter (it is
    now model-facing); the curator-facing summary is not what reaches the prompt."""

    def test_coupling_renders_chain_and_slug(self) -> None:
        block = assemble_detect_block([_coupling_advisory()])
        assert block is not None
        assert "chain B" in block
        assert "gabbr2_human" in block
        assert "G protein-coupling" in block

    def test_coupling_does_not_leak_raw_summary(self) -> None:
        block = assemble_detect_block([_coupling_advisory()])
        assert block is not None
        assert "curator-facing summary" not in block


def test_no_disputed_phrasing_in_any_ai_facing_string() -> None:
    # The 'disputed molecule' wording is retired; the field is pharmacological_role_check.
    from gpcr_tools.annotator.schema import PHARMACOLOGICAL_ROLE_CHECK_SCHEMA
    from gpcr_tools.detector.ligands import detect_incidental_candidates

    block = assemble_detect_block([_incidental_candidate("CLR")])
    assert block is not None and "disputed" not in block.lower()
    assert "disputed" not in (PHARMACOLOGICAL_ROLE_CHECK_SCHEMA.description or "").lower()
    entry = {"nonpolymer_entities": [{"nonpolymer_comp": {"chem_comp": {"id": "CLR"}}}]}
    sigs = detect_incidental_candidates("X", entry)
    assert sigs and all("disputed" not in s.summary.lower() for s in sigs)


def test_detect_block_golden_snapshot() -> None:
    # Locks the exact model-facing wording of every formatter + the deterministic
    # ordering (by kind). Any accidental wording change fails loudly here.
    chimeric = DetectSignal(
        kind=SIGNAL_CHIMERIC_GPROTEIN,
        target_ref="signaling_partners.g_protein.alpha_subunit",
        summary="x",
        payload={"family": "Gi/o", "subtype": "gnai1_human", "a5_tail": "IKENLKDCGLF", "score": 11},
        severity=SEVERITY_ADVISORY,
    )
    block = assemble_detect_block(
        [
            chimeric,
            _class_c_multi_protomer(),
            _coupling_advisory(),
            _incidental_candidate("CLR"),
            _site_ref(
                "ADN",
                [
                    _copy(
                        ["3x33", "6x51"],
                        ["TM3", "TM6"],
                        2,
                        0.88,
                        facing=0.9,
                        depth=2.0,
                        in_band=True,
                        side="mid-membrane",
                    )
                ],
            ),
        ]
    )
    expected = (
        "=== DETECTOR EVIDENCE (computed before annotation) ===\n"
        "Treat each item below as evidence to weigh against the paper, not as a "
        "settled conclusion:\n"
        "- G protein alpha5 analysis: the modelled alpha5 tail 'IKENLKDCGLF' matches "
        "the Gi/o family (subtype gnai1_human). Weigh this against the paper before "
        "assigning the G-alpha identity.\n"
        "- This is a Class C receptor structure with more than one GPCR protomer.\n"
        "- Structure geometry shows the G protein engages receptor chain B "
        "(gabbr2_human); that protomer is the active, G protein-coupling one — in a "
        "heterodimer not necessarily the agonist-binding protomer. Weigh this against "
        "the paper.\n"
        "- CLR is present; it can be a functional ligand in some structures and an "
        "incidental structural component in others. Judge its role from the paper and "
        "record a pharmacological_role_check.\n"
        "- ADN: geometry facts per modelled copy below — infer site_ref from these "
        "plus the paper, use 'unknown' if neither settles it; if copies sit at distinct "
        "sites, emit one entry per site:\n"
        "  a copy: enclosure 0.88; contacts generic numbers [3x33, 6x51] in segments "
        "[TM3, TM6] (2 Class A orthosteric-core); 0.90 pocket-facing (1=buried in "
        "pocket, 0=lipid-facing); within the membrane band, mid-membrane"
    )
    assert block == expected


class TestBuildToolForSignals:
    def test_no_incidental_candidate_returns_base_identity(self) -> None:
        assert build_tool_for_signals(ANNOTATION_TOOL, []) is ANNOTATION_TOOL
        assert build_tool_for_signals(ANNOTATION_TOOL, [_chimeric_advisory()]) is ANNOTATION_TOOL

    def test_incidental_candidate_adds_field_without_mutating_base(self) -> None:
        tool = build_tool_for_signals(ANNOTATION_TOOL, [_incidental_candidate()])
        assert tool is not ANNOTATION_TOOL
        items = tool.function_declarations[0].parameters.properties["ligands"].items
        assert "pharmacological_role_check" in items.properties
        # The base tool must be untouched (no schema leak).
        base_items = ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items
        assert "pharmacological_role_check" not in base_items.properties

    def test_site_ref_is_in_base_schema(self) -> None:
        # site_ref is a permanent field on every ligand, not injected per-signal.
        base_items = ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items
        assert "site_ref" in base_items.properties

    def test_dual_role_alone_does_not_mutate_schema(self) -> None:
        # A dual-role advisory only adds prompt evidence; site_ref is already in
        # the base schema, so the tool is returned by identity.
        assert build_tool_for_signals(ANNOTATION_TOOL, [_dual_role()]) is ANNOTATION_TOOL

    def test_incidental_candidate_adds_assessment_with_site_ref_already_present(self) -> None:
        tool = build_tool_for_signals(ANNOTATION_TOOL, [_incidental_candidate(), _dual_role()])
        items = tool.function_declarations[0].parameters.properties["ligands"].items
        assert "pharmacological_role_check" in items.properties
        assert "site_ref" in items.properties  # inherited from the base schema


class TestBuildToolConfig:
    def test_no_incidental_candidate_returns_base_config_identity(self) -> None:
        assert build_tool_config([]) is TOOL_CONFIG
        assert build_tool_config([_chimeric_advisory()]) is TOOL_CONFIG

    def test_dual_role_alone_returns_base_config_identity(self) -> None:
        # Dual-role no longer mutates the schema, so the config is unchanged.
        assert build_tool_config([_dual_role()]) is TOOL_CONFIG

    def test_incidental_candidate_returns_new_config_leaving_base_unchanged(self) -> None:
        cfg = build_tool_config([_incidental_candidate()])
        assert cfg is not TOOL_CONFIG
        assert cfg.tools[0] is not ANNOTATION_TOOL
        assert TOOL_CONFIG.tools[0] is ANNOTATION_TOOL  # base config untouched

    def test_thinking_level_unset_leaves_base_config_identity(self) -> None:
        # No signal mutation and no thinking level -> the pinned base config, so
        # the model's own default reasoning depth applies (no override sent).
        assert build_tool_config([]) is TOOL_CONFIG
        assert build_tool_config([], thinking_level=None) is TOOL_CONFIG

    def test_thinking_level_sets_thinking_config_leaving_base_unchanged(self) -> None:
        cfg = build_tool_config([], thinking_level="low")
        assert cfg is not TOOL_CONFIG
        assert cfg.thinking_config is not None
        assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
        assert TOOL_CONFIG.thinking_config is None  # base config untouched


# ---------------------------------------------------------------------------
# Per-PDB ligand_copies schema + prompt roster (per-copy fill-in)
# ---------------------------------------------------------------------------


def _np_entity(comp_id: str, copies: list[tuple[str, str, str]]) -> dict:
    """A nonpolymer entity with *copies* as (auth_asym_id, label_asym_id, auth_seq_id)."""
    return {
        "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": comp_id},
        "nonpolymer_entity_instances": [
            {
                "rcsb_nonpolymer_entity_instance_container_identifiers": {
                    "auth_asym_id": auth,
                    "asym_id": label,
                    "auth_seq_id": seq,
                }
            }
            for (auth, label, seq) in copies
        ],
    }


def _multi_ligand_entry() -> dict:
    """A structure with a drug-like ligand, incidental lipids (CLR/PLM), and a
    stripped buffer (SO4), so the candidate filter and copy-identifier formation
    are both exercised."""
    return {
        "nonpolymer_entities": [
            _np_entity("J40", [("R", "A", "601")]),
            _np_entity(
                "CLR",
                [("R", "B", "602"), ("R", "C", "603"), ("R", "D", "604"), ("R", "E", "605")],
            ),
            _np_entity("PLM", [("R", "F", "701"), ("R", "G", "702"), ("R", "H", "703")]),
            _np_entity("SO4", [("R", "I", "801")]),
        ]
    }


def _base_ligand_item_props() -> dict:
    return (
        ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items.properties
    )


class TestLigandCopyIdentifiers:
    def test_lists_every_candidate_copy_and_strips_buffers(self) -> None:
        roster = ligand_copy_identifiers(_multi_ligand_entry())
        copy_ids = ligand_copy_id_enum(roster)
        # Drug-like + both incidental lipids are on the roster; the buffer is not.
        assert set(copy_ids) == {
            "R:601",
            "R:602",
            "R:603",
            "R:604",
            "R:605",
            "R:701",
            "R:702",
            "R:703",
        }
        assert "R:801" not in copy_ids  # SO4 buffer stripped
        assert len(copy_ids) == len(set(copy_ids))  # unique (valid enum)
        comps = {comp for comp, _cid in roster}
        assert comps == {"J40", "CLR", "PLM"}  # incidental lipids kept on the exam

    def test_incidental_lipid_copy_present(self) -> None:
        roster = ligand_copy_identifiers(_multi_ligand_entry())
        assert ("CLR", "R:602") in roster  # a cholesterol (incidental) copy is on the roster

    def test_accepts_enriched_envelope(self) -> None:
        bare = ligand_copy_identifiers(_multi_ligand_entry())
        wrapped = ligand_copy_identifiers({"data": {"entry": _multi_ligand_entry()}})
        assert bare == wrapped

    def test_no_candidates_yields_empty_roster(self) -> None:
        only_buffers = {"nonpolymer_entities": [_np_entity("SO4", [("R", "A", "801")])]}
        assert ligand_copy_identifiers(only_buffers) == []

    def test_skips_copy_without_usable_identifier(self) -> None:
        # A copy missing its author residue number cannot form an identifier.
        entry = {"nonpolymer_entities": [_np_entity("J40", [("R", "A", "")])]}
        assert ligand_copy_identifiers(entry) == []


class TestBuildToolForLigandCopies:
    def test_injects_ligand_copies_array_pinned_to_copy_ids(self) -> None:
        copy_ids = ["R:601", "R:602", "R:701"]
        tool = build_tool_for_signals(ANNOTATION_TOOL, [], ligand_copy_ids=copy_ids)
        assert tool is not ANNOTATION_TOOL
        params = tool.function_declarations[0].parameters
        arr = params.properties["ligand_copies"]
        assert arr.type == types.Type.ARRAY
        item = arr.items
        # copy_id enum is exactly this structure's identifiers (incl. an incidental lipid).
        assert set(item.properties["copy_id"].enum) == set(copy_ids)
        # Required fields; evidence is optional.
        assert set(item.required) == {"copy_id", "site_ref", "role", "confidence"}
        assert "evidence" in item.properties
        assert "evidence" not in item.required
        # No fixed-length pin (rejected by the API at higher counts).
        assert arr.min_items is None
        assert arr.max_items is None

    def test_reuses_base_site_ref_and_role_enums(self) -> None:
        tool = build_tool_for_signals(ANNOTATION_TOOL, [], ligand_copy_ids=["R:601"])
        item = tool.function_declarations[0].parameters.properties["ligand_copies"].items
        base = _base_ligand_item_props()
        assert set(item.properties["site_ref"].enum) == set(base["site_ref"].enum)
        assert set(item.properties["role"].enum) == set(base["role"].properties["value"].enum)
        assert set(item.properties["confidence"].enum) == {"High", "Medium", "Low"}

    def test_base_tool_and_ligands_array_untouched(self) -> None:
        build_tool_for_signals(ANNOTATION_TOOL, [], ligand_copy_ids=["R:601"])
        base_params = ANNOTATION_TOOL.function_declarations[0].parameters
        assert "ligand_copies" not in base_params.properties  # no leak onto the base tool
        # The existing compound-level ligands array is left exactly as it was.
        assert base_params.properties["ligands"].type == types.Type.ARRAY
        assert "site_ref" in base_params.properties["ligands"].items.properties

    def test_no_copies_and_no_incidental_returns_identity(self) -> None:
        assert build_tool_for_signals(ANNOTATION_TOOL, [], ligand_copy_ids=[]) is ANNOTATION_TOOL
        assert build_tool_for_signals(ANNOTATION_TOOL, [], ligand_copy_ids=None) is ANNOTATION_TOOL

    def test_coexists_with_incidental_role_check_without_mutating_base(self) -> None:
        tool = build_tool_for_signals(
            ANNOTATION_TOOL, [_incidental_candidate()], ligand_copy_ids=["R:601"]
        )
        params = tool.function_declarations[0].parameters
        assert "ligand_copies" in params.properties  # new top-level array
        assert "pharmacological_role_check" in params.properties["ligands"].items.properties
        # Both injections leave the base tool pristine.
        base_params = ANNOTATION_TOOL.function_declarations[0].parameters
        assert "ligand_copies" not in base_params.properties
        assert (
            "pharmacological_role_check" not in base_params.properties["ligands"].items.properties
        )

    def test_no_candidate_copies_leaves_tool_identical_to_base(self) -> None:
        only_buffers = {"nonpolymer_entities": [_np_entity("SO4", [("R", "A", "801")])]}
        copy_ids = ligand_copy_id_enum(ligand_copy_identifiers(only_buffers))
        assert (
            build_tool_for_signals(ANNOTATION_TOOL, [], ligand_copy_ids=copy_ids) is ANNOTATION_TOOL
        )


class TestBuildToolConfigLigandCopies:
    def test_forwards_copy_ids_leaving_base_config_unchanged(self) -> None:
        cfg = build_tool_config([], ligand_copy_ids=["R:601"])
        assert cfg is not TOOL_CONFIG
        params = cfg.tools[0].function_declarations[0].parameters
        assert "ligand_copies" in params.properties
        assert TOOL_CONFIG.tools[0] is ANNOTATION_TOOL  # base config untouched

    def test_no_copy_ids_returns_base_config_identity(self) -> None:
        assert build_tool_config([], ligand_copy_ids=[]) is TOOL_CONFIG
        assert build_tool_config([], ligand_copy_ids=None) is TOOL_CONFIG


class TestAssembleLigandCopyBlock:
    def test_none_when_roster_empty(self) -> None:
        assert assemble_ligand_copy_block([], []) is None

    def test_one_line_per_copy_including_sparse(self) -> None:
        roster = [("CLR", "R:602"), ("CLR", "R:605"), ("J40", "R:601")]
        # Only R:602 has mapped geometry; R:605 and R:601 are sparse.
        signals = [_site_ref("CLR", [_copy(["3x33"], ["TM3"], 1, 0.9, copy_id="R:602")])]
        block = assemble_ligand_copy_block(roster, signals)
        assert block is not None
        # Every enum member gets a per-copy line carrying its component id.
        for comp, cid in roster:
            assert f"copy {cid} ({comp}):" in block
        # The mapped copy shows its geometry facts; the sparse copies show the note.
        assert "3x33" in block
        assert block.count("few receptor contacts / surface-exposed") == 2

    def test_carries_fill_in_instruction(self) -> None:
        block = assemble_ligand_copy_block([("J40", "R:601")], [])
        assert block is not None
        assert "ligand_copies" in block
        assert "may repeat across copies" in block
        assert "'unknown'" in block
        assert "do not add, omit, or alter" in block


class TestPromptRosterSchemaEnumInSync:
    """Desync guard: the copy_id set the PROMPT enumerates must be EXACTLY the
    copy_id enum baked into the SCHEMA. Both derive from ``ligand_copy_identifiers``,
    so any future edit that lets the prompt block and the schema enum drift apart
    (e.g. one filters copies the other keeps) fails here rather than in production."""

    def _copy_ids_in_block(self, block: str) -> set[str]:
        # Each per-copy line is exactly "  copy <copy_id> (<comp_id>): ...".
        return set(re.findall(r"^  copy (\S+) \(", block, re.MULTILINE))

    def test_block_copy_ids_equal_schema_copy_id_enum(self) -> None:
        entry = _multi_ligand_entry()
        roster = ligand_copy_identifiers(entry)

        # PROMPT side: the copy_ids the assembled LIGAND COPIES block actually lists.
        block = assemble_ligand_copy_block(roster, [])
        assert block is not None
        prompt_copy_ids = self._copy_ids_in_block(block)

        # SCHEMA side: the copy_id enum pinned into the built ligand_copies schema.
        tool = build_tool_for_signals(
            ANNOTATION_TOOL, [], ligand_copy_ids=ligand_copy_id_enum(roster)
        )
        ligand_copies = tool.function_declarations[0].parameters.properties["ligand_copies"]
        schema_copy_id_enum = set(ligand_copies.items.properties["copy_id"].enum)

        assert prompt_copy_ids  # non-empty, so this is a real comparison
        assert prompt_copy_ids == schema_copy_id_enum


class TestCheckLigandCopyCoverage:
    """Exact-coverage validation of a returned ligand_copies roster."""

    @staticmethod
    def _rows(*copy_ids: str) -> list[dict]:
        return [
            {"copy_id": cid, "site_ref": "unknown", "role": {"value": "agonist"}}
            for cid in copy_ids
        ]

    def test_exact_coverage_is_ok(self) -> None:
        cov = check_ligand_copy_coverage(self._rows("R:601", "R:602"), ["R:601", "R:602"])
        assert cov.ok
        assert cov.missing == () and cov.duplicated == () and cov.unexpected == ()

    def test_missing_copy_is_detected(self) -> None:
        cov = check_ligand_copy_coverage(self._rows("R:601"), ["R:601", "R:602"])
        assert not cov.ok
        assert cov.missing == ("R:602",)
        assert "missing" in cov.describe()

    def test_duplicate_copy_is_detected(self) -> None:
        cov = check_ligand_copy_coverage(self._rows("R:601", "R:601"), ["R:601", "R:602"])
        assert not cov.ok
        assert cov.duplicated == ("R:601",)
        # R:602 was never returned, so it is also flagged missing.
        assert cov.missing == ("R:602",)

    def test_out_of_set_copy_is_detected(self) -> None:
        cov = check_ligand_copy_coverage(self._rows("R:601", "R:602", "R:999"), ["R:601", "R:602"])
        assert not cov.ok
        assert cov.unexpected == ("R:999",)

    def test_empty_roster_is_a_noop(self) -> None:
        # No candidate copies: nothing to validate, always ok -- even if the
        # response somehow carried rows.
        assert check_ligand_copy_coverage(None, []).ok
        assert check_ligand_copy_coverage(self._rows("R:601"), []).ok

    def test_non_list_response_surfaces_as_missing(self) -> None:
        # A malformed (non-list) ligand_copies contributes no coverage rather
        # than raising, so every expected copy is reported missing.
        cov = check_ligand_copy_coverage("not-a-list", ["R:601"])
        assert not cov.ok
        assert cov.missing == ("R:601",)
