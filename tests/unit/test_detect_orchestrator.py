"""Tests for detect_orchestrator: prompt evidence block + tool/config routing.

Key invariants: advisory signals produce a block / augmented tool; review-only
or no signals leave the prompt block None and the tool/config returned by
identity (zero perturbation); the base ANNOTATION_TOOL / TOOL_CONFIG are never
mutated.
"""

from __future__ import annotations

from gpcr_tools.annotator.detect_orchestrator import (
    assemble_detect_block,
    build_tool_config,
    build_tool_for_signals,
)
from gpcr_tools.annotator.schema import ANNOTATION_TOOL, TOOL_CONFIG
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_DUAL_ROLE_LIGAND,
    SIGNAL_INCIDENTAL_CANDIDATE,
    SIGNAL_SITE_REF,
    DetectSignal,
)


def _site_ref(comp: str, sites: list[str]) -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_SITE_REF,
        target_ref="ligands",
        summary=f"{comp} site",
        payload={"comp_id": comp, "sites": sites},
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

    def test_deterministic_order(self) -> None:
        a = assemble_detect_block([_incidental_candidate("PLM"), _chimeric_advisory()])
        b = assemble_detect_block([_chimeric_advisory(), _incidental_candidate("PLM")])
        assert a == b

    def test_site_ref_single_site_evidence(self) -> None:
        block = assemble_detect_block([_site_ref("ADN", ["orthosteric"])])
        assert block is not None
        assert "ADN" in block and "orthosteric" in block and "site_ref" in block

    def test_site_ref_multi_site_evidence(self) -> None:
        block = assemble_detect_block(
            [_site_ref("A1AEI", ["extracellular_vestibule", "orthosteric"])]
        )
        assert block is not None
        assert "2 distinct sites" in block
        assert "one ligand entry per site" in block

    def test_split_instruction_owned_by_site_ref_not_dual_role(self) -> None:
        # The dual-role signal gives burial evidence but must NOT command a split;
        # only the site_ref multi-site signal owns the "one entry per site" nudge.
        dual = assemble_detect_block([_dual_role("A1AEI")])
        assert dual is not None
        assert "more than one role" in dual
        assert "entry per" not in dual  # no split command from dual-role
        site = assemble_detect_block(
            [_site_ref("A1AEI", ["extracellular_vestibule", "orthosteric"])]
        )
        assert "one ligand entry per site" in site

    def test_dual_role_renders_per_copy_evidence_and_site_ref(self) -> None:
        block = assemble_detect_block([_dual_role("A1AEI")])
        assert block is not None
        assert "A1AEI" in block and "site_ref" in block
        # one line per buried copy, with its enclosure and pocket residues
        assert "R/601" in block and "R/602" in block
        assert "104" in block  # a pocket residue of the first copy
        # the partner-contacting copy is flagged as the possible active-state pocket
        assert "active-state pocket" in block


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
