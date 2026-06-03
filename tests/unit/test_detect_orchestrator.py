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
    SIGNAL_DISPUTED_LIGAND,
    SIGNAL_DUAL_ROLE_LIGAND,
    DetectSignal,
)


def _chimeric_advisory() -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_CHIMERIC_GPROTEIN,
        target_ref="signaling_partners.g_protein.alpha_subunit",
        summary="alpha5 resolves to gnai1",
        payload={"family": "Gi/o", "subtype": "gnai1_human", "a5_tail": "IKENLKDCGLF", "score": 11},
        severity=SEVERITY_ADVISORY,
    )


def _disputed(comp: str = "PLM") -> DetectSignal:
    return DetectSignal(
        kind=SIGNAL_DISPUTED_LIGAND,
        target_ref="ligands",
        summary=f"{comp} disputed",
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

    def test_disputed_renders_comp_and_guidance(self) -> None:
        block = assemble_detect_block([_disputed("PLM")])
        assert block is not None
        assert "PLM" in block and "disputed_assessment" in block

    def test_deterministic_order(self) -> None:
        a = assemble_detect_block([_disputed("PLM"), _chimeric_advisory()])
        b = assemble_detect_block([_chimeric_advisory(), _disputed("PLM")])
        assert a == b

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
    def test_no_disputed_returns_base_identity(self) -> None:
        assert build_tool_for_signals(ANNOTATION_TOOL, []) is ANNOTATION_TOOL
        assert build_tool_for_signals(ANNOTATION_TOOL, [_chimeric_advisory()]) is ANNOTATION_TOOL

    def test_disputed_adds_field_without_mutating_base(self) -> None:
        tool = build_tool_for_signals(ANNOTATION_TOOL, [_disputed()])
        assert tool is not ANNOTATION_TOOL
        items = tool.function_declarations[0].parameters.properties["ligands"].items
        assert "disputed_assessment" in items.properties
        # The base tool must be untouched (no schema leak).
        base_items = ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items
        assert "disputed_assessment" not in base_items.properties

    def test_dual_role_adds_site_ref_without_mutating_base(self) -> None:
        tool = build_tool_for_signals(ANNOTATION_TOOL, [_dual_role()])
        assert tool is not ANNOTATION_TOOL
        items = tool.function_declarations[0].parameters.properties["ligands"].items
        assert "site_ref" in items.properties
        base_items = ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items
        assert "site_ref" not in base_items.properties

    def test_disputed_and_dual_role_add_both_fields(self) -> None:
        tool = build_tool_for_signals(ANNOTATION_TOOL, [_disputed(), _dual_role()])
        items = tool.function_declarations[0].parameters.properties["ligands"].items
        assert "disputed_assessment" in items.properties
        assert "site_ref" in items.properties


class TestBuildToolConfig:
    def test_no_disputed_returns_base_config_identity(self) -> None:
        assert build_tool_config([]) is TOOL_CONFIG
        assert build_tool_config([_chimeric_advisory()]) is TOOL_CONFIG

    def test_dual_role_returns_new_config_leaving_base_unchanged(self) -> None:
        cfg = build_tool_config([_dual_role()])
        assert cfg is not TOOL_CONFIG
        assert cfg.tools[0] is not ANNOTATION_TOOL
        assert TOOL_CONFIG.tools[0] is ANNOTATION_TOOL

    def test_disputed_returns_new_config_leaving_base_unchanged(self) -> None:
        cfg = build_tool_config([_disputed()])
        assert cfg is not TOOL_CONFIG
        assert cfg.tools[0] is not ANNOTATION_TOOL
        assert TOOL_CONFIG.tools[0] is ANNOTATION_TOOL  # base config untouched
