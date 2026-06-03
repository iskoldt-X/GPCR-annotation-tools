"""Tests for the dual-role ligand detector (the rule + enriched parsing).

The geometry compute (gemmi) and coordinate fetch are stubbed, so these tests
exercise the decision logic only: which studied, multi-copy ligands bound at two
distinct receptor pockets become a signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpcr_tools.detector import geometry as detector_geometry
from gpcr_tools.detector.geometry import (
    _gpcr_auth_chains,
    _subject_of_investigation_comp_ids,
    detect_dual_role_ligands,
)
from gpcr_tools.detector.signals import SIGNAL_DUAL_ROLE_LIGAND
from gpcr_tools.validator.geometry import LigandCopyGeometry


def _copy(
    seq_id: int,
    *,
    burial: float = 0.99,
    residues: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    pocket_chain: str = "R",
    partner: bool = False,
) -> LigandCopyGeometry:
    return LigandCopyGeometry(
        auth_chain="R",
        seq_id=seq_id,
        burial=burial,
        pocket_residues=frozenset((pocket_chain, r) for r in residues),
        contacts_partner=partner,
    )


def _entry(soi_comps: tuple[str, ...] = ("A1AEI",), *, gpcr_slug: str = "t2r14_human") -> dict:
    return {
        "polymer_entities": [
            {
                "uniprots": [{"gpcrdb_entry_name_slug": gpcr_slug}],
                "polymer_entity_instances": [
                    {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "R"}}
                ],
            }
        ],
        "nonpolymer_entities": [
            {
                "rcsb_nonpolymer_entity_annotation": [{"type": "SUBJECT_OF_INVESTIGATION"}],
                "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": comp},
            }
            for comp in soi_comps
        ],
    }


@pytest.fixture
def stub_geometry(monkeypatch: pytest.MonkeyPatch):
    """Stub load_structure (non-None) and let a test set the per-comp copies."""
    monkeypatch.setattr(detector_geometry, "load_structure", lambda *a, **k: object())

    copies_by_comp: dict[str, list[LigandCopyGeometry]] = {}

    def fake_analyze(structure, comp_id, gpcr_chains):
        return copies_by_comp.get(comp_id, [])

    monkeypatch.setattr(detector_geometry, "analyze_ligand_copies", fake_analyze)
    return copies_by_comp


class TestEnrichedParsing:
    def test_gpcr_auth_chains(self) -> None:
        assert _gpcr_auth_chains(_entry()) == {"R"}

    def test_non_gpcr_slug_yields_no_chains(self) -> None:
        assert _gpcr_auth_chains(_entry(gpcr_slug="gnas2_human")) == set()

    def test_subject_of_investigation_comp_ids(self) -> None:
        assert _subject_of_investigation_comp_ids(_entry(("A1AEI", "CLR"))) == {"A1AEI", "CLR"}

    def test_no_annotation_is_not_soi(self) -> None:
        entry = _entry()
        entry["nonpolymer_entities"][0]["rcsb_nonpolymer_entity_annotation"] = []
        assert _subject_of_investigation_comp_ids(entry) == set()


class TestDualRoleRule:
    def test_two_buried_distinct_pockets_flagged(self, stub_geometry, tmp_path: Path) -> None:
        stub_geometry["A1AEI"] = [
            _copy(601, residues=(1, 2, 3, 4, 5, 6), partner=True),
            _copy(602, residues=(20, 21, 22, 23, 24, 25)),
        ]
        signals = detect_dual_role_ligands("9IIX", _entry(), tmp_path)
        assert len(signals) == 1
        assert signals[0].kind == SIGNAL_DUAL_ROLE_LIGAND
        assert signals[0].payload["comp_id"] == "A1AEI"
        assert signals[0].payload["gpcr_chain"] == "R"

    def test_three_copies_two_pockets_reports_two(self, stub_geometry, tmp_path: Path) -> None:
        # Two copies share a pocket, the third is distinct: 2 sites, not 3. The
        # evidence must report 2 pockets, never tell the model to emit 3 entries.
        stub_geometry["A1AEI"] = [
            _copy(601, residues=(1, 2, 3, 4, 5, 6)),
            _copy(602, residues=(1, 2, 3, 4, 5, 6)),
            _copy(603, residues=(20, 21, 22, 23, 24, 25)),
        ]
        signals = detect_dual_role_ligands("X", _entry(), tmp_path)
        assert len(signals) == 1
        assert len(signals[0].payload["copies"]) == 2

    def test_three_copies_three_pockets_reports_three(
        self, stub_geometry, tmp_path: Path
    ) -> None:
        stub_geometry["A1AEI"] = [
            _copy(601, residues=(1, 2, 3, 4, 5, 6)),
            _copy(602, residues=(10, 11, 12, 13, 14, 15)),
            _copy(603, residues=(20, 21, 22, 23, 24, 25)),
        ]
        signals = detect_dual_role_ligands("X", _entry(), tmp_path)
        assert len(signals) == 1
        assert len(signals[0].payload["copies"]) == 3

    def test_three_copies_one_pocket_rejected(self, stub_geometry, tmp_path: Path) -> None:
        stub_geometry["A1AEI"] = [_copy(seq, residues=(1, 2, 3, 4, 5, 6)) for seq in (601, 602, 603)]
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []

    def test_copy_flood_rejected(self, stub_geometry, tmp_path: Path) -> None:
        # A detergent flood: more copies than the cap, even if some are buried.
        stub_geometry["A1AEI"] = [
            _copy(seq, residues=(seq, seq + 1, seq + 2, seq + 3, seq + 4, seq + 5))
            for seq in (601, 611, 621, 631)
        ]
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []

    def test_same_pocket_rejected(self, stub_geometry, tmp_path: Path) -> None:
        stub_geometry["A1AEI"] = [
            _copy(601, residues=(1, 2, 3, 4, 5, 6)),
            _copy(602, residues=(1, 2, 3, 4, 5, 6)),
        ]
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []

    def test_surface_copies_rejected(self, stub_geometry, tmp_path: Path) -> None:
        # Low burial = membrane-facing surface lipid, not a pocket.
        stub_geometry["A1AEI"] = [
            _copy(601, burial=0.45, residues=(1, 2, 3, 4, 5, 6)),
            _copy(602, burial=0.50, residues=(20, 21, 22, 23, 24, 25)),
        ]
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []

    def test_too_few_pocket_residues_rejected(self, stub_geometry, tmp_path: Path) -> None:
        stub_geometry["A1AEI"] = [
            _copy(601, residues=(1, 2, 3)),
            _copy(602, residues=(20, 21, 22)),
        ]
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []

    def test_distinct_chains_not_dual_role(self, stub_geometry, tmp_path: Path) -> None:
        # Same ligand in pockets of two different receptor chains = one role replicated.
        stub_geometry["A1AEI"] = [
            _copy(601, residues=(1, 2, 3, 4, 5, 6), pocket_chain="R"),
            _copy(601, residues=(1, 2, 3, 4, 5, 6), pocket_chain="B"),
        ]
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []


class TestShortCircuits:
    def test_no_soi_skips(self, stub_geometry, tmp_path: Path) -> None:
        entry = _entry()
        entry["nonpolymer_entities"] = []
        assert detect_dual_role_ligands("X", entry, tmp_path) == []

    def test_no_gpcr_chain_skips(self, stub_geometry, tmp_path: Path) -> None:
        assert detect_dual_role_ligands("X", _entry(gpcr_slug="gnas2_human"), tmp_path) == []

    def test_missing_structure_skips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(detector_geometry, "load_structure", lambda *a, **k: None)
        assert detect_dual_role_ligands("X", _entry(), tmp_path) == []
