"""Tests for the site_ref classifier rule (classify_site)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpcr_tools.config import (
    DISPUTED_MOLECULES,
    SITE_REF_ALLOSTERIC_7TM,
    SITE_REF_EXTRACELLULAR_DOMAIN,
    SITE_REF_EXTRACELLULAR_VESTIBULE,
    SITE_REF_INTRACELLULAR,
    SITE_REF_ORTHOSTERIC,
    SITE_REF_UNKNOWN,
)
from gpcr_tools.detector import site_ref as sr
from gpcr_tools.detector.site_ref import classify_site


def _entry(comp: str = "LIG", slug: str = "t2r14_human", acc: str = "Q9NYV8", soi: bool = True) -> dict:
    nonpoly = {
        "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": comp},
        "nonpolymer_entity_instances": [
            {"rcsb_nonpolymer_entity_instance_container_identifiers": {"asym_id": "B"}}
        ],
    }
    if soi:
        nonpoly["rcsb_nonpolymer_entity_annotation"] = [{"type": "SUBJECT_OF_INVESTIGATION"}]
    return {
        "polymer_entities": [
            {
                "uniprots": [{"gpcrdb_entry_name_slug": slug, "rcsb_id": acc}],
                "polymer_entity_instances": [
                    {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "R"}}
                ],
            }
        ],
        "nonpolymer_entities": [nonpoly],
    }


class TestClassifySite:
    def test_empty_is_unknown(self) -> None:
        assert classify_site("001", set(), set()) == SITE_REF_UNKNOWN

    def test_class_a_orthosteric_core(self) -> None:
        assert classify_site("001", {"3x32", "6x48"}, {"TM3", "TM6"}) == SITE_REF_ORTHOSTERIC

    def test_multiple_core_beats_vestibule(self) -> None:
        # A ligand reaching several core residues but also touching ECL2 is orthosteric.
        assert (
            classify_site("001", {"3x32", "6x48"}, {"TM3", "ECL2"}) == SITE_REF_ORTHOSTERIC
        )

    def test_single_core_with_vestibule_is_vestibule(self) -> None:
        # One grazing core residue + a vestibule signature -> vestibule, not
        # orthosteric (the M2 PAM LY2119620 case that brushes the top of TM3).
        assert (
            classify_site("001", {"3x32"}, {"TM3", "ECL2"})
            == SITE_REF_EXTRACELLULAR_VESTIBULE
        )

    def test_taste_t2_deep_pocket_is_orthosteric(self) -> None:
        assert classify_site("009", {"3x47", "6x38"}, {"TM3", "TM6"}) == SITE_REF_ORTHOSTERIC

    def test_taste_t2_does_not_use_mid_core(self) -> None:
        # 3x32 is the shared mid-core, not the T2 deep core, so with only an
        # upper-TM/ECL signature a T2 ligand is the vestibule, not orthosteric.
        assert (
            classify_site("009", {"3x32"}, {"TM3", "ECL2"}) == SITE_REF_EXTRACELLULAR_VESTIBULE
        )

    def test_allosteric_7tm(self) -> None:
        assert classify_site("001", {"3x40", "4x50"}, {"TM3", "TM4"}) == SITE_REF_ALLOSTERIC_7TM

    def test_vestibule(self) -> None:
        assert classify_site("001", set(), {"ECL2", "TM2"}) == SITE_REF_EXTRACELLULAR_VESTIBULE

    def test_intracellular(self) -> None:
        assert classify_site("001", set(), {"ICL3", "H8"}) == SITE_REF_INTRACELLULAR

    def test_class_c_vft_is_extracellular_domain(self) -> None:
        assert classify_site("004", set(), {"N-term"}) == SITE_REF_EXTRACELLULAR_DOMAIN

    def test_class_c_7tm_is_allosteric(self) -> None:
        assert classify_site("004", {"6x48"}, {"TM3", "TM6"}) == SITE_REF_ALLOSTERIC_7TM

    def test_class_c_ecd_precedence_over_tm(self) -> None:
        # A class C ligand touching both the VFT (N-term) and the 7TM is the VFT
        # orthosteric agonist -> extracellular_domain (ECD is checked before 7TM).
        assert (
            classify_site("004", {"6x48"}, {"N-term", "TM6"}) == SITE_REF_EXTRACELLULAR_DOMAIN
        )

    def test_large_ecd_class_b(self) -> None:
        assert classify_site("002", set(), {"N-term"}) == SITE_REF_EXTRACELLULAR_DOMAIN

    def test_unknown_class_falls_back_to_generic_core(self) -> None:
        # An unrecognised class still classifies on the shared core signature.
        assert classify_site(None, {"6x48", "3x32"}, {"TM6", "TM3"}) == SITE_REF_ORTHOSTERIC


class TestEnrichedParsing:
    def test_gpcr_chain_accessions(self) -> None:
        assert sr._gpcr_chain_accessions(_entry()) == {"R": "Q9NYV8"}

    def test_non_gpcr_yields_no_chains(self) -> None:
        assert sr._gpcr_chain_accessions(_entry(slug="gnas2_human")) == {}

    def test_annotated_keeps_real_ligand(self) -> None:
        # A real ligand is annotated whether or not it is flagged studied (recall).
        assert sr._annotated_ligands(_entry("LIG", soi=False)) == {"LIG"}

    def test_studied_requires_soi(self) -> None:
        assert sr._studied_ligands(_entry("LIG", soi=True)) == {"LIG"}
        assert sr._studied_ligands(_entry("SOG", soi=False)) == set()

    def test_disputed_not_studied_unless_soi(self) -> None:
        # A disputed molecule is still annotated (gets a site) but does NOT earn
        # the multi-site split nudge unless it is also a subject of investigation.
        disputed = sorted(DISPUTED_MOLECULES)[0]
        assert sr._studied_ligands(_entry(disputed, soi=False)) == set()
        assert disputed in sr._annotated_ligands(_entry(disputed, soi=False))

    def test_prefers_table_accession_over_fusion_partner(self) -> None:
        # Both UniProts pass the permissive slug denylist; the real receptor
        # (in the shipped table) must win over a crystallization fusion partner.
        entry = {
            "polymer_entities": [
                {
                    "uniprots": [
                        {"gpcrdb_entry_name_slug": "rubr_clopa", "rcsb_id": "P00268"},
                        {"gpcrdb_entry_name_slug": "adrb2_human", "rcsb_id": "P07550"},
                    ],
                    "polymer_entity_instances": [
                        {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "A"}}
                    ],
                }
            ]
        }
        assert sr._gpcr_chain_accessions(entry) == {"A": "P07550"}


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Stub the I/O so detect_site_refs exercises only its orchestration."""
    monkeypatch.setattr(sr, "load_structure", lambda *a, **k: object())
    monkeypatch.setattr(sr, "fetch_polymer_alignment", lambda *a, **k: {"R": {"Q9NYV8": [(1, 1, 400)]}})
    monkeypatch.setattr(sr, "ligand_contact_residues", lambda *a, **k: [["copy1"], ["copy2"]])
    per_copy: list[tuple[str, int]] = []
    monkeypatch.setattr(sr, "_classify_copy", lambda *a, **k: per_copy.pop(0))
    return per_copy


class TestDetectSiteRefs:
    def test_single_site_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(SITE_REF_ORTHOSTERIC, 10), (SITE_REF_ORTHOSTERIC, 8)]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals) == 1
        assert signals[0].payload["sites"] == [SITE_REF_ORTHOSTERIC]

    def test_studied_ligand_keeps_multiple_sites(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(SITE_REF_ORTHOSTERIC, 10), (SITE_REF_EXTRACELLULAR_VESTIBULE, 8)]
        signals = sr.detect_site_refs("X", _entry("LIG", soi=True), tmp_path)
        assert signals[0].payload["sites"] == [
            SITE_REF_EXTRACELLULAR_VESTIBULE,
            SITE_REF_ORTHOSTERIC,
        ]

    def test_non_studied_multi_site_collapses_to_dominant(
        self, stub_pipeline, tmp_path: Path
    ) -> None:
        # A detergent scattered across grooves must NOT be told to emit 3 entries;
        # it reports only its most-contacted site.
        stub_pipeline[:] = [(SITE_REF_ORTHOSTERIC, 6), (SITE_REF_EXTRACELLULAR_VESTIBULE, 12)]
        signals = sr.detect_site_refs("X", _entry("SOG", soi=False), tmp_path)
        assert signals[0].payload["sites"] == [SITE_REF_EXTRACELLULAR_VESTIBULE]

    def test_unknown_copies_emit_no_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(SITE_REF_UNKNOWN, 2), (SITE_REF_UNKNOWN, 1)]
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []

    def test_no_gpcr_chain_short_circuits(self, tmp_path: Path) -> None:
        assert sr.detect_site_refs("X", _entry(slug="gnas2_human"), tmp_path) == []

    def test_missing_structure_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sr, "load_structure", lambda *a, **k: None)
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []
