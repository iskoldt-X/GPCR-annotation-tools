"""Tests for the site_ref classifier rule (classify_site)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpcr_tools.config import (
    INCIDENTAL_CANDIDATES,
    SITE_REF_ALLOSTERIC_7TM,
    SITE_REF_EXTRACELLULAR_DOMAIN,
    SITE_REF_EXTRACELLULAR_VESTIBULE,
    SITE_REF_INTRACELLULAR,
    SITE_REF_ORTHOSTERIC,
    SITE_REF_UNKNOWN,
)
from gpcr_tools.detector import site_ref as sr
from gpcr_tools.detector.site_ref import classify_site


def _entry(comp: str = "LIG", slug: str = "t2r14_human", acc: str = "Q9NYV8") -> dict:
    return {
        "polymer_entities": [
            {
                "uniprots": [{"gpcrdb_entry_name_slug": slug, "rcsb_id": acc}],
                "polymer_entity_instances": [
                    {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "R"}}
                ],
            }
        ],
        "nonpolymer_entities": [
            {
                "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": comp},
                "nonpolymer_entity_instances": [
                    {"rcsb_nonpolymer_entity_instance_container_identifiers": {"asym_id": "B"}}
                ],
            }
        ],
    }


class TestClassifySite:
    def test_empty_is_unknown(self) -> None:
        assert classify_site("001", set(), set()) == SITE_REF_UNKNOWN

    def test_class_a_orthosteric_core(self) -> None:
        assert classify_site("001", {"3x32", "6x48"}, {"TM3", "TM6"}) == SITE_REF_ORTHOSTERIC

    def test_multiple_core_beats_vestibule(self) -> None:
        # A ligand reaching several core residues but also touching ECL2 is orthosteric.
        assert classify_site("001", {"3x32", "6x48"}, {"TM3", "ECL2"}) == SITE_REF_ORTHOSTERIC

    def test_single_core_with_vestibule_is_vestibule(self) -> None:
        # One grazing core residue + a vestibule signature -> vestibule, not
        # orthosteric (the M2 PAM LY2119620 case that brushes the top of TM3).
        assert classify_site("001", {"3x32"}, {"TM3", "ECL2"}) == SITE_REF_EXTRACELLULAR_VESTIBULE

    def test_taste_t2_deep_pocket_is_orthosteric(self) -> None:
        assert classify_site("009", {"3x47", "6x38"}, {"TM3", "TM6"}) == SITE_REF_ORTHOSTERIC

    def test_taste_t2_does_not_use_mid_core(self) -> None:
        # 3x32 is the shared mid-core, not the T2 deep core, so with only an
        # upper-TM/ECL signature a T2 ligand is the vestibule, not orthosteric.
        assert classify_site("009", {"3x32"}, {"TM3", "ECL2"}) == SITE_REF_EXTRACELLULAR_VESTIBULE

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
        assert classify_site("004", {"6x48"}, {"N-term", "TM6"}) == SITE_REF_EXTRACELLULAR_DOMAIN

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
        # Every real (non-buffer) ligand is annotated and gets a site (recall);
        # the multi-site split is gated later by burial, not by ligand identity.
        assert sr._annotated_ligands(_entry("LIG")) == {"LIG"}

    def test_incidental_candidate_molecule_annotated(self) -> None:
        incidental_candidate = sorted(INCIDENTAL_CANDIDATES)[0]
        assert incidental_candidate in sr._annotated_ligands(_entry(incidental_candidate))

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
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A"
                            }
                        }
                    ],
                }
            ]
        }
        assert sr._gpcr_chain_accessions(entry) == {"A": "P07550"}


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Stub the I/O so detect_site_refs exercises only its orchestration.

    Set the fixture list to (burial, site, mapped) per copy; the stub feeds each
    copy's burial through ligand_contact_residues and its (site, mapped) through
    _classify_copy.
    """
    copies: list[tuple[float, str, int]] = []
    monkeypatch.setattr(sr, "load_structure", lambda *a, **k: object())
    monkeypatch.setattr(
        sr, "fetch_polymer_alignment", lambda *a, **k: {"R": {"Q9NYV8": [(1, 1, 400)]}}
    )
    monkeypatch.setattr(
        sr, "ligand_contact_residues", lambda *a, **k: [(b, (s, m)) for b, s, m in copies]
    )
    monkeypatch.setattr(sr, "_classify_copy", lambda contacts, *a, **k: contacts)
    return copies


_BURIED = 0.95  # >= GEOMETRY_BURIAL_MIN
_SHALLOW = 0.45  # < GEOMETRY_BURIAL_MIN (surface lipid)


class TestDetectSiteRefs:
    def test_single_site_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(_BURIED, SITE_REF_ORTHOSTERIC, 10), (_BURIED, SITE_REF_ORTHOSTERIC, 8)]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals) == 1
        assert signals[0].payload["sites"] == [SITE_REF_ORTHOSTERIC]

    def test_buried_copies_in_distinct_sites_split(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [
            (_BURIED, SITE_REF_ORTHOSTERIC, 10),
            (_BURIED, SITE_REF_EXTRACELLULAR_VESTIBULE, 8),
        ]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert signals[0].payload["sites"] == [
            SITE_REF_EXTRACELLULAR_VESTIBULE,
            SITE_REF_ORTHOSTERIC,
        ]

    def test_shallow_scattered_copies_collapse_to_dominant(
        self, stub_pipeline, tmp_path: Path
    ) -> None:
        # Cholesterol scattered across shallow surface grooves (low burial) must
        # NOT be split into multiple entries, even spanning several site classes.
        stub_pipeline[:] = [
            (_SHALLOW, SITE_REF_ALLOSTERIC_7TM, 6),
            (_SHALLOW, SITE_REF_EXTRACELLULAR_VESTIBULE, 12),
        ]
        signals = sr.detect_site_refs("X", _entry("CLR"), tmp_path)
        assert signals[0].payload["sites"] == [SITE_REF_EXTRACELLULAR_VESTIBULE]

    def test_one_buried_one_shallow_does_not_split(self, stub_pipeline, tmp_path: Path) -> None:
        # Only one real (buried) pocket -> no split, even though a shallower copy
        # sits at a different site with more contacts. The dominant of all sites
        # (the shallow one, by contact count) is reported as the single site.
        stub_pipeline[:] = [
            (_BURIED, SITE_REF_ORTHOSTERIC, 5),
            (_SHALLOW, SITE_REF_EXTRACELLULAR_VESTIBULE, 12),
        ]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert signals[0].payload["sites"] == [SITE_REF_EXTRACELLULAR_VESTIBULE]

    def test_split_uses_only_buried_sites(self, stub_pipeline, tmp_path: Path) -> None:
        # Two buried pockets drive the split; a third, shallow copy at another
        # site is excluded from the per-site list (only real pockets split).
        stub_pipeline[:] = [
            (_BURIED, SITE_REF_ORTHOSTERIC, 10),
            (_BURIED, SITE_REF_EXTRACELLULAR_VESTIBULE, 8),
            (_SHALLOW, SITE_REF_INTRACELLULAR, 20),
        ]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert signals[0].payload["sites"] == [
            SITE_REF_EXTRACELLULAR_VESTIBULE,
            SITE_REF_ORTHOSTERIC,
        ]

    def test_unknown_copies_emit_no_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(_BURIED, SITE_REF_UNKNOWN, 2), (_BURIED, SITE_REF_UNKNOWN, 1)]
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []

    def test_no_gpcr_chain_short_circuits(self, tmp_path: Path) -> None:
        assert sr.detect_site_refs("X", _entry(slug="gnas2_human"), tmp_path) == []

    def test_missing_structure_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sr, "load_structure", lambda *a, **k: None)
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []
