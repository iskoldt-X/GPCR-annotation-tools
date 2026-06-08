"""Tests for the site_ref geometry-evidence detector (detect_site_refs).

The deterministic ``classify_site`` rule was retired (the architecture flip): the
detector now emits per-copy geometric FACTS and the model infers the site. These
tests cover the enriched parsing and the per-copy evidence orchestration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpcr_tools.config import INCIDENTAL_CANDIDATES
from gpcr_tools.detector import site_ref as sr


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


class TestEnrichedParsing:
    def test_gpcr_chain_accessions(self) -> None:
        assert sr._gpcr_chain_accessions(_entry()) == {"R": "Q9NYV8"}

    def test_non_gpcr_yields_no_chains(self) -> None:
        assert sr._gpcr_chain_accessions(_entry(slug="gnas2_human")) == {}

    def test_annotated_keeps_real_ligand(self) -> None:
        # Every real (non-buffer) ligand is annotated and gets facts (recall);
        # whether it is one or several sites is left to the model.
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

    Set the fixture list to ``(enclosure, evidence_or_None)`` per copy: the stub
    feeds each copy's enclosure through ligand_contact_residues and its evidence
    dict (or None for a too-sparse copy) through _copy_evidence. membrane_frame is
    stubbed to None so the facing/depth facts are skipped in these tests.
    """
    copies: list[tuple[float, dict | None]] = []
    monkeypatch.setattr(sr, "load_structure", lambda *a, **k: object())
    monkeypatch.setattr(
        sr, "fetch_polymer_alignment", lambda *a, **k: {"R": {"Q9NYV8": [(1, 1, 400)]}}
    )
    monkeypatch.setattr(sr, "membrane_frame", lambda *a, **k: None)
    monkeypatch.setattr(sr, "ligand_contact_residues", lambda *a, **k: list(copies))
    monkeypatch.setattr(sr, "_copy_evidence", lambda contacts, *a, **k: contacts)
    return copies


_ORTH = {"generic_numbers": ["3x33", "6x51"], "segments": ["TM3", "TM6"], "core_hits": 2, "mapped": 10}
_VEST = {"generic_numbers": ["45x52"], "segments": ["ECL2"], "core_hits": 0, "mapped": 8}


class TestDetectSiteRefs:
    def test_single_copy_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(0.92, dict(_ORTH))]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals) == 1
        copies = signals[0].payload["copies"]
        assert len(copies) == 1
        assert copies[0]["generic_numbers"] == ["3x33", "6x51"]
        assert copies[0]["core_hits"] == 2
        assert copies[0]["enclosure"] == 0.92  # the burial is recorded as an enclosure fact

    def test_multi_copy_facts_not_collapsed(self, stub_pipeline, tmp_path: Path) -> None:
        # Both copies' facts are emitted (distinct sites); the model decides whether
        # to emit one entry per site -- the detector no longer makes that call.
        stub_pipeline[:] = [(0.95, dict(_ORTH)), (0.9, dict(_VEST))]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals) == 1
        copies = signals[0].payload["copies"]
        assert len(copies) == 2
        assert {c["segments"][0] for c in copies} == {"TM3", "ECL2"}

    def test_shallow_copy_still_emitted_as_fact(self, stub_pipeline, tmp_path: Path) -> None:
        # A low-enclosure copy is no longer gated out; its enclosure is just a fact
        # (the model reads low enclosure + lipid-facing as a structural-lipid hint).
        stub_pipeline[:] = [(0.40, dict(_VEST))]
        signals = sr.detect_site_refs("X", _entry("CLR"), tmp_path)
        assert signals[0].payload["copies"][0]["enclosure"] == 0.40

    def test_sparse_copy_skipped(self, stub_pipeline, tmp_path: Path) -> None:
        # A copy with too few mapped contacts (_copy_evidence -> None) is dropped.
        stub_pipeline[:] = [(0.9, dict(_ORTH)), (0.5, None)]
        signals = sr.detect_site_refs("X", _entry("LIG"), tmp_path)
        assert len(signals[0].payload["copies"]) == 1

    def test_all_sparse_emits_no_signal(self, stub_pipeline, tmp_path: Path) -> None:
        stub_pipeline[:] = [(0.9, None), (0.8, None)]
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []

    def test_no_gpcr_chain_short_circuits(self, tmp_path: Path) -> None:
        assert sr.detect_site_refs("X", _entry(slug="gnas2_human"), tmp_path) == []

    def test_missing_structure_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sr, "load_structure", lambda *a, **k: None)
        assert sr.detect_site_refs("X", _entry("LIG"), tmp_path) == []
