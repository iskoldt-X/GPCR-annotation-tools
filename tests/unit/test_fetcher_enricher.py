"""Tests for fetcher/enricher.py — enrichment logic with mocked HTTP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from gpcr_tools.fetcher.enricher import (
    _determine_ligand_type,
    _enrich_siblings,
    _enrich_uniprot,
    _fetch_chem_comp_descriptors,
    _get_pubchem_cid,
    _get_pubchem_synonyms,
    _tag_polymer_ligand_types,
    enrich_single_pdb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_PDB_DATA: dict[str, Any] = {
    "data": {
        "entry": {
            "rcsb_id": "7W55",
            "polymer_entities": [
                {
                    "uniprots": [
                        {"rcsb_id": "P29274"},
                        {"rcsb_id": "P63092"},
                    ]
                }
            ],
            "nonpolymer_entities": [
                {
                    "nonpolymer_comp": {
                        "chem_comp": {
                            "id": "ZMA",
                            "formula_weight": 385.4,
                        },
                        "rcsb_chem_comp_descriptor": {
                            "InChIKey": "OIPILFWXSMYKGL-UHFFFAOYSA-N",
                        },
                    }
                }
            ],
            "rcsb_primary_citation": {
                "pdbx_database_id_DOI": "10.1038/s41586-022-04958-8",
            },
        }
    }
}


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up workspace with raw JSON and dirs."""
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))

    from gpcr_tools.config import reset_config

    reset_config()

    raw_dir = tmp_path / "raw" / "pdb_json"
    raw_dir.mkdir(parents=True)
    enriched_dir = tmp_path / "enriched"
    enriched_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Write raw JSON
    raw_file = raw_dir / "7W55.json"
    raw_file.write_text(json.dumps(_MINIMAL_PDB_DATA))

    yield tmp_path

    reset_config()


# ---------------------------------------------------------------------------
# determine_ligand_type
# ---------------------------------------------------------------------------


class TestDetermineLigandType:
    def test_lipid_from_whitelist(self) -> None:
        # Cholesterol, palmitic acid and myristic acid are genuine lipids on the
        # curated whitelist regardless of CCD type.
        for comp_id in ("CLR", "PLM", "MYR"):
            assert _determine_ligand_type(comp_id, {"type": "non-polymer"}) == "lipid"

    def test_non_lipid_acids_are_small_molecule(self) -> None:
        # Short-chain acids and retinal are deliberately excluded from the lipid
        # whitelist and stay small molecules.
        for comp_id in ("ACT", "SIN", "RET"):
            assert _determine_ligand_type(comp_id, {"type": "non-polymer"}) == "small-molecule"

    def test_heavy_nonpolymer_is_not_peptide(self) -> None:
        # Heavy detergents (e.g. maltose-neopentyl-glycol surrogates) are small
        # molecules; the old weight proxy wrongly called them peptides.
        for comp_id in ("AV0", "BYI", "WB2"):
            assert (
                _determine_ligand_type(comp_id, {"type": "non-polymer", "formula_weight": 1050.0})
                == "small-molecule"
            )

    def test_saccharide_routes_to_small_molecule(self) -> None:
        assert _determine_ligand_type("NAG", {"type": "D-saccharide"}) == "small-molecule"

    def test_free_amino_acid_is_small_molecule(self) -> None:
        assert _determine_ligand_type("GLU", {"type": "L-peptide linking"}) == "small-molecule"

    def test_free_mononucleotide_is_small_molecule(self) -> None:
        # A single nucleotide-linking component is a free mononucleotide (e.g. a
        # bound GDP/GTP cofactor) -- a small molecule. The 'na' value is reserved
        # for polymer nucleic-acid entities, classified on the polymer path.
        assert _determine_ligand_type("A", {"type": "RNA linking"}) == "small-molecule"
        assert _determine_ligand_type("DA", {"type": "DNA linking"}) == "small-molecule"

    def test_unknown_type_defaults_small_molecule(self) -> None:
        assert _determine_ligand_type("ZMA", {}) == "small-molecule"

    def test_missing_comp_id_defaults_small_molecule(self) -> None:
        # A nonpolymer entity with no comp_id cannot hit the lipid whitelist and
        # has no CCD type to route on; it falls through to small-molecule.
        assert _determine_ligand_type(None, {}) == "small-molecule"

    def test_lipid_whitelist_honored_without_ccd_type(self) -> None:
        # Older fetches lack chem_comp.type; the lipid whitelist must still win.
        assert _determine_ligand_type("CLR", {}) == "lipid"


class TestTagPolymerLigandTypes:
    @staticmethod
    def _poly(
        poly_type: str,
        length: int | None,
        uniprot_ids: list[str] | None = None,
        uniprots: list[dict[str, Any]] | None = None,
        reference_sequence_identifiers: list[Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "entity_poly": {"type": poly_type, "rcsb_sample_sequence_length": length},
            "rcsb_polymer_entity_container_identifiers": {
                "uniprot_ids": uniprot_ids,
                "reference_sequence_identifiers": reference_sequence_identifiers,
            },
            "uniprots": uniprots or [],
        }

    def _run(self, polymers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pdb_data = {"data": {"entry": {"polymer_entities": polymers}}}
        _tag_polymer_ligand_types(pdb_data)
        return polymers

    def test_short_peptide_without_uniprot_is_peptide(self) -> None:
        poly = self._poly("polypeptide(L)", 12)
        self._run([poly])
        assert poly["gpcrdb_determined_type"] == "peptide"

    def test_polypeptide_with_uniprot_is_skipped(self) -> None:
        poly = self._poly("polypeptide(L)", 12, uniprot_ids=["P12345"])
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_polypeptide_with_uniprot_object_is_skipped(self) -> None:
        poly = self._poly("polypeptide(L)", 12, uniprots=[{"rcsb_id": "P12345"}])
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_polypeptide_with_reference_sequence_is_skipped(self) -> None:
        # A non-UniProt cross-reference (PDB / EMBL / etc.) still marks a resolved
        # entity, not a peptide ligand, so the hint must not be set.
        poly = self._poly(
            "polypeptide(L)", 12, reference_sequence_identifiers=[{"database_accession": "1ABC"}]
        )
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_long_polypeptide_is_skipped(self) -> None:
        poly = self._poly("polypeptide(L)", 350)
        self._run([poly])
        assert "gpcrdb_determined_type" not in poly

    def test_nucleotide_polymer_is_na(self) -> None:
        poly = self._poly("polyribonucleotide", 20)
        self._run([poly])
        assert poly["gpcrdb_determined_type"] == "na"


# ---------------------------------------------------------------------------
# UniProt enrichment
# ---------------------------------------------------------------------------


class TestEnrichUniprot:
    def test_adds_slug_from_api(self) -> None:
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "P29274"}]}]}}
        }
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"primaryAccession": "P29274", "uniProtkbId": "AA2AR_HUMAN"}]
        }
        mock_session.post.return_value = mock_response

        _enrich_uniprot(pdb_data, mock_session, cache=None)

        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert uni["gpcrdb_entry_name_slug"] == "aa2ar_human"

    def test_uses_cache_hit(self) -> None:
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"polymer_entities": [{"uniprots": [{"rcsb_id": "P29274"}]}]}}
        }
        mock_session = MagicMock()
        cache = MagicMock()
        cache.has.return_value = True
        cache.get.return_value = "aa2ar_human"

        _enrich_uniprot(pdb_data, mock_session, cache=cache)

        # Session should not be called (cache hit)
        mock_session.post.assert_not_called()
        uni = pdb_data["data"]["entry"]["polymer_entities"][0]["uniprots"][0]
        assert uni["gpcrdb_entry_name_slug"] == "aa2ar_human"


# ---------------------------------------------------------------------------
# Sibling enrichment
# ---------------------------------------------------------------------------


class TestEnrichSiblings:
    def test_adds_siblings_excluding_self(self) -> None:
        pdb_data: dict[str, Any] = {
            "data": {"entry": {"rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1038/test"}}}
        }
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result_set": [
                {"identifier": "7W55"},
                {"identifier": "8ABC"},
                {"identifier": "9XYZ"},
            ]
        }
        mock_session.post.return_value = mock_response

        _enrich_siblings(pdb_data, "7W55", mock_session, cache=None)

        siblings = pdb_data["data"]["entry"]["sibling_pdbs"]
        assert siblings == ["8ABC", "9XYZ"]
        assert "7W55" not in siblings

    def test_no_doi_gives_empty_list(self) -> None:
        pdb_data: dict[str, Any] = {"data": {"entry": {"rcsb_primary_citation": None}}}
        _enrich_siblings(pdb_data, "7W55", MagicMock(), cache=None)
        assert pdb_data["data"]["entry"]["sibling_pdbs"] == []


# ---------------------------------------------------------------------------
# Full enrich_single_pdb
# ---------------------------------------------------------------------------


class TestEnrichSinglePdb:
    def test_skips_if_enriched_exists(self, workspace: Path) -> None:
        enriched_path = workspace / "enriched" / "7W55.json"
        enriched_path.write_text("{}")

        result = enrich_single_pdb("7W55")
        assert result is True  # skipped successfully

    def test_fails_if_raw_missing(self, workspace: Path) -> None:
        result = enrich_single_pdb("XXXX")
        assert result is False

    @patch("gpcr_tools.fetcher.enricher._enrich_siblings")
    @patch("gpcr_tools.fetcher.enricher._enrich_ligands")
    @patch("gpcr_tools.fetcher.enricher._enrich_uniprot")
    def test_writes_enriched_output(
        self,
        mock_uniprot: MagicMock,
        mock_ligands: MagicMock,
        mock_siblings: MagicMock,
        workspace: Path,
    ) -> None:
        result = enrich_single_pdb("7W55")
        assert result is True

        enriched_path = workspace / "enriched" / "7W55.json"
        assert enriched_path.exists()

        data = json.loads(enriched_path.read_text())
        assert data["data"]["entry"]["rcsb_id"] == "7W55"

    def test_all_lookups_failed_returns_false_and_writes_nothing(self, workspace: Path) -> None:
        """If every external lookup hard-fails (transient outage), don't persist
        a hollow enriched file or report success — leave it for the next run."""
        enriched = workspace / "enriched" / "7W55.json"
        if enriched.exists():
            enriched.unlink()

        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("down")
        session.get.side_effect = requests.exceptions.ConnectionError("down")

        result = enrich_single_pdb("7W55", session=session, force=True)

        assert result is False
        assert not enriched.exists()


class TestEnrichmentCacheNotPoisoned:
    """A transient lookup failure must never be written to the shared cache —
    the cache is keyed by InChIKey/CID/comp_id (not PDB), persists across runs,
    and is not cleared by --force, so a cached negative from one network blip
    would suppress that ligand's enrichment for every future run.
    """

    @staticmethod
    def _resp(status, payload=None):
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload or {}
        return r

    def test_cid_transient_failure_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("boom")
        assert _get_pubchem_cid("KEY", session, cache) is None
        cache.set.assert_not_called()

    def test_cid_non_200_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(503)
        assert _get_pubchem_cid("KEY", session, cache) is None
        cache.set.assert_not_called()

    def test_cid_200_is_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.return_value = self._resp(200, {"IdentifierList": {"CID": [271]}})
        assert _get_pubchem_cid("KEY", session, cache) == "271"
        cache.set.assert_called_once_with("KEY", "271")

    def test_synonyms_transient_failure_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ReadTimeout("slow")
        assert _get_pubchem_synonyms("271", session, cache) is None
        cache.set.assert_not_called()

    def test_chem_comp_transient_failure_not_cached(self):
        cache = MagicMock()
        cache.has.return_value = False
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("boom")
        assert _fetch_chem_comp_descriptors("ATP", session, cache) is None
        cache.set.assert_not_called()
