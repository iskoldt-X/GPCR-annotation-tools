"""Tests for ligand cross-validation and chemical identity injection (Epic 3).

Covers: small molecule match, polymer match, ghost ligand, buffer exclusion,
APO handling, None-safety, and warning format compliance.
"""

from __future__ import annotations

import re
from typing import Any

from gpcr_tools.config import (
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_MATCHED_POLYMER,
    VALIDATION_MATCHED_SMALL_MOLECULE,
    VALIDATION_SKIPPED_APO,
)
from gpcr_tools.validator.ligand_validator import validate_and_enrich_ligands

# Regex contract from Blood Lesson 3
_WARNING_REGEX = re.compile(r"at ['\"]([^'\"]+)['\"]")


def _make_enriched(
    *,
    nonpolymer: list[dict[str, Any]] | None = None,
    polymer: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if nonpolymer is not None:
        entry["nonpolymer_entities"] = nonpolymer
    if polymer is not None:
        entry["polymer_entities"] = polymer
    return entry


def _np_entity(
    comp_id: str,
    name: str = "Test",
    inchikey: str = "IK123",
    pubchem_cid: str = "12345",
) -> dict[str, Any]:
    return {
        "nonpolymer_comp": {
            "chem_comp": {"id": comp_id, "name": name},
            "rcsb_chem_comp_descriptor": {
                "InChIKey": inchikey,
                "SMILES": "C=O",
                "SMILES_stereo": "C=O",
            },
            "gpcrdb_pubchem_cid": pubchem_cid,
        }
    }


def _poly_entity(chain_id: str, sequence: str = "MDEF") -> dict[str, Any]:
    return {
        "entity_poly": {
            "pdbx_seq_one_letter_code_can": sequence,
            "type": "polypeptide(L)",
        },
        "rcsb_polymer_entity": {"pdbx_description": "Test protein"},
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": chain_id}}
        ],
    }


class TestSmallMoleculeMatch:
    def test_matched(self) -> None:
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "ATP", "name": "Adenosine"}]}
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        lig = data["ligands"][0]
        assert lig["validation_status"] == VALIDATION_MATCHED_SMALL_MOLECULE
        assert lig["InChIKey"] == "IK123"
        assert lig["api_pubchem_cid"] == "12345"
        assert lig["SMILES"] == "C=O"
        assert lig["SMILES_stereo"] == "C=O"


class TestPolymerMatch:
    def test_peptide_by_chain(self) -> None:
        data: dict[str, Any] = {
            "ligands": [{"chain_id": "B", "name": "Peptide X", "type": "peptide"}]
        }
        enriched = _make_enriched(polymer=[_poly_entity("B", sequence="ACDEF")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        lig = data["ligands"][0]
        assert lig["validation_status"] == VALIDATION_MATCHED_POLYMER
        assert lig["Sequence"] == "ACDEF"

    def test_protein_by_chain(self) -> None:
        data: dict[str, Any] = {
            "ligands": [{"chain_id": "C", "name": "Some protein", "type": "protein"}]
        }
        enriched = _make_enriched(polymer=[_poly_entity("C")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        assert data["ligands"][0]["validation_status"] == VALIDATION_MATCHED_POLYMER

    def test_protein_multi_chain(self) -> None:
        data: dict[str, Any] = {
            "ligands": [
                {"chain_id": "X, Y", "name": "Follicle stimulating hormone", "type": "protein"}
            ]
        }
        # Simulate a PDB where chain X and chain Y exist
        enriched = _make_enriched(polymer=[_poly_entity("X"), _poly_entity("Y")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        lig = data["ligands"][0]
        assert lig["validation_status"] == VALIDATION_MATCHED_POLYMER
        assert lig["Sequence"] == "MDEF / MDEF"


class TestGhostLigand:
    def test_ghost_ligand_with_comp_id(self) -> None:
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "XYZ", "name": "Fake Drug"}]}
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1
        assert data["ligands"][0]["validation_status"] == VALIDATION_GHOST_LIGAND
        assert "GHOST_LIGAND" in warnings[0]
        assert "XYZ" in warnings[0]
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_ghost_ligand_no_comp_id(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "Mystery"}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1
        assert "Mystery" in warnings[0]
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_ghost_ligand_is_flagged_not_removed(self) -> None:
        # The validator only flags; it never drops the entry. Deciding whether
        # an unverified ligand reaches the export is left to the curator and the
        # CSV writer, so the flagged ligand must survive validation intact.
        data: dict[str, Any] = {
            "ligands": [
                {"chem_comp_id": "ATP", "name": "Adenosine"},
                {"chem_comp_id": "XYZ", "name": "Fake Drug"},
            ]
        }
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        validate_and_enrich_ligands("TEST", data, enriched)
        assert len(data["ligands"]) == 2
        assert data["ligands"][1]["name"] == "Fake Drug"
        assert data["ligands"][1]["validation_status"] == VALIDATION_GHOST_LIGAND


class TestBufferExclusion:
    def test_excluded_buffer(self) -> None:
        """Buffer comp_ids in LIGAND_EXCLUDE_LIST are excluded at context build time,
        so they won't match as small molecules and should be ghost unless the
        buffer appears in the AI data explicitly."""
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "GOL", "name": "Glycerol"}]}
        enriched = _make_enriched(nonpolymer=[_np_entity("ATP")])
        validate_and_enrich_ligands("TEST", data, enriched)
        assert data["ligands"][0]["validation_status"] == VALIDATION_EXCLUDED_BUFFER


class TestApoHandling:
    def test_apo_name(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "apo", "chem_comp_id": ""}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        assert data["ligands"][0]["validation_status"] == VALIDATION_SKIPPED_APO

    def test_apo_comp_id(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "No ligand", "chem_comp_id": "apo"}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert warnings == []
        assert data["ligands"][0]["validation_status"] == VALIDATION_SKIPPED_APO

    def test_warns_when_apo_coexists_with_real_ligand(self) -> None:
        # An apo (ligand-free) placeholder alongside a real ligand is
        # contradictory; surface it for the curator, do not silently edit.
        data: dict[str, Any] = {
            "ligands": [
                {"name": "apo", "chem_comp_id": ""},
                {"name": "ATP", "chem_comp_id": "ATP"},
            ]
        }
        warnings = validate_and_enrich_ligands("TEST", data, _make_enriched())
        assert any("apo" in w.lower() and "coexist" in w.lower() for w in warnings)
        # warning-only: the apo entry must NOT be removed
        assert len(data["ligands"]) == 2

    def test_no_apo_coexistence_warning_for_pure_apo(self) -> None:
        data: dict[str, Any] = {"ligands": [{"name": "apo", "chem_comp_id": ""}]}
        warnings = validate_and_enrich_ligands("TEST", data, _make_enriched())
        assert not any("coexist" in w.lower() for w in warnings)

    def test_no_apo_coexistence_warning_with_only_buffer(self) -> None:
        # Apo + an excluded buffer (glycerol) is normal, not contradictory.
        data: dict[str, Any] = {
            "ligands": [
                {"name": "apo", "chem_comp_id": ""},
                {"name": "glycerol", "chem_comp_id": "GOL"},
            ]
        }
        warnings = validate_and_enrich_ligands("TEST", data, _make_enriched())
        assert not any("coexist" in w.lower() for w in warnings)


class TestNoneSafety:
    def test_null_chem_comp_id(self) -> None:
        """Blood Lesson 1: explicit null chem_comp_id must not crash."""
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": None, "name": "Test"}]}
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1

    def test_null_pdbx_description(self) -> None:
        """Null pdbx_description must not crash polymer context building."""
        data: dict[str, Any] = {"ligands": [{"chain_id": "A", "name": "Test", "type": "peptide"}]}
        enriched = _make_enriched(
            polymer=[
                {
                    "entity_poly": {},
                    "rcsb_polymer_entity": {"pdbx_description": None},
                    "polymer_entity_instances": [
                        {
                            "rcsb_polymer_entity_instance_container_identifiers": {
                                "auth_asym_id": "A"
                            }
                        }
                    ],
                }
            ]
        )
        # Should not crash
        validate_and_enrich_ligands("TEST", data, enriched)

    def test_missing_enriched_fields(self) -> None:
        """Empty enriched entry must not crash."""
        data: dict[str, Any] = {"ligands": [{"chem_comp_id": "TEST", "name": "Test"}]}
        enriched: dict[str, Any] = {}
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert len(warnings) == 1

    def test_no_ligands_key(self) -> None:
        data: dict[str, Any] = {}
        warnings = validate_and_enrich_ligands("TEST", data, {})
        assert warnings == []

    def test_ligands_not_a_list(self) -> None:
        data: dict[str, Any] = {"ligands": "not a list"}
        warnings = validate_and_enrich_ligands("TEST", data, {})
        assert warnings == []


class TestWarningFormat:
    def test_all_warnings_match_regex(self) -> None:
        """Blood Lesson 3: every warning must match the UI regex contract."""
        data: dict[str, Any] = {
            "ligands": [
                {"chem_comp_id": "FAKE1", "name": "Drug1"},
                {"chem_comp_id": None, "name": "Drug2"},
                {"name": "Drug3"},
            ]
        }
        enriched = _make_enriched()
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        for warn in warnings:
            assert _WARNING_REGEX.search(warn) is not None, f"Warning fails regex: {warn}"
