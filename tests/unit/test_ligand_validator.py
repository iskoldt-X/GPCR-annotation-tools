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


def _poly_entity(
    chain_id: str,
    sequence: str = "MDEF",
    description: str = "Test protein",
    slug: str | None = None,
) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "entity_poly": {
            "pdbx_seq_one_letter_code_can": sequence,
            "type": "polypeptide(L)",
        },
        "rcsb_polymer_entity": {"pdbx_description": description},
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": chain_id}}
        ],
    }
    if slug is not None:
        entity["uniprots"] = [{"gpcrdb_entry_name_slug": slug}]
    return entity


def _lig(
    *,
    name: str = "LIG",
    comp: str = "LIG",
    type_: str = "small-molecule",
    role: str | None = None,
    site: str | None = None,
) -> dict[str, Any]:
    lig: dict[str, Any] = {"name": name, "chem_comp_id": comp, "type": type_}
    if role is not None:
        lig["role"] = {"value": role}
    if site is not None:
        lig["site_ref"] = site
    return lig


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


class TestRoleSiteMismatch:
    """The role/site self-consistency net flags only unambiguous contradictions in
    the AI's own output; legitimate non-Class-A sites must never be flagged."""

    def _mismatch_warnings(self, lig: dict[str, Any]) -> list[str]:
        warnings = validate_and_enrich_ligands("TEST", {"ligands": [lig]}, _make_enriched())
        return [w for w in warnings if "ROLE_SITE_MISMATCH" in w]

    def test_lipid_at_orthosteric_warns(self) -> None:
        # A lipid with NO functional role at the orthosteric pocket is the
        # mislabelled-structural-lipid case (the original 7E2X failure).
        assert self._mismatch_warnings(_lig(type_="lipid", site="orthosteric"))

    def test_lipid_agonist_at_orthosteric_ok(self) -> None:
        # An endogenous lipid agonist (S1P / LPA / 2-AG / prostaglandin) legitimately
        # binds the orthosteric pocket as type 'lipid' -> not a contradiction.
        assert not self._mismatch_warnings(_lig(type_="lipid", role="Agonist", site="orthosteric"))

    def test_allosteric_role_at_orthosteric_warns(self) -> None:
        assert self._mismatch_warnings(_lig(role="PAM", site="orthosteric"))
        assert self._mismatch_warnings(_lig(role="Allosteric agonist", site="orthosteric"))

    def test_functional_role_at_lipidic_warns(self) -> None:
        assert self._mismatch_warnings(_lig(role="Agonist", site="lipidic"))

    def test_agonist_at_orthosteric_ok(self) -> None:
        assert not self._mismatch_warnings(_lig(role="Agonist", site="orthosteric"))

    def test_agonist_at_extracellular_domain_ok(self) -> None:
        # Class C / B orthosteric site IS the extracellular domain -> not flagged.
        assert not self._mismatch_warnings(_lig(role="Agonist", site="extracellular_domain"))

    def test_agonist_at_intracellular_ok(self) -> None:
        # Intracellular agonist binding sites are real -> not flagged.
        assert not self._mismatch_warnings(_lig(role="Agonist", site="intracellular"))

    def test_pam_at_allosteric_ok(self) -> None:
        assert not self._mismatch_warnings(_lig(role="PAM", site="allosteric_7tm"))

    def test_lipid_at_lipidic_ok(self) -> None:
        assert not self._mismatch_warnings(_lig(type_="lipid", site="lipidic"))

    def test_warning_matches_regex(self) -> None:
        warnings = self._mismatch_warnings(_lig(role="PAM", site="orthosteric"))
        assert warnings and _WARNING_REGEX.search(warnings[0]) is not None


_G_ALPHA_DESC = "GUANINE NUCLEOTIDE-BINDING PROTEIN G(T) SUBUNIT ALPHA-3"


class TestGProteinPeptideAsLigand:
    """A transducer-derived / G-protein-mimetic peptide filed as a receptor ligand
    with a functional pocket role is a signaling partner mis-annotation. The net
    surfaces it for the curator; an honest abstention (unknown role) is never
    flagged, and genuine small-molecule agonists / auxiliary proteins are untouched.
    """

    def _gp_warnings(
        self,
        lig: dict[str, Any],
        polymer: list[dict[str, Any]],
    ) -> list[str]:
        warnings = validate_and_enrich_ligands(
            "TEST", {"ligands": [lig]}, _make_enriched(polymer=polymer)
        )
        return [w for w in warnings if "G-PROTEIN PEPTIDE AS LIGAND" in w]

    def test_g_alpha_description_peptide_agonist_warns(self) -> None:
        # Chain B's polymer description reads as a G-alpha subunit, yet the peptide
        # is filed as an Agonist -> surface for the curator.
        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        warnings = self._gp_warnings(lig, polymer)
        assert warnings
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_g_alpha_warning_reaches_accept_all_disabling_channel(self) -> None:
        # The returned warning list is extended verbatim into critical_warnings,
        # the channel that disables one-click accept-all.
        from gpcr_tools.aggregator.runner import _build_validation_report

        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        ligand_warnings = validate_and_enrich_ligands(
            "TEST", {"ligands": [lig]}, _make_enriched(polymer=polymer)
        )
        report = _build_validation_report("TEST", {}, {}, ligand_warnings, {}, None)
        assert any("G-PROTEIN PEPTIDE AS LIGAND" in w for w in report["critical_warnings"])

    def test_slug_branch_peptide_agonist_warns(self) -> None:
        # Description does NOT read as a G-alpha, but the chain's GPCRdb slug is a
        # G-protein subunit -> the slug branch (+ poly_by_chain slug extension) fire.
        lig = {"chain_id": "B", "name": "Transducin mimetic", "type": "peptide"}
        lig["role"] = {"value": "Agonist"}
        polymer = [
            _poly_entity(
                "B", sequence="ACDEF", description="Engineered peptide", slug="gnat1_bovin"
            )
        ]
        warnings = self._gp_warnings(lig, polymer)
        assert warnings
        assert _WARNING_REGEX.search(warnings[0]) is not None

    def test_slug_branch_gna_prefix_warns(self) -> None:
        lig = {"chain_id": "C", "name": "Mini-G peptide", "type": "protein"}
        lig["role"] = {"value": "Agonist"}
        polymer = [
            _poly_entity("C", sequence="ACDEF", description="some construct", slug="gnas2_human")
        ]
        assert self._gp_warnings(lig, polymer)

    def test_slug_branch_gbb_and_gbg_prefixes_warn(self) -> None:
        # The G-beta / G-gamma subunit slug prefixes are also signaling-partner
        # markers: a beta/gamma chain filed as a functional ligand should fire.
        for slug in ("gbb1_human", "gbg2_human"):
            lig = {"chain_id": "D", "name": "Subunit peptide", "type": "peptide"}
            lig["role"] = {"value": "Agonist"}
            polymer = [_poly_entity("D", sequence="ACDEF", description="some construct", slug=slug)]
            assert self._gp_warnings(lig, polymer), slug

    def test_small_molecule_agonist_not_flagged(self) -> None:
        # The real agonist is a small molecule, not a peptide on a G-protein chain.
        data: dict[str, Any] = {
            "ligands": [
                {
                    "chem_comp_id": "RET",
                    "name": "all-trans-retinal",
                    "type": "small-molecule",
                    "role": {"value": "Agonist"},
                }
            ]
        }
        enriched = _make_enriched(nonpolymer=[_np_entity("RET")])
        warnings = validate_and_enrich_ligands("TEST", data, enriched)
        assert not any("G-PROTEIN PEPTIDE AS LIGAND" in w for w in warnings)

    def test_auxiliary_fab_not_flagged(self) -> None:
        # A stabilising Fab/nanobody is an auxiliary protein, not a peptide ligand
        # with a functional pocket role -> no agonist mis-annotation to flag.
        lig = {"chain_id": "H", "name": "Fab heavy chain", "type": "protein"}
        polymer = [_poly_entity("H", sequence="ACDEF", description="Antibody Fab fragment")]
        # No functional role: this is just a structural auxiliary protein.
        assert not self._gp_warnings(lig, polymer)

    def test_g_alpha_peptide_with_unknown_role_not_flagged(self) -> None:
        # The model honestly abstaining (role unknown) must never be flagged.
        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        lig["role"] = {"value": "unknown"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        assert not self._gp_warnings(lig, polymer)

    def test_g_alpha_peptide_with_absent_role_not_flagged(self) -> None:
        lig = {"chain_id": "B", "name": "C-terminal peptide", "type": "peptide"}
        polymer = [_poly_entity("B", sequence="ACDEF", description=_G_ALPHA_DESC)]
        assert not self._gp_warnings(lig, polymer)
