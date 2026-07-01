"""Tests for CSV transformation and writing logic.

These tests cover the pure data transformation layer — no UI, no user interaction.
"""

import copy
import csv
from dataclasses import replace
from unittest.mock import patch

from gpcr_tools.aggregator.runner import _prune_excluded_buffer_ligands
from gpcr_tools.config import (
    CSV_SCHEMA,
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_GHOST_LIGAND,
    VALIDATION_MATCHED_SMALL_MOLECULE,
)
from gpcr_tools.csv_generator.csv_writer import (
    append_to_csvs,
    sanitize_value,
    transform_for_csv,
)


class TestGpcrdbColumnContract:
    """The annotation CSVs are read positionally by the downstream build, so the
    leading columns must match its contract exactly; our extra columns
    (label_asym_id, chemistry fields) are appended after, never inserted.
    """

    def test_structures_core_columns(self):
        assert CSV_SCHEMA["structures.csv"][:8] == (
            "PDB",
            "Receptor_UniProt",
            "Method",
            "Resolution",
            "State",
            "ChainID",
            "Note",
            "Date",
        )
        # Our extra columns are appended after the positional contract, never inserted.
        assert {"label_asym_id", "Partner_UniProt", "Partner_ChainID"} <= set(
            CSV_SCHEMA["structures.csv"][8:]
        )

    def test_ligands_core_columns(self):
        # The downstream build reads the leading 9 columns positionally; "Site"
        # (and the chemistry columns) are appended after, never inserted.
        assert CSV_SCHEMA["ligands.csv"][:9] == (
            "PDB",
            "ChainID",
            "Name",
            "PubChemID",
            "Role",
            "Title",
            "Type",
            "Date",
            "In structure",
        )
        assert {
            "label_asym_id",
            "SMILES",
            "InChIKey",
            "Sequence",
            "is_endogenous",
            "Site",
            "Residue_seq_id",
        } <= set(CSV_SCHEMA["ligands.csv"][9:])

    def test_g_proteins_core_columns(self):
        assert CSV_SCHEMA["g_proteins.csv"][:8] == (
            "PDB",
            "Alpha_identity",
            "Alpha_ChainID",
            "Beta_UniProt",
            "Beta_ChainID",
            "Gamma_UniProt",
            "Gamma_ChainID",
            "Note",
        )
        # The label_asym_id columns and the alpha5 functional-coupling / backbone
        # columns are appended after the positional contract, never inserted.
        assert {
            "Alpha_label_asym_id",
            "Beta_label_asym_id",
            "Gamma_label_asym_id",
            "Alpha_alpha5_identity",
            "Alpha_backbone",
        } <= set(CSV_SCHEMA["g_proteins.csv"][8:])

    def test_arrestins_core_columns(self):
        assert CSV_SCHEMA["arrestins.csv"][:4] == ("PDB", "UniProt", "ChainID", "Note")
        assert "label_asym_id" in CSV_SCHEMA["arrestins.csv"][4:]


class TestSanitizeValue:
    def test_none_returns_empty(self):
        assert sanitize_value(None) == ""

    def test_string_stripped(self):
        assert sanitize_value("  hello  ") == "hello"

    def test_numeric(self):
        assert sanitize_value(2.5) == "2.5"

    def test_zero(self):
        assert sanitize_value(0) == "0"

    def test_bool(self):
        assert sanitize_value(True) == "True"


class TestTransformForCSV:
    def test_produces_all_csv_keys(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        from gpcr_tools.config import CSV_SCHEMA

        assert set(result.keys()) == set(CSV_SCHEMA.keys())

    def test_structures_csv_row(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["structures.csv"]
        assert len(rows) == 1
        row = rows[0]
        assert row["PDB"] == "TEST1"
        assert row["Receptor_UniProt"] == "aa2ar_human"
        assert row["Method"] == "ELECTRON MICROSCOPY"
        assert row["Resolution"] == "2.5"
        assert row["State"] == "Active"
        assert row["ChainID"] == "R"
        assert row["Date"] == "2025-01-15"
        # A monomer has no partner protomer.
        assert row["Partner_UniProt"] == ""
        assert row["Partner_ChainID"] == ""

    def test_heterodimer_partner_columns(self, sample_pdb_data):
        # GABA-B: primary GABBR2 (B); the partner GABBR1 (A) is recorded, not dropped.
        data = copy.deepcopy(sample_pdb_data)
        data["receptor_info"] = {"uniprot_entry_name": "gabr2_human", "chain_id": "B"}
        data["oligomer_analysis"] = {
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "gabr1_human"},
                {"chain_id": "B", "slug": "gabr2_human"},
            ],
            "primary_protomer_suggestion": {"chain_id": "B", "reason": "coupling"},
        }
        row = transform_for_csv("TEST1", data)["structures.csv"][0]
        assert row["Receptor_UniProt"] == "gabr2_human"
        assert row["ChainID"] == "B"
        assert row["Partner_UniProt"] == "gabr1_human"
        assert row["Partner_ChainID"] == "A"

    def test_ligands_csv_row(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["ligands.csv"]
        assert len(rows) == 1
        row = rows[0]
        assert row["PDB"] == "TEST1"
        # Name carries the canonical PDBe chemical-component code; the descriptive
        # name moves to Title.
        assert row["Name"] == "ADN"
        assert row["Title"] == "Adenosine"
        assert row["PubChemID"] == "2519"
        assert row["Role"] == "Agonist"
        assert row["ChainID"] == "A"
        assert row["InChIKey"] == "OIRDTQYFTABQOQ-KQYNXXCUSA-N"
        # An ordinary ligand has no dual-role site_ref, so the Site column is blank.
        assert row["Site"] == ""
        # No nonpolymer instance index in this fixture -> no residue numbers.
        assert row["Residue_seq_id"] == ""

    def test_site_ref_populates_site_column(self, sample_pdb_data):
        data = copy.deepcopy(sample_pdb_data)
        data["ligands"][0]["site_ref"] = "allosteric"
        row = transform_for_csv("TEST1", data)["ligands.csv"][0]
        assert row["Site"] == "allosteric"

    def test_smiles_stereo_priority(self, sample_pdb_data):
        """SMILES_stereo should take priority over SMILES."""
        result = transform_for_csv("TEST1", sample_pdb_data)
        row = result["ligands.csv"][0]
        expected_smiles = sample_pdb_data["ligands"][0]["SMILES_stereo"]
        assert row["SMILES"] == expected_smiles

    def test_g_protein_mapping(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["g_proteins.csv"]
        assert len(rows) == 1
        row = rows[0]
        assert row["Alpha_identity"] == "gnas2_human"
        assert row["Alpha_ChainID"] == "G"
        assert row["Beta_UniProt"] == "gbb1_human"
        assert row["Gamma_UniProt"] == "gbg2_human"

    def test_g_protein_functional_coupling_and_backbone_columns(self, sample_pdb_data):
        # The aggregator records the alpha5 functional coupling identity and the
        # modelled backbone scaffold as distinct fields on the alpha subunit; the
        # CSV exports them as trailing columns while Alpha_identity stays the
        # deposited slug.
        data = copy.deepcopy(sample_pdb_data)
        alpha = data["signaling_partners"]["g_protein"]["alpha_subunit"]
        alpha["functional_coupling"] = "gnaq_human"
        alpha["backbone"] = "gnas2_human"
        row = transform_for_csv("TEST1", data)["g_proteins.csv"][0]
        assert row["Alpha_identity"] == "gnas2_human"
        assert row["Alpha_alpha5_identity"] == "gnaq_human"
        assert row["Alpha_backbone"] == "gnas2_human"

    def test_g_protein_new_columns_blank_when_absent(self, sample_pdb_data):
        # When the aggregator left functional_coupling unset (family mismatch /
        # off-roster), the column is blank rather than a stray "None".
        row = transform_for_csv("TEST1", sample_pdb_data)["g_proteins.csv"][0]
        assert row["Alpha_alpha5_identity"] == ""
        assert row["Alpha_backbone"] == ""

    def test_g_protein_chain_collapses_multivalue(self, sample_pdb_data):
        # Redundant complexes in the asymmetric unit give a multi-chain subunit
        # value ("C, D"); it collapses to the primary complex's chain.
        data = copy.deepcopy(sample_pdb_data)
        data["signaling_partners"]["g_protein"]["alpha_subunit"]["chain_id"] = "C, D"
        row = transform_for_csv("TEST1", data)["g_proteins.csv"][0]
        assert row["Alpha_ChainID"] == "C"
        # The label column follows the collapsed single chain (not "C, D").
        assert row["Alpha_label_asym_id"] == "C"

    def test_apo_placeholder_ligand_skipped(self, sample_pdb_data):
        # An apo / "no ligand" placeholder must not become a ligand row.
        data = copy.deepcopy(sample_pdb_data)
        data["ligands"].append(
            {
                "name": "Apo",
                "chem_comp_id": "",
                "chain_id": "None",
                "type": "none",
                "role": {"value": "Apo (no ligand)"},
                "site_ref": "orthosteric",
            }
        )
        # Descriptive names now live in Title (Name carries the component code).
        names = [r["Title"] for r in transform_for_csv("TEST1", data)["ligands.csv"]]
        assert "Apo" not in names
        assert "Adenosine" in names

    def test_excluded_buffer_pruned_upstream_yields_no_ligand_row(self, sample_pdb_data):
        # Belt-and-suspenders: a BOG / NAG-shaped EXCLUDED_BUFFER ligand is dropped
        # upstream by the aggregator's prune helper, so it never reaches the CSV
        # transform and produces no ligands.csv row -- the bona-fide ligand stays.
        data = copy.deepcopy(sample_pdb_data)
        data["ligands"].append(
            {
                "name": "n-octyl-beta-D-glucoside",
                "chem_comp_id": "BOG",
                "chain_id": "A",
                "type": "small-molecule",
                "role": {"value": "Cofactor"},
                "validation_status": VALIDATION_EXCLUDED_BUFFER,
            }
        )
        _prune_excluded_buffer_ligands(data)
        # Descriptive names now live in Title (Name carries the component code).
        names = [r["Title"] for r in transform_for_csv("TEST1", data)["ligands.csv"]]
        assert "n-octyl-beta-D-glucoside" not in names
        assert "Adenosine" in names

    def test_nanobody_dispatch(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        rows = result["nanobodies.csv"]
        assert len(rows) == 1
        assert rows[0]["Name"] == "Nb35"

    def test_no_arrestin_when_absent(self, sample_pdb_data):
        result = transform_for_csv("TEST1", sample_pdb_data)
        assert result["arrestins.csv"] == []

    def test_empty_data_produces_structure_row(self):
        """Even minimal data should produce a structures.csv entry."""
        result = transform_for_csv("EMPTY", {})
        assert len(result["structures.csv"]) == 1
        assert result["structures.csv"][0]["PDB"] == "EMPTY"

    def test_controversy_data_transform(self, sample_controversy_data):
        """Test that controversy fixture also transforms correctly."""
        result = transform_for_csv("TEST2", sample_controversy_data)
        assert len(result["structures.csv"]) == 1
        assert result["structures.csv"][0]["Method"] == "X-RAY DIFFRACTION"
        assert result["g_proteins.csv"] == []  # no g protein in this fixture

    def test_label_asym_id_with_oligomer(self, sample_oligomer_data):
        """Oligomer fixture with label_asym_id_map → mapped values in CSV rows."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        struct_row = result["structures.csv"][0]
        # Oligomer fixture has chain "A, B" → truncated to "A" (primary)
        # label_map: {"A": "A"} → label_asym_id = "A"
        assert "label_asym_id" in struct_row
        assert struct_row["label_asym_id"] == "A"

    def test_label_asym_id_without_oligo(self, sample_pdb_data):
        """Without oligomer_analysis, label_asym_id falls back to chain_id."""
        result = transform_for_csv("TEST1", sample_pdb_data)
        struct_row = result["structures.csv"][0]
        # No label_map → fallback: chain_id "R" mapped to itself
        assert struct_row["label_asym_id"] == "R"

    def test_truncation_in_structures(self, sample_oligomer_data):
        """Multi-chain oligomer fixture → structures.csv has single primary chain."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        struct_row = result["structures.csv"][0]
        # receptor_info.chain_id = "A, B" → truncated to "A"
        assert struct_row["ChainID"] == "A"
        assert "DB TRUNCATION" in struct_row["Note"]

    def test_note_enriched_with_oligo(self, sample_oligomer_data):
        """Oligomer fixture → Note contains classification + alerts."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        note = result["structures.csv"][0]["Note"]
        assert "HOMOMER" in note
        assert "MISSED_PROTOMER" in note

    def test_g_protein_label_asym_id(self, sample_oligomer_data):
        """G protein subunit chain IDs are mapped via label_asym_id_map."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        gp_row = result["g_proteins.csv"][0]
        # label_map: D→A, C→D, E→B
        assert gp_row["Alpha_label_asym_id"] == "A"  # chain D → A
        assert gp_row["Beta_label_asym_id"] == "D"  # chain C → D
        assert gp_row["Gamma_label_asym_id"] == "B"  # chain E → B

    def test_ligands_not_polymer_mapped(self, sample_oligomer_data):
        """A ligand's label_asym_id never comes from the polymer chain map; with
        no nonpolymer instance index the column is blank (not a protein chain)."""
        result = transform_for_csv("OLIGO1", sample_oligomer_data)
        lig_rows = result["ligands.csv"]
        assert len(lig_rows) == 2
        assert lig_rows[0]["label_asym_id"] == ""
        assert lig_rows[1]["label_asym_id"] == ""


def _mock_config_with_csv_dir(csv_dir):
    """Return a patched get_config that redirects csv_output_dir to *csv_dir*."""
    from gpcr_tools.config import get_config

    real_cfg = get_config()
    fake_cfg = replace(real_cfg, csv_output_dir=csv_dir)
    return patch("gpcr_tools.csv_generator.csv_writer.get_config", return_value=fake_cfg)


class TestAppendToCSVs:
    def test_creates_file_with_header(self, tmp_path, monkeypatch, sample_pdb_data):
        """Test that a new CSV file gets a header row."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            csv_data = transform_for_csv("TEST1", sample_pdb_data)
            append_to_csvs(csv_data)

        structures_file = csv_dir / "structures.csv"
        assert structures_file.exists()

        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert rows[0][0] == "PDB"  # header
        assert rows[1][0] == "TEST1"  # data

    def test_files_use_lf_line_endings(self, tmp_path, monkeypatch, sample_pdb_data):
        """Output uses LF, not CRLF, to match the consumed annotation data."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            append_to_csvs(transform_for_csv("TEST1", sample_pdb_data))

        raw = (csv_dir / "structures.csv").read_bytes()
        assert b"\r\n" not in raw
        assert b"\n" in raw

    def test_append_no_duplicate_header(self, tmp_path, monkeypatch, sample_pdb_data):
        """Test that appending to an existing file does NOT duplicate the header."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            csv_data_1 = transform_for_csv("TEST1", sample_pdb_data)
            csv_data_2 = transform_for_csv("TEST2", sample_pdb_data)

            append_to_csvs(csv_data_1)
            append_to_csvs(csv_data_2)

        structures_file = csv_dir / "structures.csv"
        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0][0] == "PDB"  # header
        assert rows[1][0] == "TEST1"
        assert rows[2][0] == "TEST2"

    def test_empty_csv_data_creates_header_only_files(self, tmp_path, monkeypatch):
        """A batch with no rows for a file still emits a header-only file, so the
        downstream build never hits a missing file (e.g. grk/ramp)."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        with _mock_config_with_csv_dir(csv_dir):
            from gpcr_tools.config import CSV_SCHEMA

            empty_data = {fname: [] for fname in CSV_SCHEMA}
            append_to_csvs(empty_data)

            # Every schema file exists, header-only (one line, the header).
            for fname, expected_fields in CSV_SCHEMA.items():
                fpath = csv_dir / fname
                assert fpath.exists(), f"{fname} not created"
                lines = fpath.read_text(encoding="utf-8").splitlines()
                assert lines == ["\t".join(expected_fields)]
            # The header-only write path also uses LF, not CRLF.
            assert b"\r\n" not in (csv_dir / "grk.csv").read_bytes()

    def test_mismatched_headers_raises_error(self, tmp_path, monkeypatch, sample_pdb_data):
        """Existing CSV with outdated headers → CsvSchemaMismatchError raised."""
        import pytest

        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import reset_config
        from gpcr_tools.csv_generator.exceptions import CsvSchemaMismatchError

        reset_config()

        csv_dir = tmp_path / "csv_out"
        csv_dir.mkdir(parents=True)

        # Write a file with old headers (missing label_asym_id)
        old_headers = "PDB\tReceptor_UniProt\tMethod\tResolution\tState\tChainID\tNote\tDate\n"
        structures_file = csv_dir / "structures.csv"
        structures_file.write_text(old_headers)

        with _mock_config_with_csv_dir(csv_dir):
            csv_data = transform_for_csv("TEST1", sample_pdb_data)
            with pytest.raises(CsvSchemaMismatchError):
                append_to_csvs(csv_data)

        # File should still have only the old header — no data appended
        content = structures_file.read_text()
        assert "TEST1" not in content
        assert content.strip() == old_headers.strip()

    def test_matching_headers_appended(self, tmp_path, monkeypatch, sample_pdb_data):
        """Existing CSV with correct headers → rows appended normally."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import CSV_SCHEMA, reset_config

        reset_config()

        csv_dir = tmp_path / "csv_out"
        csv_dir.mkdir(parents=True)

        # Write a file with current correct headers
        correct_headers = "\t".join(CSV_SCHEMA["structures.csv"]) + "\n"
        structures_file = csv_dir / "structures.csv"
        structures_file.write_text(correct_headers)

        with _mock_config_with_csv_dir(csv_dir):
            csv_data = transform_for_csv("TEST1", sample_pdb_data)
            append_to_csvs(csv_data)

        with open(structures_file) as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert rows[1][0] == "TEST1"


class TestGhostLigandExport:
    """A ligand the validator could not find in the structure (GHOST_LIGAND) is
    excluded from ligands.csv unless a curator explicitly confirmed it."""

    def test_ghost_ligand_excluded_by_default(self, sample_pdb_data):
        sample_pdb_data["ligands"] = [
            {
                "name": "Real",
                "chem_comp_id": "ATP",
                "chain_id": "A",
                "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
                "role": {"value": "Agonist"},
            },
            {
                "name": "Sucralose",
                "chem_comp_id": "SUL",
                "chain_id": "None",
                "validation_status": VALIDATION_GHOST_LIGAND,
                "role": {"value": "Agonist"},
            },
        ]
        rows = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"]
        # Name now carries the component code; "Real" is the descriptive name (ATP).
        assert [r["Name"] for r in rows] == ["ATP"]
        assert [r["Title"] for r in rows] == ["Real"]

    def test_ghost_ligand_kept_when_curator_confirms(self, sample_pdb_data):
        sample_pdb_data["ligands"] = [
            {
                "name": "Sucralose",
                "chem_comp_id": "SUL",
                "chain_id": "None",
                "validation_status": VALIDATION_GHOST_LIGAND,
                "curator_kept_ghost": True,
                "role": {"value": "Agonist"},
            },
        ]
        rows = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"]
        # Name now carries the component code (SUL); the descriptive name is Title.
        assert [r["Name"] for r in rows] == ["SUL"]
        assert [r["Title"] for r in rows] == ["Sucralose"]

    def test_non_ghost_ligands_unaffected(self, sample_pdb_data):
        sample_pdb_data["ligands"] = [
            {
                "name": "Matched",
                "chem_comp_id": "ATP",
                "chain_id": "A",
                "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
                "role": {"value": "Agonist"},
            },
            {
                "name": "NoStatus",
                "chem_comp_id": "GTP",
                "chain_id": "B",
                "role": {"value": "Agonist"},
            },
        ]
        rows = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"]
        # Name now carries the component codes; the descriptive names are in Title.
        assert {r["Name"] for r in rows} == {"ATP", "GTP"}
        assert {r["Title"] for r in rows} == {"Matched", "NoStatus"}


class TestNonFunctionalLigandExport:
    """A dual-use molecule the model judged to be a non-functional, incidental
    species (e.g. a structural lipid or covalent PTM such as palmitate in
    rhodopsin) must not be recorded as a bound ligand. The skip gates on the
    model's explicit negative verdict only — a missing or null check is unaffected."""

    def test_non_functional_ligand_excluded(self, sample_pdb_data):
        sample_pdb_data["ligands"] = [
            {
                "name": "Retinal",
                "chem_comp_id": "RET",
                "chain_id": "A",
                "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
                "role": {"value": "Agonist"},
            },
            {
                "name": "Palmitate",
                "chem_comp_id": "PLM",
                "chain_id": "A",
                "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
                "role": {"value": "Cofactor"},
                "pharmacological_role_check": {
                    "is_functional_ligand": False,
                    "evidence": "Covalent palmitoylation site, not a bound ligand.",
                },
            },
        ]
        rows = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"]
        # Name now carries the component code (Retinal -> RET); Palmitate is dropped.
        assert [r["Name"] for r in rows] == ["RET"]
        assert [r["Title"] for r in rows] == ["Retinal"]

    def test_functional_ligand_kept(self, sample_pdb_data):
        sample_pdb_data["ligands"] = [
            {
                "name": "Sphingosine-1-phosphate",
                "chem_comp_id": "S1P",
                "chain_id": "A",
                "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
                "role": {"value": "Agonist"},
                "pharmacological_role_check": {
                    "is_functional_ligand": True,
                    "evidence": "Endogenous agonist of the S1P receptor.",
                },
            },
        ]
        rows = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"]
        # Name now carries the component code (S1P); descriptive name -> Title.
        assert [r["Name"] for r in rows] == ["S1P"]
        assert [r["Title"] for r in rows] == ["Sphingosine-1-phosphate"]

    def test_ligand_without_role_check_unaffected(self, sample_pdb_data):
        sample_pdb_data["ligands"] = [
            {
                "name": "Adenosine",
                "chem_comp_id": "ADN",
                "chain_id": "A",
                "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
                "role": {"value": "Agonist"},
            },
        ]
        rows = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"]
        # Name now carries the component code (Adenosine -> ADN); name -> Title.
        assert [r["Name"] for r in rows] == ["ADN"]
        assert [r["Title"] for r in rows] == ["Adenosine"]


class TestLigandLabelAsymId:
    """A ligand's label_asym_id is its OWN mmCIF instance label(s): one copy ->
    its label, several -> comma-joined, unindexed -> blank. The polymer chain
    map (protein chains only) is never used for a non-polymer ligand."""

    def _ligand(self, **extra):
        base = {
            "name": "Octylglucoside",
            "chem_comp_id": "SOG",
            "chain_id": "A",
            "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
            "role": {"value": "Agonist"},
        }
        base.update(extra)
        return base

    def test_single_instance_uses_true_instance_label(self, sample_pdb_data):
        sample_pdb_data["oligomer_analysis"] = {
            "label_asym_id_map": {"A": "Z"},  # the polymer map would wrongly give 'Z'
            "nonpolymer_instance_index": {
                "SOG": [{"auth_asym_id": "A", "label_asym_id": "F", "auth_seq_id": "501"}]
            },
        }
        sample_pdb_data["ligands"] = [self._ligand()]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        # 'F' is the ligand's own label, not its author chain 'A' nor polymer 'Z'.
        assert row["label_asym_id"] == "F"
        # The residue token comes from the same instance, aligned to the label, and
        # is prefixed with the copy's author chain: "<auth_asym_id>:<auth_seq_id>".
        assert row["Residue_seq_id"] == "A:501"

    def test_multi_instance_joins_labels(self, sample_pdb_data):
        sample_pdb_data["oligomer_analysis"] = {
            "label_asym_id_map": {"A": "Z"},  # polymer map would wrongly give 'Z'
            "nonpolymer_instance_index": {
                "SOG": [
                    {"auth_asym_id": "A", "label_asym_id": "F", "auth_seq_id": "501"},
                    {"auth_asym_id": "A", "label_asym_id": "G", "auth_seq_id": "502"},
                ]
            },
        }
        sample_pdb_data["ligands"] = [self._ligand()]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        # Both copies' own labels, never the receptor polymer label 'Z'.
        assert row["label_asym_id"] == "F, G"
        # Residue tokens track the same instance order, copy-for-copy, each carrying
        # the copy's own author chain prefix.
        assert row["Residue_seq_id"] == "A:501, A:502"

    def test_unindexed_ligand_has_blank_label(self, sample_pdb_data):
        sample_pdb_data["oligomer_analysis"] = {"label_asym_id_map": {"A": "Z"}}
        sample_pdb_data["ligands"] = [self._ligand()]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        # No instance index -> blank, NOT the receptor's polymer label 'Z'.
        assert row["label_asym_id"] == ""
        # No instance -> no residue numbers either.
        assert row["Residue_seq_id"] == ""


class TestLigandNameAndResidue:
    """Name carries the canonical PDBe chemical-component code (falling back to the
    descriptive name only when no code exists); Title carries the descriptive name;
    Residue_seq_id carries an "<auth_asym_id>:<auth_seq_id>" token per modelled copy
    (author chain : author residue number), aligned copy-for-copy with label_asym_id,
    with the chain sourced from the instance's own auth_asym_id (NOT the AI
    chain_id)."""

    def _ligand(self, **extra):
        base = {
            "name": "descriptive name",
            "chem_comp_id": "LIG",
            "chain_id": "A",
            "validation_status": VALIDATION_MATCHED_SMALL_MOLECULE,
            "role": {"value": "Agonist"},
        }
        base.update(extra)
        return base

    def test_three_letter_code_with_residue(self, sample_pdb_data):
        # A three-letter small-molecule code with one modelled copy (6WHA's U0G,
        # the agonist 25CN-NBOH, sits at auth_seq_id 501 on label chain F).
        sample_pdb_data["oligomer_analysis"] = {
            "nonpolymer_instance_index": {
                "U0G": [{"auth_asym_id": "A", "label_asym_id": "F", "auth_seq_id": "501"}]
            }
        }
        sample_pdb_data["ligands"] = [self._ligand(name="25CN-NBOH", chem_comp_id="U0G")]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        assert row["Name"] == "U0G"
        assert row["Title"] == "25CN-NBOH"
        assert row["label_asym_id"] == "F"
        assert row["Residue_seq_id"] == "A:501"

    def test_five_letter_ccd_code_passes_through(self, sample_pdb_data):
        # A five-letter CCD code (the newer PDBe extended namespace) must pass
        # through into Name verbatim, not be truncated or rejected.
        sample_pdb_data["oligomer_analysis"] = {
            "nonpolymer_instance_index": {
                "A1H1S": [{"auth_asym_id": "A", "label_asym_id": "C", "auth_seq_id": "201"}]
            }
        }
        sample_pdb_data["ligands"] = [self._ligand(name="some inhibitor", chem_comp_id="A1H1S")]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        assert row["Name"] == "A1H1S"
        assert row["Title"] == "some inhibitor"
        assert row["Residue_seq_id"] == "A:201"

    def test_no_comp_id_peptide_falls_back_to_name(self, sample_pdb_data):
        # A peptide ligand carries no chemical-component code (the schema emits the
        # "None" sentinel); Name must fall back to the descriptive name, never the
        # literal string "None", and the residue column stays blank.
        sample_pdb_data["oligomer_analysis"] = {}
        sample_pdb_data["ligands"] = [
            self._ligand(name="Substance P", chem_comp_id="None", type="peptide")
        ]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        assert row["Name"] == "Substance P"
        assert row["Name"] != "None"
        assert row["Title"] == "Substance P"
        assert row["Residue_seq_id"] == ""

    def test_no_comp_id_and_no_name_yields_blank_not_sentinel(self, sample_pdb_data):
        # When BOTH the component code and the descriptive name are empty/"None"
        # sentinels, the fallback must collapse to an empty string -- never the
        # literal "None" leaking into Name (nor into the residue column).
        sample_pdb_data["oligomer_analysis"] = {}
        sample_pdb_data["ligands"] = [self._ligand(name="None", chem_comp_id="None")]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        assert row["Name"] == ""
        assert row["Residue_seq_id"] == ""

    def test_branched_glycan_no_instance_blank_residue(self, sample_pdb_data):
        # A branched glycan has no entry in the non-polymer instance index, so the
        # residue column is blank (and Name still carries whatever code it has).
        sample_pdb_data["oligomer_analysis"] = {
            "nonpolymer_instance_index": {"OTHER": [{"label_asym_id": "F", "auth_seq_id": "1"}]}
        }
        sample_pdb_data["ligands"] = [
            self._ligand(name="glycan", chem_comp_id="NAG", type="branched")
        ]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        assert row["Name"] == "NAG"
        assert row["label_asym_id"] == ""
        assert row["Residue_seq_id"] == ""

    def test_residue_aligns_to_label_not_ai_chain_order(self, sample_pdb_data):
        # 9AYF's 9IG (NPS R-568) is modelled twice; the index is sorted by
        # label_asym_id to ["EA" (chain R, 1011), "T" (chain Q, 1010)] -- the
        # REVERSE of the AI chain_id order "Q, R". The residue column must follow
        # the label order, proving it is sourced from the instance list and not
        # zipped against the AI chain_id.
        sample_pdb_data["oligomer_analysis"] = {
            "nonpolymer_instance_index": {
                "9IG": [
                    {"auth_asym_id": "R", "label_asym_id": "EA", "auth_seq_id": "1011"},
                    {"auth_asym_id": "Q", "label_asym_id": "T", "auth_seq_id": "1010"},
                ]
            }
        }
        sample_pdb_data["ligands"] = [
            self._ligand(name="NPS R-568", chem_comp_id="9IG", chain_id="Q, R")
        ]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        assert row["Name"] == "9IG"
        # label order is "EA, T" (chain R's copy first); residue tokens follow in
        # lockstep, each carrying that copy's own author chain prefix (R, then Q) --
        # the REVERSE of the AI chain_id order "Q, R", proving the chain is sourced
        # per-copy from the instance list, not zipped against the AI chain_id.
        assert row["label_asym_id"] == "EA, T"
        assert row["Residue_seq_id"] == "R:1011, Q:1010"

    def test_high_cardinality_multi_copy(self, sample_pdb_data):
        # 9AYF models calcium eight times across two author chains (Q and R), so the
        # bare residue numbers repeat (1006-1009 on each chain) and are ambiguous on
        # their own. The chain prefix makes every token unique. The instance list is
        # sorted by label_asym_id (AA, BA, CA on R; then P, Q, R, S on Q; then Z on
        # R), so the emitted order is NOT grouped by chain -- the prefix is what
        # disambiguates the two "1006"s, the two "1007"s, etc.
        instances = [
            {"auth_asym_id": "R", "label_asym_id": "AA", "auth_seq_id": "1007"},
            {"auth_asym_id": "R", "label_asym_id": "BA", "auth_seq_id": "1008"},
            {"auth_asym_id": "R", "label_asym_id": "CA", "auth_seq_id": "1009"},
            {"auth_asym_id": "Q", "label_asym_id": "P", "auth_seq_id": "1006"},
            {"auth_asym_id": "Q", "label_asym_id": "Q", "auth_seq_id": "1007"},
            {"auth_asym_id": "Q", "label_asym_id": "R", "auth_seq_id": "1008"},
            {"auth_asym_id": "Q", "label_asym_id": "S", "auth_seq_id": "1009"},
            {"auth_asym_id": "R", "label_asym_id": "Z", "auth_seq_id": "1006"},
        ]
        sample_pdb_data["oligomer_analysis"] = {"nonpolymer_instance_index": {"CA": instances}}
        sample_pdb_data["ligands"] = [self._ligand(name="Calcium ion", chem_comp_id="CA")]
        row = transform_for_csv("TEST1", sample_pdb_data)["ligands.csv"][0]
        labels = row["label_asym_id"].split(", ")
        residues = row["Residue_seq_id"].split(", ")
        assert len(labels) == 8
        assert len(residues) == 8
        # Residue tokens stay 1:1 and in the same order as label_asym_id.
        assert labels == ["AA", "BA", "CA", "P", "Q", "R", "S", "Z"]
        assert residues == [
            "R:1007",
            "R:1008",
            "R:1009",
            "Q:1006",
            "Q:1007",
            "Q:1008",
            "Q:1009",
            "R:1006",
        ]
        # The repeating bare number "1006" is disambiguated by its chain prefix.
        assert "Q:1006" in residues
        assert "R:1006" in residues


def test_transform_skips_non_dict_ligand():
    """A non-dict ligand entry must be skipped, not crash the whole transform."""
    data = {"ligands": ["bogus-string", {"chem_comp_id": "ATP", "chain_id": "A"}]}
    result = transform_for_csv("X1", data)  # must not raise
    # The bogus string is skipped; the one valid ligand still produces a row.
    assert len(result["ligands.csv"]) == 1


def test_pubchem_none_sentinel_blanked():
    """The schema's literal "None" pubchem_id must become a blank PubChemID
    column, not the string 'None'; a real CID is preserved."""
    data = {
        "ligands": [
            {"name": "A", "chem_comp_id": "ATP", "chain_id": "A", "pubchem_id": "None"},
            {"name": "B", "chem_comp_id": "GDP", "chain_id": "B", "pubchem_id": "271"},
        ]
    }
    rows = transform_for_csv("X1", data)["ligands.csv"]
    assert rows[0]["PubChemID"] == ""
    assert rows[1]["PubChemID"] == "271"


def test_append_to_csvs_upserts_by_pdb(configure_paths):
    """Re-curating a PDB replaces its rows instead of appending duplicates;
    other PDBs are preserved."""
    from gpcr_tools.config import CSV_SCHEMA, get_config

    fields = CSV_SCHEMA["structures.csv"]
    pdb_col = fields[0]

    def _row(pdb: str) -> dict[str, str]:
        return {f: (pdb if f == pdb_col else "x") for f in fields}

    def _read() -> list[dict[str, str]]:
        path = get_config().csv_output_dir / "structures.csv"
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    append_to_csvs({"structures.csv": [_row("AAA")]})
    append_to_csvs({"structures.csv": [_row("AAA")]})  # re-curate same PDB
    assert sum(1 for r in _read() if r[pdb_col] == "AAA") == 1

    append_to_csvs({"structures.csv": [_row("BBB")]})  # a different PDB
    assert {r[pdb_col] for r in _read()} == {"AAA", "BBB"}
