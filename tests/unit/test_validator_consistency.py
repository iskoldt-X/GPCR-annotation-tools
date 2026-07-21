"""Tests for the cross-field state/ligand consistency advisory.

Covers ``state_ligand_consistency_warnings``: the coupling-aware check that asks
a curator to confirm an active-state call when an inactive-stabilising ligand is
bound and no G protein transducer is modelled. The advisory must fire only in
that exact combination, and must not fire for antagonist/NAM ligands or when a
G protein is present.
"""

from __future__ import annotations

from gpcr_tools.config import ALERT_PREFIX_STATE_CONFIRMATION
from gpcr_tools.validator.consistency import state_ligand_consistency_warnings


def _data(state="active", role="Inverse agonist", g_protein=None, ligand_name="CGP 55845"):
    d = {
        "structure_info": {"state": {"value": state}},
        "ligands": [{"name": ligand_name, "role": {"value": role}}],
        "signaling_partners": {},
    }
    if g_protein is not None:
        d["signaling_partners"]["g_protein"] = g_protein
    return d


class TestStateLigandConsistency:
    def test_fires_active_inverse_agonist_no_g_protein(self):
        warnings = state_ligand_consistency_warnings(_data())
        assert len(warnings) == 1
        assert warnings[0].startswith(ALERT_PREFIX_STATE_CONFIRMATION)
        assert "structure_info.state" in warnings[0]
        assert "CGP 55845" in warnings[0]
        # Domain-language, light-touch: no "contradiction"/"error" wording.
        lowered = warnings[0].lower()
        assert "contradiction" not in lowered
        assert "error" not in lowered

    def test_silent_when_g_protein_present(self):
        # A G-protein-coupled inverse-agonist complex is real biology, not a flag.
        gp = {"alpha_subunit": {"uniprot_entry_name": "gnai1_human"}}
        assert state_ligand_consistency_warnings(_data(g_protein=gp)) == []

    def test_silent_when_g_protein_present_but_no_alpha(self):
        # A g_protein record with no alpha subunit is not a modelled transducer.
        assert state_ligand_consistency_warnings(_data()) != []
        assert state_ligand_consistency_warnings(_data(g_protein={"note": "only beta"})) != []

    def test_silent_for_antagonist(self):
        assert state_ligand_consistency_warnings(_data(role="Antagonist")) == []

    def test_silent_for_nam(self):
        assert state_ligand_consistency_warnings(_data(role="NAM")) == []

    def test_silent_when_state_not_active(self):
        assert state_ligand_consistency_warnings(_data(state="inactive")) == []
        assert state_ligand_consistency_warnings(_data(state="unknown")) == []

    def test_silent_when_no_ligands(self):
        d = {"structure_info": {"state": {"value": "active"}}, "ligands": []}
        assert state_ligand_consistency_warnings(d) == []

    def test_empty_and_missing_inputs_are_safe(self):
        assert state_ligand_consistency_warnings({}) == []
        assert state_ligand_consistency_warnings({"structure_info": None}) == []

    def test_role_as_plain_string(self):
        d = {
            "structure_info": {"state": {"value": "active"}},
            "ligands": [{"name": "X", "role": "Inverse agonist"}],
            "signaling_partners": {},
        }
        assert len(state_ligand_consistency_warnings(d)) == 1
