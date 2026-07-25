"""Tests for the shared gating predicate (validator/gating.py).

Covers the two pure functions that decide whether a PDB needs human review,
from one place: ``oligomer_gating_warnings`` (the curator-facing strings) and
``is_pdb_gated`` (the all-sources truth table). The strings must match what the
interactive curator surfaces; absence of any source must contribute nothing.
"""

from __future__ import annotations

from gpcr_tools.config import (
    ALERT_ASSEMBLY_MISMATCH,
    ALERT_CONFIRMED_OLIGOMER,
    ALERT_HALLUCINATION,
    ALERT_MISSED_PROTOMER,
    ALERT_MULTI_COPY_LIGAND,
    ALERT_NO_GPCR,
    ALERT_NON_RECEPTOR_PARTNER,
    ALERT_OLIGOMER_DISAGREEMENT,
    ALERT_SUSPICIOUS_7TM,
    ALERT_TM_DATA_UNAVAILABLE,
    TM_STATUS_INCOMPLETE,
)
from gpcr_tools.validator.gating import (
    has_gating_controversy,
    is_pdb_gated,
    oligomer_gating_warnings,
)

# ── oligomer_gating_warnings ────────────────────────────────────────────


class TestOligomerGatingWarnings:
    def test_none_and_empty_return_empty(self):
        assert oligomer_gating_warnings(None) == []
        assert oligomer_gating_warnings({}) == []

    def test_chain_id_override(self):
        oligo = {
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "G",
                "corrected_chain_id": "R",
                "trigger": "HALLUCINATION",
            },
            "alerts": [],
            "all_gpcr_chains": [],
        }
        out = oligomer_gating_warnings(oligo)
        assert out == [
            "CHAIN_ID CORRECTED at 'receptor_info': G -> R (HALLUCINATION). "
            "Human confirmation required."
        ]

    def test_each_promote_set_type_with_prefixed_message(self):
        """Every gating oligomer type produces one receptor_info alert with a
        single [TYPE] prefix preserved (message already carries it)."""
        for atype in (
            ALERT_HALLUCINATION,
            ALERT_MISSED_PROTOMER,
            ALERT_SUSPICIOUS_7TM,
            ALERT_NO_GPCR,
            ALERT_TM_DATA_UNAVAILABLE,
            ALERT_OLIGOMER_DISAGREEMENT,
            ALERT_NON_RECEPTOR_PARTNER,
        ):
            msg = f"[{atype}] at 'oligomer_analysis': something to confirm"
            oligo = {
                "chain_id_override": {"applied": False},
                "alerts": [{"type": atype, "message": msg}],
                "all_gpcr_chains": [],
            }
            out = oligomer_gating_warnings(oligo)
            assert len(out) == 1, atype
            assert out[0] == f"OLIGOMER ALERT at 'receptor_info': {msg}"
            # The type label appears exactly once, never doubled.
            assert out[0].count(f"[{atype}]") == 1

    def test_bare_message_gets_prefix_added(self):
        """An older record with a bare (un-prefixed) message gets [TYPE] added
        exactly once -- not dropped, not duplicated."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_MISSED_PROTOMER, "message": "Missed chain B"}],
            "all_gpcr_chains": [],
        }
        out = oligomer_gating_warnings(oligo)
        assert out == [
            f"OLIGOMER ALERT at 'receptor_info': [{ALERT_MISSED_PROTOMER}] Missed chain B"
        ]

    def test_multi_copy_ligand_verbatim_keeps_ligand_path(self):
        """MULTI_COPY_LIGAND is promoted verbatim, keeping its own ligands[...]
        path so it buckets with the ligand block, not receptor_info."""
        msg = (
            "[MULTI_COPY_LIGAND] at 'ligands[CA]': modelled in 2 copies "
            "(instances D, E); one annotation row may hide copies. Human review recommended."
        )
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_MULTI_COPY_LIGAND, "message": msg}],
            "all_gpcr_chains": [],
        }
        out = oligomer_gating_warnings(oligo)
        assert out == [msg]
        assert "ligands[CA]" in out[0]
        assert "receptor_info" not in out[0]

    def test_multi_copy_ligand_advisory_flag_does_not_gate(self):
        """A MULTI_COPY_LIGAND alert whose copies share one site is stamped
        ``gating=False`` by the aggregator and is not surfaced as a gating warning."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {
                    "type": ALERT_MULTI_COPY_LIGAND,
                    "message": "[MULTI_COPY_LIGAND] at 'ligands[CLR]': modelled in 3 copies",
                    "gating": False,
                }
            ],
            "all_gpcr_chains": [],
        }
        assert oligomer_gating_warnings(oligo) == []

    def test_multi_copy_ligand_gating_flag_true_surfaces(self):
        msg = "[MULTI_COPY_LIGAND] at 'ligands[BU1]': modelled in 2 copies"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_MULTI_COPY_LIGAND, "message": msg, "gating": True}],
            "all_gpcr_chains": [],
        }
        assert oligomer_gating_warnings(oligo) == [msg]

    def test_multi_copy_ligand_absent_flag_defaults_to_gating(self):
        """A back-catalogue alert recorded before the flag existed still gates
        (the flag defaults to True)."""
        msg = "[MULTI_COPY_LIGAND] at 'ligands[PLM]'"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_MULTI_COPY_LIGAND, "message": msg}],
            "all_gpcr_chains": [],
        }
        assert oligomer_gating_warnings(oligo) == [msg]

    def test_multi_copy_mixed_only_gating_one_surfaces(self):
        """One advisory + one gating multi-copy alert: only the gating one is
        surfaced; the advisory one is dropped."""
        gating_msg = "[MULTI_COPY_LIGAND] at 'ligands[BU1]': divergent"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {
                    "type": ALERT_MULTI_COPY_LIGAND,
                    "message": "[MULTI_COPY_LIGAND] at 'ligands[CLR]': shared",
                    "gating": False,
                },
                {"type": ALERT_MULTI_COPY_LIGAND, "message": gating_msg, "gating": True},
            ],
            "all_gpcr_chains": [],
        }
        assert oligomer_gating_warnings(oligo) == [gating_msg]

    def test_incomplete_7tm_chain(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [{"chain_id": "A", "7tm_status": TM_STATUS_INCOMPLETE}],
        }
        out = oligomer_gating_warnings(oligo)
        assert len(out) == 1
        assert "INCOMPLETE 7TM" in out[0]
        assert out[0].startswith("STRUCTURAL QUALITY at 'receptor_info':")

    def test_confirmed_oligomer_and_assembly_mismatch_do_not_gate(self):
        """CONFIRMED_OLIGOMER (roster matched) and ASSEMBLY_MISMATCH (advisory
        confirm note) are informational -- neither produces a gating warning."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {"type": ALERT_CONFIRMED_OLIGOMER, "message": "[CONFIRMED_OLIGOMER] all matched"},
                {"type": ALERT_ASSEMBLY_MISMATCH, "message": "[ASSEMBLY_MISMATCH] confirm"},
            ],
            "all_gpcr_chains": [{"chain_id": "A", "7tm_status": "COMPLETE"}],
        }
        assert oligomer_gating_warnings(oligo) == []

    def test_oligomer_disagreement_advisory_flag_does_not_gate(self):
        """An OLIGOMER_DISAGREEMENT stamped gating=False by the aggregator (AI
        undercounts the receptor and RCSB's assembly shows no homo-oligomer) is
        surfaced elsewhere but is not a gating warning."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {
                    "type": ALERT_OLIGOMER_DISAGREEMENT,
                    "message": "[OLIGOMER_DISAGREEMENT] at 'receptor_info': confirm",
                    "gating": False,
                }
            ],
            "all_gpcr_chains": [],
        }
        assert oligomer_gating_warnings(oligo) == []

    def test_oligomer_disagreement_gating_flag_true_surfaces(self):
        msg = "[OLIGOMER_DISAGREEMENT] at 'receptor_info': confirm"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_OLIGOMER_DISAGREEMENT, "message": msg, "gating": True}],
            "all_gpcr_chains": [],
        }
        out = oligomer_gating_warnings(oligo)
        assert out == [f"OLIGOMER ALERT at 'receptor_info': {msg}"]

    def test_oligomer_disagreement_absent_flag_defaults_to_gating(self):
        """A back-catalogue OD alert recorded before the gating flag existed still
        gates -- the flag defaults to True."""
        msg = "[OLIGOMER_DISAGREEMENT] at 'receptor_info': confirm"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_OLIGOMER_DISAGREEMENT, "message": msg}],
            "all_gpcr_chains": [],
        }
        out = oligomer_gating_warnings(oligo)
        assert out == [f"OLIGOMER ALERT at 'receptor_info': {msg}"]

    def test_order_is_override_then_alerts_then_7tm(self):
        oligo = {
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "X",
                "corrected_chain_id": "A",
                "trigger": "7TM_UPGRADE",
            },
            "alerts": [
                {
                    "type": ALERT_MISSED_PROTOMER,
                    "message": f"[{ALERT_MISSED_PROTOMER}] at 'oligomer_analysis': Missed B",
                }
            ],
            "all_gpcr_chains": [{"chain_id": "B", "7tm_status": TM_STATUS_INCOMPLETE}],
        }
        out = oligomer_gating_warnings(oligo)
        assert len(out) == 3
        assert out[0].startswith("CHAIN_ID CORRECTED")
        assert out[1].startswith("OLIGOMER ALERT")
        assert out[2].startswith("STRUCTURAL QUALITY")


# ── has_gating_controversy ──────────────────────────────────────────────


class TestHasGatingControversy:
    def test_none_and_empty(self):
        assert has_gating_controversy(None) is False
        assert has_gating_controversy({}) is False

    def test_default_gates_when_flag_absent(self):
        # A voting-log record with no explicit ``gating`` flag defaults to gating.
        assert has_gating_controversy({"p": {"path": "p"}}) is True

    def test_minority_omission_does_not_gate(self):
        assert has_gating_controversy({"p": {"gating": False}}) is False

    def test_mixed_gates_if_any_gates(self):
        controversies = {"a": {"gating": False}, "b": {"gating": True}}
        assert has_gating_controversy(controversies) is True


# ── is_pdb_gated truth table ────────────────────────────────────────────


class TestIsPdbGated:
    def test_all_empty_or_absent_not_gated(self):
        assert is_pdb_gated(None, None, None) is False
        assert is_pdb_gated({}, {}, {}) is False
        assert (
            is_pdb_gated({"critical_warnings": [], "algo_conflicts": []}, {"alerts": []}, {})
            is False
        )

    def test_validation_only_gates(self):
        assert is_pdb_gated({"critical_warnings": ["x"]}, None, None) is True
        assert is_pdb_gated({"algo_conflicts": ["x"]}, None, None) is True

    def test_oligomer_only_gates_4zwj_shape(self):
        """4ZWJ shape: empty validation log, OLIGOMER_DISAGREEMENT alert."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {"type": ALERT_CONFIRMED_OLIGOMER, "message": "[CONFIRMED_OLIGOMER] matched"},
                {"type": ALERT_ASSEMBLY_MISMATCH, "message": "[ASSEMBLY_MISMATCH] confirm"},
                {"type": ALERT_OLIGOMER_DISAGREEMENT, "message": "[OLIGOMER_DISAGREEMENT] confirm"},
            ],
            "all_gpcr_chains": [{"7tm_status": "COMPLETE"}],
        }
        assert is_pdb_gated({"critical_warnings": [], "algo_conflicts": []}, oligo, {}) is True

    def test_oligomer_only_gates_2g87_shape(self):
        """2G87 shape: empty validation log, MULTI_COPY_LIGAND alerts."""
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {"type": ALERT_CONFIRMED_OLIGOMER, "message": "[CONFIRMED_OLIGOMER] matched"},
                {
                    "type": ALERT_MULTI_COPY_LIGAND,
                    "message": "[MULTI_COPY_LIGAND] at 'ligands[PLM]'",
                },
            ],
            "all_gpcr_chains": [{"7tm_status": "COMPLETE"}],
        }
        assert is_pdb_gated({}, oligo, {}) is True

    def test_voting_only_gates_default_record(self):
        # A controversy record with no gating flag (default True) gates alone.
        assert is_pdb_gated({}, None, {"p": {"path": "p"}}) is True

    def test_minority_omission_alone_does_not_gate(self):
        assert is_pdb_gated({}, None, {"p": {"gating": False}}) is False

    def test_non_gating_oligomer_alerts_alone_do_not_gate(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {"type": ALERT_CONFIRMED_OLIGOMER, "message": "[CONFIRMED_OLIGOMER] matched"},
                {"type": ALERT_ASSEMBLY_MISMATCH, "message": "[ASSEMBLY_MISMATCH] confirm"},
            ],
            "all_gpcr_chains": [{"7tm_status": "COMPLETE"}],
        }
        assert is_pdb_gated({}, oligo, {}) is False
