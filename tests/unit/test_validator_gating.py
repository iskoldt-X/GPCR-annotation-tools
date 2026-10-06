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
        assert out[0] == msg
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
        assert oligomer_gating_warnings(oligo)[0] == msg

    def test_multi_copy_ligand_absent_flag_defaults_to_gating(self):
        """A back-catalogue alert recorded before the flag existed still gates
        (the flag defaults to True)."""
        msg = "[MULTI_COPY_LIGAND] at 'ligands[PLM]'"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_MULTI_COPY_LIGAND, "message": msg}],
            "all_gpcr_chains": [],
        }
        assert oligomer_gating_warnings(oligo)[0] == msg

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
        out = oligomer_gating_warnings(oligo)
        assert out[0] == gating_msg
        # Nothing from the advisory alert reaches the curator: neither its own
        # compound-level line nor a per-copy mirror of it.
        assert not any("CLR" in w for w in out)

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


# ── multi-copy ligand: per-copy mirror ──────────────────────────────────


def _multi_copy_oligo(message: str, gating: bool | None = None) -> dict:
    """An oligomer analysis carrying a single MULTI_COPY_LIGAND alert.

    ``gating=None`` omits the flag entirely, the shape of a record written before
    the flag existed.
    """
    alert: dict = {"type": ALERT_MULTI_COPY_LIGAND, "message": message}
    if gating is not None:
        alert["gating"] = gating
    return {
        "chain_id_override": {"applied": False},
        "alerts": [alert],
        "all_gpcr_chains": [],
    }


class TestMultiCopyLigandPerCopyMirror:
    """A multi-copy ligand alert must be reachable from the per-copy ligand table.

    The alert is raised against the compound (``ligands[<comp>]``), but a compound
    row has no field for a per-copy binding site: the copies are only separable in
    the ``ligand_copies`` table, which is where a wrong site assignment is actually
    corrected. So a gating alert is emitted twice -- once at the compound anchor and
    once re-anchored at the per-copy table. Copies that share a single site
    (``gating=False``) need no per-copy decision and get no mirror.
    """

    COMPOUND_MESSAGE = (
        "[MULTI_COPY_LIGAND] at 'ligands[CLR]': modelled in 2 copies (instances D, E); "
        "one annotation row may hide copies at distinct sites or with distinct roles. "
        "Human review recommended."
    )

    def test_gating_alert_emits_compound_and_per_copy_anchors(self):
        out = oligomer_gating_warnings(_multi_copy_oligo(self.COMPOUND_MESSAGE))
        assert len(out) == 2
        # The compound-level warning is untouched: the mirror is additional.
        assert out[0] == self.COMPOUND_MESSAGE
        assert out[1].startswith("[MULTI_COPY_LIGAND] at 'ligand_copies'")

    def test_mirror_keeps_the_type_label_exactly_once(self):
        mirror = oligomer_gating_warnings(_multi_copy_oligo(self.COMPOUND_MESSAGE))[1]
        assert mirror.count(f"[{ALERT_MULTI_COPY_LIGAND}]") == 1

    def test_mirror_names_the_component_and_keeps_the_finding(self):
        """The curator opening the per-copy table must still learn which compound
        is multi-copy and what the finding was."""
        mirror = oligomer_gating_warnings(_multi_copy_oligo(self.COMPOUND_MESSAGE))[1]
        assert "component CLR" in mirror
        assert "modelled in 2 copies (instances D, E)" in mirror
        assert "Human review recommended." in mirror

    def test_mirror_drops_the_compound_anchor(self):
        """The mirror must not carry a second ``ligands[...]`` anchor, or it would
        also be pulled into the compound-level block as a near-duplicate line."""
        mirror = oligomer_gating_warnings(_multi_copy_oligo(self.COMPOUND_MESSAGE))[1]
        assert "ligands[" not in mirror
        assert "ligands" not in mirror

    def test_advisory_alert_emits_no_mirror(self):
        """Copies sharing one binding site are advisory: no compound warning and no
        per-copy mirror, so the per-copy table is not opened for nothing."""
        oligo = _multi_copy_oligo(self.COMPOUND_MESSAGE, gating=False)
        assert oligomer_gating_warnings(oligo) == []

    def test_absent_gating_flag_defaults_to_mirroring(self):
        """A back-catalogue alert recorded before the gating flag existed gates, so
        it is mirrored too rather than silently waved past the per-copy table."""
        out = oligomer_gating_warnings(_multi_copy_oligo(self.COMPOUND_MESSAGE))
        assert len(out) == 2

    def test_bare_back_catalogue_message_is_anchored_and_labelled(self):
        """An older record stored without a [TYPE] label or a compound anchor still
        yields a mirror the per-copy table can be reached by."""
        oligo = _multi_copy_oligo("modelled in 3 copies")
        mirror = oligomer_gating_warnings(oligo)[1]
        assert mirror == "[MULTI_COPY_LIGAND] at 'ligand_copies': modelled in 3 copies"

    def test_one_mirror_per_gating_component(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {
                    "type": ALERT_MULTI_COPY_LIGAND,
                    "message": "[MULTI_COPY_LIGAND] at 'ligands[CLR]': modelled in 2 copies",
                },
                {
                    "type": ALERT_MULTI_COPY_LIGAND,
                    "message": "[MULTI_COPY_LIGAND] at 'ligands[NA]': modelled in 4 copies",
                    "gating": False,
                },
                {
                    "type": ALERT_MULTI_COPY_LIGAND,
                    "message": "[MULTI_COPY_LIGAND] at 'ligands[BU1]': modelled in 3 copies",
                    "gating": True,
                },
            ],
            "all_gpcr_chains": [],
        }
        mirrors = [w for w in oligomer_gating_warnings(oligo) if "ligand_copies" in w]
        assert len(mirrors) == 2
        assert "component CLR" in mirrors[0]
        assert "component BU1" in mirrors[1]

    def test_structure_without_multi_copy_alert_is_unchanged(self):
        """No multi-copy alert, no extra warning: an unrelated gating finding still
        produces exactly its own single receptor_info line."""
        msg = f"[{ALERT_MISSED_PROTOMER}] at 'oligomer_analysis': Missed chain B"
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [{"type": ALERT_MISSED_PROTOMER, "message": msg}],
            "all_gpcr_chains": [],
        }
        out = oligomer_gating_warnings(oligo)
        assert out == [f"OLIGOMER ALERT at 'receptor_info': {msg}"]
        assert not any("ligand_copies" in w for w in out)


class TestMultiCopyMirrorRouting:
    """The mirror must reach the per-copy block under the review UI's own path
    matching, and must not disturb where the compound-level warning lands."""

    COMPOUND_MESSAGE = TestMultiCopyLigandPerCopyMirror.COMPOUND_MESSAGE

    def _validation_data(self) -> dict:
        from gpcr_tools.csv_generator.validation_display import inject_oligomer_alerts

        oligo = _multi_copy_oligo(self.COMPOUND_MESSAGE)
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        return validation_data

    def test_per_copy_block_now_receives_a_warning(self):
        from gpcr_tools.csv_generator.validation_display import (
            get_relevant_validation_warnings,
        )

        relevant = get_relevant_validation_warnings("ligand_copies", self._validation_data())
        assert len(relevant) == 1
        assert "component CLR" in relevant[0]

    def test_compound_block_still_receives_only_its_own_warning(self):
        from gpcr_tools.csv_generator.validation_display import (
            get_relevant_validation_warnings,
        )

        relevant = get_relevant_validation_warnings("ligands", self._validation_data())
        assert relevant == [self.COMPOUND_MESSAGE]

    def test_mirror_matches_the_per_copy_block_by_extracted_path(self):
        """The structured path parser (used for block-level impact analysis) reads
        the mirror's anchor as the per-copy block, not the compound block."""
        from gpcr_tools.csv_generator.validation_display import (
            extract_validation_entries,
            warning_matches_block,
        )

        entries = extract_validation_entries(self._validation_data())
        mirrors = [e for e in entries if "ligand_copies" in (e.get("path") or "")]
        assert len(mirrors) == 1
        assert mirrors[0]["path"] == "ligand_copies"
        assert warning_matches_block(mirrors[0], "ligand_copies")
        assert not warning_matches_block(mirrors[0], "ligands")

    def test_advisory_alert_leaves_the_per_copy_block_unrouted(self):
        from gpcr_tools.csv_generator.validation_display import (
            get_relevant_validation_warnings,
            inject_oligomer_alerts,
        )

        oligo = _multi_copy_oligo(self.COMPOUND_MESSAGE, gating=False)
        validation_data: dict = {}
        inject_oligomer_alerts(oligo, validation_data)
        assert get_relevant_validation_warnings("ligand_copies", validation_data) == []


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
