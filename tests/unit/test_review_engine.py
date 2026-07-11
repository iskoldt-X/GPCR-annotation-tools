"""Tests for the review engine — pure logic, no interactive prompts.

Tests focus on controversy detection, auto-resolve, and significance checks.
"""

import pytest

from gpcr_tools import config
from gpcr_tools.csv_generator.exceptions import ReviewAbortedError
from gpcr_tools.csv_generator.review_engine import (
    _list_item_path,
    _resolve_list_key_field,
    get_verified_paths,
    has_downstream_controversy,
    has_gating_controversy,
    is_controversy_significant,
    review_leaf,
    review_node,
    review_toplevel_blocks,
)


class TestListKeyFieldResolution:
    @pytest.mark.parametrize("segment,field", list(config.LIST_ITEM_KEY_FIELDS.items()))
    def test_resolves_from_shared_config(self, segment, field):
        # List-item review keys must come from the shared config map (not a
        # private hardcoded copy) so review paths line up with vote aggregation.
        assert _resolve_list_key_field(f"root.{segment}.child") == field

    def test_follows_config_changes(self, monkeypatch):
        # Proves resolution reads config rather than a hardcoded literal.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.LIST_ITEM_KEY_FIELDS",
            {"widgets": "widget_id"},
        )
        assert _resolve_list_key_field("root.widgets.x") == "widget_id"
        assert _resolve_list_key_field("root.ligands.x") is None

    def test_none_for_unknown_path(self):
        assert _resolve_list_key_field("root.unknown.child") is None


class TestListItemPathMatchesAggregatorIdentity:
    """The review navigation path for a list item must equal the identity vote
    aggregation stored its controversy/flag under — otherwise the curator never
    sees contested keyless calls (protein/Apo ligands with chem_comp_id="None").
    """

    def test_keyless_ligand_path_matches_aggregator(self):
        # chem_comp_id="None" is the placeholder the schema injects for protein
        # and Apo ligands — the routine GPCR case, not an edge case.
        item = {"chem_comp_id": "None", "name": "GLP-1"}
        review_path = _list_item_path("ligands", item, "chem_comp_id", 0)
        agg_path = f"ligands[{config.list_item_identity(item, 'chem_comp_id', 0)}]"
        assert review_path == agg_path
        # The keyless name is SAFE-normalized (casefold + separator collapse), so
        # review navigation and aggregation share the same normalized identity.
        assert review_path == "ligands[__keyless__:glp 1]"

    def test_real_key_path_unchanged(self):
        item = {"chem_comp_id": "ATP", "name": "x"}
        assert _list_item_path("ligands", item, "chem_comp_id", 0) == "ligands[ATP]"

    def test_non_dict_item_uses_index(self):
        assert (
            _list_item_path("ligands[X].synonyms", "a-string", "chem_comp_id", 2)
            == "ligands[X].synonyms[2]"
        )


class TestHasDownstreamControversy:
    def test_empty_prefix_with_controversies(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("", controversies) is True

    def test_empty_prefix_no_controversies(self):
        assert has_downstream_controversy("", {}) is False

    def test_exact_match(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("receptor_info.chain_id", controversies) is True

    def test_nested_match(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("receptor_info", controversies) is True

    def test_no_match(self):
        controversies = {"receptor_info.chain_id": {}}
        assert has_downstream_controversy("signaling_partners", controversies) is False

    def test_list_path(self):
        controversies = {"ligands[ADN].role.value": {}}
        assert has_downstream_controversy("ligands", controversies) is True
        assert has_downstream_controversy("ligands[ADN]", controversies) is True
        assert has_downstream_controversy("ligands[ZMA]", controversies) is False


class TestHasGatingControversy:
    """The one-click accept-all gate is driven by this helper. Minority-omission
    advisories (gating=False) stay in the controversy map for display but must
    not disable accept-all; near-tie / real disagreements still gate."""

    def _omission_advisory(self, path: str) -> dict:
        # Same shape voting.py emits for an entity the chosen run omitted.
        return {
            "path": path,
            "best_run_value": None,
            "majority_vote_value": {"chem_comp_id": "RET"},
            "all_votes": {"role": {"agonist": 2}},
            "needs_review": True,
            "gating": False,
        }

    def _near_tie(self, path: str) -> dict:
        # Same shape voting.py emits for a near-tie disagreement (no gating key).
        return {
            "path": path,
            "best_run_value": "agonist",
            "majority_vote_value": "antagonist",
            "all_votes": {"agonist": 5, "antagonist": 5},
            "needs_review": True,
            "vote_margin": 0,
        }

    def test_empty_map_does_not_gate(self):
        assert has_gating_controversy({}) is False

    def test_only_omission_advisories_do_not_gate(self):
        # A PDB whose sole controversy is a minority-omission advisory must keep
        # one-click accept-all enabled.
        controversies = {c["path"]: c for c in [self._omission_advisory("ligands[RET]")]}
        assert has_gating_controversy(controversies) is False

    def test_near_tie_still_gates(self):
        # A genuine near-tie disagreement disables accept-all exactly as before.
        controversies = {c["path"]: c for c in [self._near_tie("ligands[ADN].role.value")]}
        assert has_gating_controversy(controversies) is True

    def test_mixed_gates_on_the_near_tie(self):
        controversies = {
            c["path"]: c
            for c in [
                self._omission_advisory("ligands[RET]"),
                self._near_tie("ligands[ADN].role.value"),
            ]
        }
        # The advisory is still present (visible for review)...
        assert "ligands[RET]" in controversies
        # ...but the near-tie is what gates.
        assert has_gating_controversy(controversies) is True


class TestIsControversySignificant:
    def test_trivial_only(self):
        """Controversies in AUTO_RESOLVE_KEYS should be non-significant."""
        controversies = {"receptor_info.confidence": {}}
        validation_data = {"critical_warnings": [], "algo_conflicts": []}
        assert is_controversy_significant("receptor_info", controversies, validation_data) is False

    def test_significant_key(self):
        """Controversies in non-trivial keys should be significant."""
        controversies = {"receptor_info.chain_id": {}}
        validation_data = {"critical_warnings": [], "algo_conflicts": []}
        assert is_controversy_significant("receptor_info", controversies, validation_data) is True

    def test_validation_warning_makes_significant(self):
        """Even trivial keys become significant with validation warnings."""
        controversies = {"receptor_info.confidence": {}}
        validation_data = {
            "critical_warnings": ["Ghost Chain at 'receptor_info': 'Z' not in PDB Source."],
            "algo_conflicts": [],
        }
        assert is_controversy_significant("receptor_info", controversies, validation_data) is True

    def test_empty_path(self):
        """Empty path prefix with non-trivial controversies."""
        controversies = {"structure_info.method": {}}
        validation_data = {"critical_warnings": [], "algo_conflicts": []}
        assert is_controversy_significant("", controversies, validation_data) is True


class TestGetVerifiedPaths:
    def test_extracts_verified_fields(self):
        data = {
            "receptor_info": {
                "chain_id": "A",
                "uniprot_entry_name": "test_human",
                "_verified_fields": ["chain_id", "uniprot_entry_name"],
            },
            "structure_info": {"method": "ELECTRON MICROSCOPY"},
        }
        verified = get_verified_paths(data)
        assert "receptor_info.chain_id" in verified
        assert "receptor_info.uniprot_entry_name" in verified
        assert len(verified) == 2

    def test_no_verified_fields(self):
        data = {"structure_info": {"method": "ELECTRON MICROSCOPY"}}
        verified = get_verified_paths(data)
        assert len(verified) == 0

    def test_empty_data(self):
        assert get_verified_paths({}) == set()


class TestConfidenceStyle:
    def test_high_is_success(self):
        from gpcr_tools.csv_generator.review_engine import _confidence_style

        assert _confidence_style("High") == "success"

    def test_low_is_warning(self):
        from gpcr_tools.csv_generator.review_engine import _confidence_style

        assert _confidence_style("Low") == "warning"

    def test_medium_is_warning(self):
        from gpcr_tools.csv_generator.review_engine import _confidence_style

        assert _confidence_style("Medium") == "warning"


class _ScriptedResponses:
    """Deterministic stand-in for a monkeypatched Prompt.ask / Confirm.ask.

    Each call pops the next scripted response; raises if the queue runs dry.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)

    def __call__(self, *args, **kwargs):
        assert self._responses, f"scripted responses exhausted: args={args}, kwargs={kwargs}"
        return self._responses.pop(0)


class TestNullLeafAcceptedNotAborted:
    """A JSON ``null`` leaf accepted during review must survive as ``None`` in the
    result, never be mistaken for the 'q' quit signal. Quitting raises
    ``ReviewAbortedError`` (a control-flow signal) so the two can no longer collide
    on a bare ``return None``.
    """

    @pytest.fixture(autouse=True)
    def _silence_audit(self, monkeypatch):
        # These tests assert value-shape preservation, not audit content. The
        # accept path calls log_audit_trail -> get_config() (default workspace
        # /workspace), so no-op it to keep the tests hermetic and off any real
        # workspace's audit trail.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.log_audit_trail",
            lambda *a, **k: None,
        )

    def test_deep_review_dict_preserves_null_leaf(self, monkeypatch):
        # A non-blacklisted leaf whose value is null, accepted with "y", must be
        # kept as None in the returned dict — not collapse the whole node to None.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses(["y"]),
        )
        node = {"annotation_note": None}
        result = review_node(
            "XXXX", node, {}, path="key_findings", force_deep=True, validation_data={}
        )
        assert result == {"annotation_note": None}
        assert result["annotation_note"] is None

    def test_deep_review_list_row_preserves_null_soft_fields(self, monkeypatch):
        # A ligand row carrying the site-undetermined "unknown" shape (soft
        # fields left null == "not assessed"), accepted leaf-by-leaf, must be
        # preserved intact rather than aborting the review at the first null.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses(["y", "y", "y"]),
        )
        ligands = [
            {
                "chem_comp_id": "CA",
                "site_ref_justification": None,
                "pharmacological_role_check": None,
            }
        ]
        result = review_node(
            "XXXX", ligands, {}, path="ligands", force_deep=True, validation_data={}
        )
        assert result == [
            {
                "chem_comp_id": "CA",
                "site_ref_justification": None,
                "pharmacological_role_check": None,
            }
        ]
        assert result[0]["site_ref_justification"] is None
        assert result[0]["pharmacological_role_check"] is None

    def test_review_leaf_quit_raises(self, monkeypatch):
        # The genuine quit path is now an exception, not a None return.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses(["q"]),
        )
        with pytest.raises(ReviewAbortedError):
            review_leaf("XXXX", "some-value", {}, "receptor_info.chain_id", {})

    def test_toplevel_quit_at_leaf_raises(self, monkeypatch):
        # Declining a clean block enters deep review; quitting at a leaf aborts
        # the whole review via ReviewAbortedError rather than returning None.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            _ScriptedResponses([False]),
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses(["q"]),
        )
        main_data = {"structure_info": {"method": "X-RAY DIFFRACTION"}}
        with pytest.raises(ReviewAbortedError):
            review_toplevel_blocks("XXXX", main_data, {}, {})


class TestLigandCopiesReviewable:
    """The per-copy binding-site sidecar (``ligand_copies``) is reviewed as a
    top-level block, so the review walker now descends into it: a contested
    per-copy ``site_ref`` is reachable and the curator's choice is captured in the
    returned data; a null soft field survives an accept; and a clean block ships
    verbatim (or auto-accepts in fix mode) without a spurious prompt.
    """

    @pytest.fixture(autouse=True)
    def _silence_audit(self, monkeypatch):
        # The accept/edit paths call log_audit_trail -> get_config() (default
        # workspace), so no-op it to keep the tests hermetic and off any real
        # workspace's audit trail. (Same pattern as TestNullLeafAcceptedNotAborted.)
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.log_audit_trail",
            lambda *a, **k: None,
        )

    def test_contested_site_ref_reachable_and_edit_captured(self, monkeypatch):
        # A per-copy site_ref where the best run disagrees with the majority vote --
        # the same vote-controversy shape aggregation records for a real
        # disagreement (best_run_value != majority_vote_value). Because the values
        # differ, the contested leaf is offered with NO pre-selected default, so a
        # bare Enter cannot commit it and the curator must choose explicitly.
        controversies = {
            "ligand_copies[R:602].site_ref": {
                "path": "ligand_copies[R:602].site_ref",
                "best_run_value": "orthosteric",
                "majority_vote_value": "intracellular",
                "all_votes": {"intracellular": 3, "orthosteric": 2},
            }
        }
        main_data = {
            "ligand_copies": [
                {"copy_id": "R:602", "site_ref": "orthosteric"},
            ]
        }
        # Scripted interaction, in order:
        #   "r" -> at the ligand_copies block, choose to review it (not accept-all).
        #   "y" -> accept the clean copy_id leaf as-is.
        #   "1" -> at the site_ref CONTROVERSY, pick option 1. Candidates sort by
        #          vote count, so the 3-vote "intracellular" is option 1; selecting
        #          it changes the copy's site_ref away from the original
        #          "orthosteric" -- a deliberate choice, since the differing values
        #          suppress the default.
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses(["r", "y", "1"]),
        )
        # Completing the call (no ReviewAbortedError) is itself the "did not abort"
        # assertion; the edited value proves the walker reached the contested leaf.
        final_data = review_toplevel_blocks("XXXX", main_data, controversies, {})
        assert final_data["ligand_copies"] == [{"copy_id": "R:602", "site_ref": "intracellular"}]
        assert final_data["ligand_copies"][0]["site_ref"] == "intracellular"

    def test_null_soft_field_row_accepted_returns_dict_not_abort(self, monkeypatch):
        # A per-copy row whose soft fields are null (role/evidence == None, the
        # "not assessed" shape) must survive a deep review: a null leaf accepted
        # with "y" is kept as None, never mistaken for the quit signal. Guards that
        # the null-vs-quit fix covers this now-reachable block.
        main_data = {
            "ligand_copies": [
                {"copy_id": "R:602", "role": None, "evidence": None},
            ]
        }
        # "no" declines the clean-block accept, entering a deep review of every
        # leaf; then "y", "y" accept the copy_id and the null role leaf. (evidence
        # is a soft/blacklisted field and passes through untouched, no prompt.)
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            _ScriptedResponses([False]),
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses(["y", "y"]),
        )
        final_data = review_toplevel_blocks("XXXX", main_data, {}, {})
        assert final_data["ligand_copies"] == [{"copy_id": "R:602", "role": None, "evidence": None}]
        assert final_data["ligand_copies"][0]["role"] is None

    def test_clean_block_ships_verbatim_when_accepted(self, monkeypatch):
        # A clean block (no controversy, no validation alert) offers one "Accept?"
        # confirm; accepting ships it verbatim, with no per-leaf prompt.
        block = [{"copy_id": "R:602", "site_ref": "orthosteric"}]
        main_data = {"ligand_copies": [dict(block[0])]}
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            _ScriptedResponses([True]),
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses([]),  # asserts if any per-leaf prompt fires
        )
        final_data = review_toplevel_blocks("XXXX", main_data, {}, {})
        assert final_data["ligand_copies"] == block

    def test_clean_block_auto_accepts_in_fix_mode_without_prompt(self, monkeypatch):
        # In fix mode a clean block auto-accepts silently: NEITHER a confirm NOR a
        # per-leaf prompt should fire (both scripted empty -> assert if called).
        block = [{"copy_id": "R:602", "site_ref": "orthosteric"}]
        main_data = {"ligand_copies": [dict(block[0])]}
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Confirm.ask",
            _ScriptedResponses([]),
        )
        monkeypatch.setattr(
            "gpcr_tools.csv_generator.review_engine.Prompt.ask",
            _ScriptedResponses([]),
        )
        final_data = review_toplevel_blocks("XXXX", main_data, {}, {}, fix_mode=True)
        assert final_data["ligand_copies"] == block
