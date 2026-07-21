"""Tests for controversy default suppression in the review engine.

Safety core of the decision-first curate UI: at a genuine vote disagreement on
an identity/biology-bearing value, the review prompt must offer NO pre-selected
default, so a bare Enter cannot silently commit a value. A display-string field
(a ligand/protein name) keeps its default because a wording variant carries no
identity risk.

Fixtures are real controversy records extracted from batch voting logs:
  - an is_chimeric fork where the best run disagrees with the majority vote;
  - four per-copy ligand-role forks split evenly (margin 0, best == majority);
  - a decision-unit role value at margin 1 (best == majority);
  - a ligand molecular-type fork where the best run disagrees with the majority
    vote (small-molecule vs lipid) -- an identity-bearing flat leaf;
  - a ligand-name tie (margin 0, best == majority) as the negative case.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from gpcr_tools.aggregator.voting import flag_low_confidence_consensus
from gpcr_tools.config import LOW_CONFIDENCE_LEVELS, SEMANTIC_CONTROVERSY_KEYS
from gpcr_tools.csv_generator import review_engine
from gpcr_tools.csv_generator.review_engine import (
    _top_two_vote_margin,
    fork_requires_explicit_choice,
    review_leaf,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "controversy_default_forks.json"


def _load_forks() -> list[dict[str, Any]]:
    return json.loads(_FIXTURE.read_text())


def _terminal_key(path: str) -> str:
    """Terminal key of a controversy path, exactly as review_leaf derives it."""
    key = path.split(".")[-1]
    if "[" in key:
        key = key.split("[")[0]
    return key


def _fork(path_suffix: str) -> dict[str, Any]:
    """The single fixture record whose path ends with *path_suffix*."""
    matches = [r for r in _load_forks() if r["path"].endswith(path_suffix)]
    assert len(matches) == 1, f"expected exactly one fixture for {path_suffix}"
    return matches[0]


def _requires_explicit(record: dict[str, Any]) -> bool:
    """Run the decision on a raw voting-log record."""
    return fork_requires_explicit_choice(
        _terminal_key(record["path"]),
        record.get("best_run_value"),
        record.get("majority_vote_value"),
        list(record["all_votes"].values()),
    )


# ── _top_two_vote_margin ────────────────────────────────────────────────


class TestTopTwoVoteMargin:
    def test_even_split_is_zero(self):
        assert _top_two_vote_margin([5, 5]) == 0

    def test_lead_is_difference(self):
        assert _top_two_vote_margin([6, 4]) == 2

    def test_order_independent(self):
        assert _top_two_vote_margin([2, 3]) == 1

    def test_single_candidate_is_not_a_tie(self):
        # A lone candidate has no runner-up to tie with -> not a near-tie.
        assert _top_two_vote_margin([10]) is None

    def test_empty_is_not_a_tie(self):
        assert _top_two_vote_margin([]) is None

    def test_ignores_third_place(self):
        assert _top_two_vote_margin([5, 5, 3, 1]) == 0


# ── fork_requires_explicit_choice (pure decision) ───────────────────────


class TestSemanticForksSuppressDefault:
    """Semantic forks (a) + (b): a genuine disagreement on an identity value
    must be offered with NO default."""

    def test_is_chimeric_best_disagrees_with_majority(self):
        # best run false, majority true -> best != majority -> no default.
        record = _fork("signaling_partners.g_protein.is_chimeric")
        assert record["best_run_value"] != record["majority_vote_value"]
        assert _requires_explicit(record) is True

    @pytest.mark.parametrize("copy_id", ["Q:1006", "Q:1007", "R:1006", "R:1007"])
    def test_ligand_role_even_split(self, copy_id):
        # Agonist 5 / Co-agonist 5 -> best == majority but margin 0 -> no default.
        record = _fork(f"ligand_copies[{copy_id}].role")
        assert record["best_run_value"] == record["majority_vote_value"]
        assert _top_two_vote_margin(record["all_votes"].values()) == 0
        assert _requires_explicit(record) is True

    def test_role_value_at_near_tie_margin(self):
        # Cofactor 3 / PAM 2 -> best == majority, margin 1 (<= near-tie) -> no default.
        record = _fork("ligands[CLR:membrane_facing].role.value")
        assert record["best_run_value"] == record["majority_vote_value"]
        assert _top_two_vote_margin(record["all_votes"].values()) == 1
        assert _requires_explicit(record) is True

    def test_ligand_type_best_disagrees_with_majority(self):
        # A flat ligand molecular-type leaf: best run lipid, majority
        # small-molecule (an even 5:5 split) -> best != majority -> no default.
        # This value carries the ligand's biology, so a bare Enter must not
        # commit the majority over a real minority.
        record = _fork("ligands[K6G:intracellular].type")
        assert _terminal_key(record["path"]) == "type"
        assert record["best_run_value"] != record["majority_vote_value"]
        assert _requires_explicit(record) is True

    def test_ligand_type_disagreements_across_categories(self):
        # The same guard holds whichever molecular categories disagree: a
        # peptide/protein swap and a real ligand collapsing to none/Apo both
        # require an explicit choice rather than a pre-selected default.
        assert fork_requires_explicit_choice("type", "peptide", "protein", [2, 8]) is True
        assert fork_requires_explicit_choice("type", "none", "small-molecule", [6, 10]) is True


class TestNonSemanticForksKeepDefault:
    """Negative case (c): a display-string field keeps its default even at a
    perfect tie, and a semantic field with a comfortable lead keeps its."""

    def test_name_tie_keeps_default(self):
        # A ligand-name tie (margin 0, best == majority) is a wording variant,
        # not an identity risk -> keep the default.
        record = _fork("ligands[U0G:orthosteric].name")
        assert _terminal_key(record["path"]) == "name"
        assert record["best_run_value"] == record["majority_vote_value"]
        assert _requires_explicit(record) is False

    def test_semantic_field_with_comfortable_margin_keeps_default(self):
        # best == majority and the lead is wide -> settled -> keep the default.
        assert fork_requires_explicit_choice("role", "Agonist", "Agonist", [8, 2]) is False

    def test_pubchem_id_is_not_semantic(self):
        assert fork_requires_explicit_choice("pubchem_id", "1", "2", [5, 5]) is False


class TestUnanimousLowConfidenceKeepsDefault:
    """A unanimous but low-confidence call is surfaced for review, yet every run
    agreed: it is a consensus, not a disagreement, so it keeps its default. Such
    a record carries no per-value votes (``all_votes`` is empty), which must not
    be mistaken for a near-tie."""

    def _low_confidence_record(self) -> dict[str, Any]:
        """A real-shaped flag emitted by the aggregator for a unanimous but
        low-confidence semantic decision unit."""
        best_run_data = {
            "structure_info": {"state": {"value": "Active", "confidence": "Low", "evidence": None}}
        }
        flags = flag_low_confidence_consensus(best_run_data, LOW_CONFIDENCE_LEVELS)
        assert len(flags) == 1
        record = flags[0]
        # Guard the shape this test depends on: consensus value, no per-value votes.
        assert record["path"] == "structure_info.state.value"
        assert record["best_run_value"] == record["majority_vote_value"]
        assert record["all_votes"] == {}
        return record

    def test_pure_decision_keeps_default(self):
        # Empty votes on a semantic consensus is a single candidate, not a tie.
        assert _requires_explicit(self._low_confidence_record()) is False

    def test_single_collapsed_candidate_keeps_default(self):
        # review_leaf collapses an empty-votes record to one candidate (count 0);
        # a lone candidate is a consensus, so the default is kept.
        assert fork_requires_explicit_choice("value", "Active", "Active", [0]) is False

    def test_review_leaf_keeps_default_kwarg(self, monkeypatch):
        record = self._low_confidence_record()
        ask = _CapturingAsk(["1"])
        _run_review_leaf(record, ask, monkeypatch, leaf_val="Active")
        assert ask.calls, "Prompt.ask was never called"
        assert "default" in ask.calls[0]


class TestSemanticKeyMembership:
    """Guard the exact set of keys treated as identity/biology-bearing."""

    def test_expected_keys_are_semantic(self):
        for key in (
            "value",
            "role",
            "site_ref",
            "type",
            "uniprot_entry_name",
            "state",
            "oligomeric_state",
            "is_functional_ligand",
            "is_chimeric",
            "family",
            "subtype",
            "functional_coupling",
        ):
            assert key in SEMANTIC_CONTROVERSY_KEYS

    def test_display_string_keys_are_not_semantic(self):
        assert "name" not in SEMANTIC_CONTROVERSY_KEYS
        assert "pubchem_id" not in SEMANTIC_CONTROVERSY_KEYS


# ── review_leaf integration: prompt kwargs + crash safety ────────────────


class _CapturingAsk:
    """Stand-in for Prompt.ask that records kwargs and returns queued values."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _run_review_leaf(record: dict[str, Any], ask, monkeypatch, leaf_val: Any):
    monkeypatch.setattr(review_engine, "Prompt", type("P", (), {"ask": staticmethod(ask)}))
    monkeypatch.setattr(review_engine, "log_audit_trail", lambda *a, **k: None)
    controversies = {record["path"]: record}
    return review_leaf("TEST", leaf_val, controversies, record["path"], {})


class TestReviewLeafPromptDefault:
    def test_semantic_fork_omits_default_kwarg(self, monkeypatch):
        # A semantic disagreement -> Prompt.ask is called WITHOUT a default kwarg.
        record = _fork("signaling_partners.g_protein.is_chimeric")
        ask = _CapturingAsk(["1"])
        _run_review_leaf(record, ask, monkeypatch, leaf_val=False)
        assert ask.calls, "Prompt.ask was never called"
        assert "default" not in ask.calls[0]

    def test_name_tie_keeps_default_kwarg(self, monkeypatch):
        # The negative case -> Prompt.ask IS called with a default kwarg.
        record = _fork("ligands[U0G:orthosteric].name")
        ask = _CapturingAsk(["1"])
        _run_review_leaf(record, ask, monkeypatch, leaf_val="25CN-NBOH")
        assert "default" in ask.calls[0]

    def test_ligand_type_disagreement_omits_default_kwarg(self, monkeypatch):
        # A ligand molecular-type disagreement -> Prompt.ask is called WITHOUT a
        # default kwarg, so a bare Enter cannot silently commit the majority.
        record = _fork("ligands[K6G:intracellular].type")
        ask = _CapturingAsk(["1"])
        _run_review_leaf(record, ask, monkeypatch, leaf_val="lipid")
        assert ask.calls, "Prompt.ask was never called"
        assert "default" not in ask.calls[0]


class TestReviewLeafCrashSafety:
    def test_no_default_path_survives_empty_input(self, monkeypatch):
        # Drive the real Rich prompt: empty Enter must re-ask cleanly (not crash)
        # when no default is offered, then accept a valid selection.
        record = _fork("signaling_partners.g_protein.is_chimeric")
        monkeypatch.setattr(review_engine, "log_audit_trail", lambda *a, **k: None)

        replies = iter(["", "1"])

        def fake_get_input(self, console, prompt, password, stream=None):
            return next(replies)

        monkeypatch.setattr("rich.prompt.Prompt.get_input", fake_get_input)

        controversies = {record["path"]: record}
        result = review_leaf("TEST", False, controversies, record["path"], {})
        # Empty was consumed (re-asked), then "1" selected the top candidate.
        assert next(replies, "DRAINED") == "DRAINED"
        assert result is True

    def test_non_string_return_does_not_crash(self, monkeypatch):
        # Defensive guard: a non-string from Prompt.ask must not reach .lower();
        # the review falls back to the explicit edit path instead of crashing.
        record = _fork("signaling_partners.g_protein.is_chimeric")
        ask = _CapturingAsk([None, "still-editing"])
        result = _run_review_leaf(record, ask, monkeypatch, leaf_val=False)
        assert result == "still-editing"
