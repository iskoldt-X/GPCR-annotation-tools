"""Tests for the per-signal decision enumerator (validator/decisions.py).

``enumerate_decisions`` builds the curator's read-only brief: one item per
signal, ranked most-severe first. The invariants that matter:

  - vote forks are read from the RAW voting log, independent of the
    whole-structure gated boolean, so they are enumerated even when the
    structure looks clean by its validation and oligomer reasons;
  - a cross-field contradiction (an algorithm conflict) is representable as an
    item that carries its message;
  - the ``gating`` flag is pinned so an explicit ``None`` (like an absent flag)
    routes to a human, and only an explicit ``False`` is advisory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpcr_tools.validator.decisions import (
    CATEGORY_ALGORITHM_CONFLICT,
    CATEGORY_OLIGOMER,
    CATEGORY_VOTE_ADVISORY,
    CATEGORY_VOTE_DISAGREEMENT,
    CATEGORY_VOTE_LOW_CONFIDENCE,
    CATEGORY_VOTE_NEAR_TIE,
    CATEGORY_VOTE_OMISSION,
    DecisionItem,
    enumerate_decisions,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "decision_brief_signals.json"


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def _fixture_items() -> list[DecisionItem]:
    data = _load_fixture()
    return enumerate_decisions(data["main_data"], data["validation_data"], data["voting_log"])


# ── empty / absent sources ──────────────────────────────────────────────


class TestEmptyInputs:
    def test_all_none_returns_empty(self):
        assert enumerate_decisions(None, None, None) == []

    def test_all_empty_returns_empty(self):
        assert enumerate_decisions({}, {}, []) == []


# ── forks enumerated independent of the gated boolean ────────────────────


class TestForksEnumeratedFromRawLog:
    def test_forks_listed_when_clean_by_validation_and_oligomer(self):
        # No critical warnings, no algorithm conflicts, no oligomer findings: by
        # those reasons the structure looks clean. The raw voting log still
        # carries a fork, and it must be enumerated regardless.
        voting_log = [
            {
                "path": "receptor_info.uniprot_entry_name",
                "best_run_value": "aa2ar_human",
                "majority_vote_value": "adora2a_human",
                "all_votes": {"aa2ar_human": 2, "adora2a_human": 3},
            }
        ]
        items = enumerate_decisions({}, {}, voting_log)
        assert [i.path for i in items] == ["receptor_info.uniprot_entry_name"]
        item = items[0]
        assert item.gating is True
        assert item.category == CATEGORY_VOTE_DISAGREEMENT

    def test_accepts_raw_list_and_path_keyed_map(self):
        # The curator holds a path-keyed map; an offline caller holds the raw
        # list. Both must enumerate to the same forks.
        record = {
            "path": "x.role",
            "best_run_value": "Agonist",
            "majority_vote_value": "Antagonist",
            "all_votes": {"Agonist": 4, "Antagonist": 6},
        }
        as_list = enumerate_decisions({}, {}, [record])
        as_map = enumerate_decisions({}, {}, {"x.role": record})
        assert [i.path for i in as_list] == [i.path for i in as_map] == ["x.role"]

    def test_vote_item_carries_best_majority_votes_and_margin(self):
        record = {
            "path": "receptor_info.uniprot_entry_name",
            "best_run_value": "aa2ar_human",
            "majority_vote_value": "adora2a_human",
            "all_votes": {"aa2ar_human": 2, "adora2a_human": 3},
        }
        (item,) = enumerate_decisions({}, {}, [record])
        assert item.best_value == "aa2ar_human"
        assert item.majority_value == "adora2a_human"
        assert item.votes == {"aa2ar_human": 2, "adora2a_human": 3}
        assert item.margin == 1


# ── whole-entity minority omission (dict majority, nested all_votes) ─────


class TestMinorityOmission:
    """A fork where some runs report a ligand/auxiliary protein the selected run
    left out: its ``majority_vote_value`` is the FULL entity dict and its
    ``all_votes`` is a nested per-field breakdown, not a scalar leaf with a flat
    ``{value: count}`` tally. This is the most common non-scalar fork shape in
    real aggregated logs and must read as an omitted entity, not be dumped raw.
    """

    @staticmethod
    def _omission_record() -> dict:
        return {
            "path": "ligands[CLR:allosteric_7tm]",
            "best_run_value": None,
            "majority_vote_value": {
                "chain_id": "A, B",
                "chem_comp_id": "CLR",
                "name": "Cholesterol",
                "role": {"confidence": None, "evidence": None, "value": "Cofactor"},
                "site_ref": "allosteric_7tm",
                "type": "lipid",
            },
            "all_votes": {
                "chain_id": {"A, B": 4},
                "chem_comp_id": {"CLR": 4},
                "name": {"Cholesterol": 4},
                "role": {"confidence": {}, "evidence": {}, "value": {"Cofactor": 4}},
                "site_ref": {"allosteric_7tm": 4},
                "type": {"lipid": 4},
            },
            "needs_review": True,
            "gating": False,
        }

    def test_dict_majority_reads_as_omission_not_low_confidence(self):
        (item,) = enumerate_decisions({}, {}, [self._omission_record()])
        assert item.category == CATEGORY_VOTE_OMISSION
        # The entity dict is never dumped into the one-line summary, and it is
        # not mislabelled as a confidence issue.
        assert "{" not in item.summary
        assert "low-confidence" not in item.summary
        # The entity is named so the curator knows what was omitted.
        assert "Cholesterol" in item.summary
        assert "CLR" in item.summary
        assert "omitted" in item.summary.lower()

    def test_nested_all_votes_leaves_no_flat_counts_or_margin(self):
        (item,) = enumerate_decisions({}, {}, [self._omission_record()])
        # A nested per-field breakdown is not a flat {value: count} tally, so
        # neither a candidate map nor a top-two margin is claimed for it.
        assert item.votes is None
        assert item.margin is None

    def test_omission_is_advisory_and_carries_role_evidence(self):
        (item,) = enumerate_decisions({}, {}, [self._omission_record()])
        assert item.gating is False  # gating=False -> advisory, does not route
        # The entity's own role is surfaced as evidence rather than lost.
        assert item.evidence and "Cofactor" in item.evidence

    def test_omission_from_fixture_is_enumerated(self):
        items = _fixture_items()
        omissions = [i for i in items if i.category == CATEGORY_VOTE_OMISSION]
        assert omissions, "a whole-entity omission fork must surface as a decision item"
        assert "Cholesterol" in omissions[0].summary


# ── cross-field contradiction is representable ───────────────────────────


class TestCrossFieldContradiction:
    def test_algorithm_conflict_becomes_an_item(self):
        items = _fixture_items()
        conflicts = [i for i in items if i.category == CATEGORY_ALGORITHM_CONFLICT]
        assert conflicts, "an algorithm conflict must surface as a decision item"
        conflict = conflicts[0]
        assert conflict.gating is True
        # The contradiction's own message is carried through as evidence, and its
        # inline path anchor is surfaced so the brief can point at the block.
        assert "family" in conflict.summary.lower()
        assert conflict.evidence == conflict.summary
        assert conflict.path == "signaling_partners.g_protein.alpha_subunit"

    def test_oligomer_finding_is_representable(self):
        main = {
            "oligomer_analysis": {
                "chain_id_override": {"applied": False},
                "alerts": [
                    {
                        "type": "MISSED_PROTOMER",
                        "message": "[MISSED_PROTOMER] at 'oligomer_analysis': Missed chain B",
                    }
                ],
                "all_gpcr_chains": [],
            }
        }
        items = enumerate_decisions(main, {}, [])
        oligo_items = [i for i in items if i.category == CATEGORY_OLIGOMER]
        assert oligo_items
        assert oligo_items[0].gating is True


# ── gating flag: pinned semantics ────────────────────────────────────────


class TestGatingFlagSemantics:
    def test_explicit_none_gates(self):
        # PINNED: an explicit gating=None means "not cleared" -> routes to a human.
        record = {
            "path": "structure_info.state.value",
            "best_run_value": "Active",
            "majority_vote_value": "Active",
            "all_votes": {},
            "gating": None,
        }
        (item,) = enumerate_decisions({}, {}, [record])
        assert item.gating is True

    def test_absent_flag_gates(self):
        record = {
            "path": "x.role",
            "best_run_value": "Agonist",
            "majority_vote_value": "Agonist",
            "all_votes": {"Agonist": 5, "Co-agonist": 4},
        }
        (item,) = enumerate_decisions({}, {}, [record])
        assert item.gating is True

    def test_explicit_false_is_advisory(self):
        record = {
            "path": "ligands[U0G:orthosteric].name",
            "best_run_value": "25CN-NBOH",
            "majority_vote_value": "25CN-NBOH",
            "all_votes": {"25CN-NBOH": 5, "25-CN-NBOH": 5},
            "gating": False,
        }
        (item,) = enumerate_decisions({}, {}, [record])
        assert item.gating is False
        assert item.category == CATEGORY_VOTE_ADVISORY

    def test_fixture_none_flag_gates_and_reads_low_confidence(self):
        items = _fixture_items()
        state = next(i for i in items if i.path == "structure_info.state.value")
        assert state.gating is True
        # Empty votes -> no runner-up to tie with -> a low-confidence consensus,
        # not a near-tie split.
        assert state.category == CATEGORY_VOTE_LOW_CONFIDENCE
        assert state.margin is None

    def test_even_split_reads_near_tie(self):
        items = _fixture_items()
        role = next(i for i in items if i.path == "ligand_copies[Q:1006].role")
        assert role.gating is True  # gating flag absent -> gates
        assert role.category == CATEGORY_VOTE_NEAR_TIE
        assert role.margin == 0


# ── severity ordering ─────────────────────────────────────────────────────


class TestSeverityOrdering:
    def test_most_severe_first(self):
        items = _fixture_items()
        severities = [i.severity for i in items]
        assert severities == sorted(severities, reverse=True)
        # The algorithm conflict outranks every vote fork.
        assert items[0].category == CATEGORY_ALGORITHM_CONFLICT

    def test_disagreement_ranks_above_advisory(self):
        cats = [i.category for i in _fixture_items()]
        assert cats.index(CATEGORY_VOTE_DISAGREEMENT) < cats.index(CATEGORY_VOTE_ADVISORY)
