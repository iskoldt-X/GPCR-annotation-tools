"""Tests for the read-only decision brief renderer (csv_generator/ui.py).

The brief is informational: it renders for a gated structure and stays silent
for a clean one, and it never prompts or mutates the data it is handed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from gpcr_tools.csv_generator.ui import display_decision_brief

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "decision_brief_signals.json"


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


class TestDecisionBriefRender:
    def test_renders_identity_calls_and_decisions_when_gated(self, capsys):
        data = _load_fixture()
        rendered = display_decision_brief(
            "TEST", data["main_data"], data["voting_log"], data["validation_data"]
        )
        out = capsys.readouterr().out
        assert rendered is True
        assert "DECISION BRIEF" in out
        assert "TEST" in out
        # identity + oligomer + G-protein one-liners
        assert "aa2ar_human" in out
        assert "MONOMER" in out
        assert "gnas2_human" in out
        # a severity-ordered decision (the algorithm conflict) is listed
        assert "ALGORITHM CONFLICT" in out

    def test_omitted_entity_rendered_cleanly(self, capsys):
        # A whole-entity minority omission (dict-valued majority) must render as a
        # named omitted entity, not as a dumped entity dict mislabelled
        # low-confidence.
        data = _load_fixture()
        display_decision_brief(
            "TEST", data["main_data"], data["voting_log"], data["validation_data"]
        )
        out = capsys.readouterr().out
        assert "MINORITY OMISSION" in out
        assert "Cholesterol" in out
        # The entity dict is not dumped into the brief (collapse whitespace so a
        # folded repr would still be caught as one contiguous token).
        assert "'chem_comp_id'" not in "".join(out.split())

    def test_silent_when_not_gated(self, capsys):
        main = {"receptor_info": {"uniprot_entry_name": "aa2ar_human"}}
        # A lone advisory fork does not gate -> no brief.
        advisory = [
            {
                "path": "ligands[U0G:orthosteric].name",
                "best_run_value": "x",
                "majority_vote_value": "x",
                "all_votes": {"x": 5, "y": 5},
                "gating": False,
            }
        ]
        rendered = display_decision_brief("TEST", main, advisory, {})
        assert rendered is False
        assert capsys.readouterr().out == ""

    def test_read_only_does_not_mutate_inputs(self):
        data = _load_fixture()
        main_before = copy.deepcopy(data["main_data"])
        votes_before = copy.deepcopy(data["voting_log"])
        val_before = copy.deepcopy(data["validation_data"])
        display_decision_brief(
            "TEST", data["main_data"], data["voting_log"], data["validation_data"]
        )
        assert data["main_data"] == main_before
        assert data["voting_log"] == votes_before
        assert data["validation_data"] == val_before
