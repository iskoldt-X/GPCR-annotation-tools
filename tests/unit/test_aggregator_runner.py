"""Unit tests for aggregator runner helpers."""

from __future__ import annotations

from gpcr_tools.aggregator.runner import _build_validation_report
from gpcr_tools.config import (
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_SUBTYPE_FAMILY_ONLY,
    CHIMERA_SUBTYPE_INSEPARABLE_SET,
    CHIMERA_SUBTYPE_LOW_CONFIDENCE,
    CHIMERA_SUBTYPE_RESOLVED,
)


def _report(best, monkeypatch):
    # Isolate from real integrity checks; we only assert chimeric handling.
    monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
    return _build_validation_report("X", best, {}, [], {}, None)


class TestChimericForcesReview:
    def test_chimeric_g_protein_forces_manual_review(self, monkeypatch):
        # A chimeric G-protein cannot be resolved from sequence alone, so it
        # must raise a critical warning (which disables one-click accept-all
        # and surfaces in review) rather than be accepted silently.
        best = {
            "signaling_partners": {
                "g_protein": {
                    "is_chimeric": True,
                    "alpha_subunit": {"uniprot_entry_name": "gnai1_human"},
                }
            }
        }
        report = _report(best, monkeypatch)
        warnings = report["critical_warnings"]
        assert any("chimeric" in w.lower() for w in warnings)
        # surfaces under the signaling_partners block (warning names g_protein)
        assert any("g_protein" in w for w in warnings)

    def test_non_chimeric_not_forced(self, monkeypatch):
        best = {"signaling_partners": {"g_protein": {"is_chimeric": False, "alpha_subunit": {}}}}
        report = _report(best, monkeypatch)
        assert not any("chimeric" in w.lower() for w in report["critical_warnings"])

    def test_missing_g_protein_no_crash(self, monkeypatch):
        report = _report({}, monkeypatch)
        assert not any("chimeric" in w.lower() for w in report["critical_warnings"])


def _chimera_report(chimera_result, ai_uniprot, monkeypatch):
    monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
    best: dict = {}
    if ai_uniprot is not None:
        best = {
            "signaling_partners": {
                "g_protein": {"alpha_subunit": {"uniprot_entry_name": ai_uniprot}}
            }
        }
    return _build_validation_report("X", best, {}, [], chimera_result, None)


def _success(**overrides):
    base = {
        "status": CHIMERA_STATUS_SUCCESS,
        "family": None,
        "family_confident": False,
        "subtype": None,
        "subtype_resolution": None,
        "candidate_set": [],
        "score": 0,
        "a5_window": 11,
        "a5_tail": "XXXXXXXXXXX",
        "candidates_checked": [],
        "error": None,
    }
    base.update(overrides)
    return base


class TestChimeraAlpha5Routing:
    def test_resolved_subtype_aligned(self, monkeypatch):
        chim = _success(
            family="Gs",
            subtype="gnas2_human",
            subtype_resolution=CHIMERA_SUBTYPE_RESOLVED,
            candidate_set=["gnas2_human"],
            score=11,
            a5_tail="QRMHLRQYELL",
        )
        report = _chimera_report(chim, "gnas2_human", monkeypatch)
        assert report["algo_conflicts"] == []
        assert any("gnas2_human" in n for n in report["algo_notes"])

    def test_resolved_subtype_mismatch_is_conflict(self, monkeypatch):
        chim = _success(
            family="Gs",
            subtype="gnas2_human",
            subtype_resolution=CHIMERA_SUBTYPE_RESOLVED,
            candidate_set=["gnas2_human"],
            score=11,
            a5_tail="QRMHLRQYELL",
        )
        report = _chimera_report(chim, "gnao_human", monkeypatch)
        assert any("gnas2_human" in c for c in report["algo_conflicts"])

    def test_inseparable_set_routes_to_review(self, monkeypatch):
        # alpha5 = transducin: family Gi/o, subtype indistinguishable -> review.
        chim = _success(
            family="Gi/o",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnat1_human", "gnat2_human", "gnat3_human"],
            score=11,
            a5_tail="IKENLKDCGLF",
        )
        report = _chimera_report(chim, "gnat1_human", monkeypatch)  # AI same family
        assert any("g_protein" in w and "Gi/o" in w for w in report["critical_warnings"])
        assert report["algo_conflicts"] == []

    def test_family_disagreement_is_conflict(self, monkeypatch):
        # AI picked a different coupling family than the alpha5 indicates.
        chim = _success(
            family="Gi/o",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnat1_human", "gnat2_human", "gnat3_human"],
            score=11,
            a5_tail="IKENLKDCGLF",
        )
        report = _chimera_report(chim, "gnas2_human", monkeypatch)  # AI says Gs
        assert any("Gi/o" in c and "Gs" in c for c in report["algo_conflicts"])

    def test_low_confidence_is_noted_not_crashed(self, monkeypatch):
        chim = _success(subtype_resolution=CHIMERA_SUBTYPE_LOW_CONFIDENCE, score=3)
        report = _chimera_report(chim, None, monkeypatch)
        assert any("weak" in n.lower() or "unverified" in n.lower() for n in report["algo_notes"])

    def test_cross_family_tie_is_not_silent(self, monkeypatch):
        # Winners span more than one family -> family is None. This must NOT be
        # silently dropped; it surfaces as a conflict for manual resolution.
        chim = _success(
            subtype_resolution=CHIMERA_SUBTYPE_FAMILY_ONLY,
            candidate_set=["gnas2_human", "gnaq_human"],
            score=9,
            a5_tail="ABCDEFGHIJK",
        )
        report = _chimera_report(chim, None, monkeypatch)
        signals = report["algo_conflicts"] + report["critical_warnings"] + report["algo_notes"]
        assert any("does not map" in s or "cannot be determined" in s for s in signals)
