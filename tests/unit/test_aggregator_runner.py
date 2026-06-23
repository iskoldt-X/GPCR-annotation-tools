"""Unit tests for aggregator runner helpers."""

from __future__ import annotations

import json

import pytest

from gpcr_tools.aggregator.runner import (
    _build_validation_report,
    _coupling_protomer,
    _prune_excluded_buffer_ligands,
    _write_outputs,
)
from gpcr_tools.config import (
    ALERT_MULTI_COPY_LIGAND,
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_NO_VALID_COMPARISONS,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_STATUS_TOO_SHORT,
    CHIMERA_SUBTYPE_FAMILY_ONLY,
    CHIMERA_SUBTYPE_INSEPARABLE_SET,
    CHIMERA_SUBTYPE_LOW_CONFIDENCE,
    CHIMERA_SUBTYPE_RESOLVED,
    SUBTYPE_BASIS_CONSTRUCT_NAME,
    SUBTYPE_BASIS_FAMILY_VERIFIED,
    SUBTYPE_BASIS_RESOLVED,
    VALIDATION_EXCLUDED_BUFFER,
    VALIDATION_MATCHED_SMALL_MOLECULE,
)
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_COUPLING_PROTOMER,
    SIGNAL_SITE_REF,
    DetectSignal,
)


def _sig(kind, payload):
    return DetectSignal(
        kind=kind,
        target_ref="receptor_info",
        summary="",
        payload=payload,
        severity=SEVERITY_ADVISORY,
    )


class TestCouplingProtomer:
    def test_extracts_coupling_chain(self, monkeypatch):
        monkeypatch.setattr(
            "gpcr_tools.aggregator.runner.load_detect_signals",
            lambda pdb: [_sig(SIGNAL_COUPLING_PROTOMER, {"coupling_chain": "B"})],
        )
        assert _coupling_protomer("7C7Q") == "B"

    def test_no_coupling_signal_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "gpcr_tools.aggregator.runner.load_detect_signals",
            lambda pdb: [_sig(SIGNAL_SITE_REF, {"x": 1})],  # other kinds ignored
        )
        assert _coupling_protomer("X") is None

    def test_no_signals_returns_none(self, monkeypatch):
        monkeypatch.setattr("gpcr_tools.aggregator.runner.load_detect_signals", lambda pdb: [])
        assert _coupling_protomer("X") is None

    def test_non_string_coupling_chain_returns_none(self, monkeypatch):
        # A malformed payload must not crash or return a non-chain value.
        monkeypatch.setattr(
            "gpcr_tools.aggregator.runner.load_detect_signals",
            lambda pdb: [_sig(SIGNAL_COUPLING_PROTOMER, {"coupling_chain": 42})],
        )
        assert _coupling_protomer("X") is None


def _no_detect_signals(monkeypatch):
    """Isolate the report from the real detect sidecar (return no signals)."""
    monkeypatch.setattr("gpcr_tools.aggregator.runner.load_detect_signals", lambda pdb: [])


def _report(best, monkeypatch):
    # Isolate from real integrity checks; we only assert chimeric handling.
    monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
    _no_detect_signals(monkeypatch)
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

    def test_ai_chimeric_flag_ignored_once_alpha5_ran(self, monkeypatch):
        # When the deterministic alpha5 analysis ran (status SUCCESS), it owns
        # the chimeric review: a cleanly-resolved, aligned subtype needs no
        # manual confirmation. The model's is_chimeric flag must NOT add a
        # redundant generic 'confirm manually' warning on top.
        monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
        _no_detect_signals(monkeypatch)
        best = {
            "signaling_partners": {
                "g_protein": {
                    "is_chimeric": True,
                    "alpha_subunit": {"uniprot_entry_name": "gnas2_human"},
                }
            }
        }
        chim = _success(
            family="Gs",
            subtype="gnas2_human",
            subtype_resolution=CHIMERA_SUBTYPE_RESOLVED,
            candidate_set=["gnas2_human"],
            score=11,
            a5_tail="QRMHLRQYELL",
        )
        report = _build_validation_report("X", best, {}, [], chim, None)
        assert not any(
            "confirm the alpha-subunit identity manually" in w for w in report["critical_warnings"]
        )

    @pytest.mark.parametrize(
        "inconclusive_status",
        [CHIMERA_STATUS_TOO_SHORT, CHIMERA_STATUS_NO_VALID_COMPARISONS, "some_error_status"],
    )
    def test_ai_chimeric_flag_fires_when_alpha5_inconclusive(
        self, inconclusive_status, monkeypatch
    ):
        # The alpha5 ran but could not conclude (too short / no references /
        # error). There is no deterministic ruling, so a self-declared chimera
        # must STILL route to manual review -- not be downgraded to a generic
        # "verification could not run" note.
        monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
        _no_detect_signals(monkeypatch)
        best = {"signaling_partners": {"g_protein": {"is_chimeric": True, "alpha_subunit": {}}}}
        chim = {"status": inconclusive_status, "score": 0, "error": "n/a"}
        report = _build_validation_report("X", best, {}, [], chim, None)
        assert any(
            "confirm the alpha-subunit identity manually" in w for w in report["critical_warnings"]
        )

    def test_ai_chimeric_flag_suppressed_when_no_g_protein(self, monkeypatch):
        # The algorithm positively found no G-protein: the hallucination branch
        # owns that case, so the generic chimeric "confirm manually" warning is
        # not also emitted (it would misleadingly ask to confirm an identity that
        # the structure does not contain).
        monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
        _no_detect_signals(monkeypatch)
        best = {
            "signaling_partners": {
                "g_protein": {
                    "is_chimeric": True,
                    "alpha_subunit": {"uniprot_entry_name": "gnas2_human"},
                }
            }
        }
        chim = {"status": CHIMERA_STATUS_NO_G_PROTEIN, "score": 0}
        report = _build_validation_report("X", best, {}, [], chim, None)
        assert not any(
            "confirm the alpha-subunit identity manually" in w for w in report["critical_warnings"]
        )
        # ...but the hallucination IS surfaced (AI named a G-protein, algo found none).
        assert any(
            "NO G-protein" in c or "no g-protein" in c.lower() for c in report["algo_conflicts"]
        )


class TestUnrecognisedGAlphaBackstop:
    """A specific alpha-subunit slug off the curated G-alpha roster must reach a
    human; an in-roster slug or an honest abstention must not."""

    @staticmethod
    def _alpha(slug):
        # The alpha-subunit name field is present and carries the given value
        # (including an explicit null). For the field-absent case, pass {} directly.
        return {
            "signaling_partners": {"g_protein": {"alpha_subunit": {"uniprot_entry_name": slug}}}
        }

    def test_off_roster_slug_disables_accept_all(self, monkeypatch):
        # gnas_crigr (hamster Gs) is a real, specific slug that is not one of the
        # curated human G-alpha candidates: it must land in critical_warnings (the
        # channel that disables one-click accept-all).
        report = _report(self._alpha("gnas_crigr"), monkeypatch)
        assert any("gnas_crigr" in w for w in report["critical_warnings"])
        assert any("g_protein" in w for w in report["critical_warnings"])

    def test_in_roster_slug_no_warning(self, monkeypatch):
        report = _report(self._alpha("gnas2_human"), monkeypatch)
        assert report["critical_warnings"] == []

    def test_missing_alpha_subunit_no_warning(self, monkeypatch):
        # The g_protein block is absent entirely (the field-not-present path).
        report = _report({}, monkeypatch)
        assert report["critical_warnings"] == []

    @pytest.mark.parametrize("abstention", ["unknown", "Unknown", "none", "", "  ", None])
    def test_honest_abstention_no_warning(self, abstention, monkeypatch):
        # The AI honestly declining to name a subtype must never be flagged. The
        # None arm exercises the explicit-null (uniprot_entry_name: null) path,
        # distinct from the field-absent case above.
        report = _report(self._alpha(abstention), monkeypatch)
        assert report["critical_warnings"] == []

    def test_backstop_is_alpha_only(self, monkeypatch):
        # The candidate roster is alpha-specific; an off-roster beta/gamma slug is
        # out of scope and must not trigger the alpha-subunit backstop.
        best = {
            "signaling_partners": {
                "g_protein": {
                    "alpha_subunit": {"uniprot_entry_name": "gnas2_human"},
                    "beta_subunit": {"uniprot_entry_name": "gbb1_human"},
                    "gamma_subunit": {"uniprot_entry_name": "gbg2_human"},
                }
            }
        }
        report = _report(best, monkeypatch)
        assert report["critical_warnings"] == []

    def test_backstop_fires_independent_of_alpha5(self, monkeypatch):
        # Deterministic: it does not depend on the alpha5 API check, so it fires
        # even with the default skipped chimera_result (the --skip-api-checks path).
        report = _report(self._alpha("gnas_crigr"), monkeypatch)
        assert report["chimera_status"] is not None
        assert any("not a recognised G-alpha candidate" in w for w in report["critical_warnings"])


def _chimera_report(chimera_result, ai_uniprot, monkeypatch):
    monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
    _no_detect_signals(monkeypatch)
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
        "backbone_family": None,
        "backbone_slug": None,
        "is_alpha5_graft": False,
        "transient_abstained": [],
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
        assert any("gnas2_human" in n for n in report["detector_notes"])

    def test_alpha5_graft_records_backbone_and_notes(self, monkeypatch):
        monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
        _no_detect_signals(monkeypatch)
        best = {
            "signaling_partners": {
                "g_protein": {"alpha_subunit": {"uniprot_entry_name": "gna11_human"}}
            }
        }
        chim = _success(
            family="Gq/11",
            subtype="gna11_human",
            subtype_resolution=CHIMERA_SUBTYPE_RESOLVED,
            candidate_set=["gna11_human"],
            score=11,
            a5_tail="LQMNLREYNLV",
            is_alpha5_graft=True,
            backbone_slug="gnas2_human",
            backbone_family="Gs",
        )
        report = _build_validation_report("X", best, {}, [], chim, None)
        # Informational note recorded, never a blocking critical warning.
        assert any("ALPHA5 GRAFT" in n for n in report["detector_notes"])
        assert not any("ALPHA5 GRAFT" in w for w in report["critical_warnings"])
        # The functional coupling identity and the scaffold backbone are recorded
        # as distinct fields on the alpha subunit; the deposited slug is untouched.
        alpha = best["signaling_partners"]["g_protein"]["alpha_subunit"]
        assert alpha["uniprot_entry_name"] == "gna11_human"
        assert alpha["backbone"] == "gnas2_human"
        # alpha5 RESOLVED the subtype -> functional coupling is the detector's
        # resolved slug (here equal to the model's vote) with the RESOLVED basis.
        assert alpha["functional_coupling"] == "gna11_human"
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_RESOLVED

    def test_no_graft_records_backbone_on_alpha_subunit(self, monkeypatch):
        monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
        _no_detect_signals(monkeypatch)
        best = {
            "signaling_partners": {
                "g_protein": {"alpha_subunit": {"uniprot_entry_name": "gnas2_human"}}
            }
        }
        chim = _success(
            family="Gs",
            subtype="gnas2_human",
            subtype_resolution=CHIMERA_SUBTYPE_RESOLVED,
            candidate_set=["gnas2_human"],
            score=11,
            a5_tail="QRMHLRQYELL",
            is_alpha5_graft=False,
            backbone_slug="gnas2_human",
            backbone_family="Gs",
        )
        report = _build_validation_report("X", best, {}, [], chim, None)
        # No graft note (scaffold == functional family), but the backbone is still
        # recorded on the alpha subunit for provenance -- never gated on graft.
        assert not any("ALPHA5 GRAFT" in n for n in report["detector_notes"])
        alpha = best["signaling_partners"]["g_protein"]["alpha_subunit"]
        assert alpha["backbone"] == "gnas2_human"
        assert alpha["functional_coupling"] == "gnas2_human"

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
        assert any(
            "weak" in n.lower() or "unverified" in n.lower() for n in report["detector_notes"]
        )

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
        signals = report["algo_conflicts"] + report["critical_warnings"] + report["detector_notes"]
        assert any("does not map" in s or "cannot be determined" in s for s in signals)


def _alpha_after(chimera_result, ai_uniprot, monkeypatch):
    """Run the report and return (report, alpha_subunit dict) so the new
    aggregator-owned fields written onto the alpha subunit can be inspected."""
    monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
    _no_detect_signals(monkeypatch)
    best = {
        "signaling_partners": {"g_protein": {"alpha_subunit": {"uniprot_entry_name": ai_uniprot}}}
    }
    report = _build_validation_report("X", best, {}, [], chimera_result, None)
    return report, best["signaling_partners"]["g_protein"]["alpha_subunit"]


class TestFunctionalCouplingAndBackboneFields:
    """The functional coupling identity (from the alpha5) and the modelled
    backbone scaffold are recorded as two distinct aggregator-owned fields on
    the alpha subunit, independent of the model-facing schema."""

    def test_family_aligned_inseparable_set_keeps_model_subtype(self, monkeypatch):
        # 8XQL-shaped: alpha5 = Gi/o (transducin set, inseparable), model named
        # the specific subtype gnat3_human. Family matches -> functionally correct;
        # store the model's slug. No new conflict; backbone recorded separately.
        chim = _success(
            family="Gi/o",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnat1_human", "gnat2_human", "gnat3_human"],
            score=11,
            a5_tail="IKENLKDCGLF",
            is_alpha5_graft=True,
            backbone_slug="gnas2_human",
            backbone_family="Gs",
        )
        report, alpha = _alpha_after(chim, "gnat3_human", monkeypatch)
        assert alpha["functional_coupling"] == "gnat3_human"
        assert alpha["backbone"] == "gnas2_human"
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_FAMILY_VERIFIED
        # Family-correct within an inseparable set is functionally correct: no
        # override conflict, only the inseparable-set review note.
        assert not any("TIE-BREAKER OVERRIDE" in c for c in report["algo_conflicts"])

    def test_family_mismatch_leaves_functional_coupling_unset(self, monkeypatch):
        # 9JR3-shaped: alpha5 = Gq/11 (Gs backbone), model voted gnas2_human (Gs).
        # Family disagrees -> [TIE-BREAKER OVERRIDE]; functional_coupling unset;
        # backbone still recorded.
        chim = _success(
            family="Gq/11",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnaq_human", "gna11_human"],
            score=11,
            a5_tail="LQLNLKEYNLV",
            is_alpha5_graft=True,
            backbone_slug="gnas2_human",
            backbone_family="Gs",
        )
        report, alpha = _alpha_after(chim, "gnas2_human", monkeypatch)
        assert any("TIE-BREAKER OVERRIDE" in c for c in report["algo_conflicts"])
        assert alpha["backbone"] == "gnas2_human"
        assert "functional_coupling" not in alpha
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_CONSTRUCT_NAME

    def test_empty_uniprot_backbone_fallback(self, monkeypatch):
        # 9IIX-shaped: the G-alpha entity has no attached UniProt, so the detector
        # reports backbone_slug=None. The backbone field must still be populated
        # by the honest fallback (not absent), and the family review still gates.
        chim = _success(
            family="Gi/o",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnat1_human", "gnat2_human", "gnat3_human"],
            score=11,
            a5_tail="IKENLKDCGLF",
            is_alpha5_graft=False,
            backbone_slug=None,
            backbone_family=None,
        )
        report, alpha = _alpha_after(chim, "gnat3_human", monkeypatch)
        assert alpha["backbone"] == "unknown"
        # The model's slug is family-consistent within the inseparable set, so the
        # functional coupling is still set (family-verified) even with no backbone.
        assert alpha["functional_coupling"] == "gnat3_human"
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_FAMILY_VERIFIED
        # The family-confident, subtype-inseparable case still routes to review.
        assert any("g_protein" in w and "Gi/o" in w for w in report["critical_warnings"])

    def test_non_human_ortholog_says_species_not_subtype(self, monkeypatch):
        # 3SN6-shaped: a non-human Gs ortholog (bovine). The winners carry an
        # off-roster slug -> the review must say "non-human ortholog" / "species",
        # NOT "cannot distinguish the subtype". Still gates (critical warning).
        chim = _success(
            family="Gs",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_FAMILY_ONLY,
            candidate_set=["gnas2_bovin", "gnas2_human"],
            score=11,
            a5_tail="QRMHLRQYELL",
        )
        report, alpha = _alpha_after(chim, None, monkeypatch)
        msg = " ".join(report["critical_warnings"])
        assert "non-human ortholog" in msg or "species" in msg
        assert "cannot distinguish the subtype" not in msg
        # New aggregator-owned fields: no family-verified model slug (off-roster /
        # absent) -> functional coupling unset, basis construct-name, backbone the
        # honest "unknown" fallback (no backbone_slug on this fixture).
        assert "functional_coupling" not in alpha
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_CONSTRUCT_NAME
        assert alpha["backbone"] == "unknown"

    def test_human_only_inseparable_set_keeps_subtype_wording(self, monkeypatch):
        # Genuine human-only inseparable set {gnaq, gna11}: the wording stays
        # "cannot distinguish the subtype" -- this is a real subtype ambiguity,
        # not a species-mapping question.
        chim = _success(
            family="Gq/11",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnaq_human", "gna11_human"],
            score=11,
            a5_tail="LQLNLKEYNLV",
        )
        report, alpha = _alpha_after(chim, None, monkeypatch)
        msg = " ".join(report["critical_warnings"])
        assert "cannot distinguish the subtype" in msg
        assert "non-human ortholog" not in msg
        # New aggregator-owned fields: no model slug on this fixture -> functional
        # coupling unset, construct-name basis, backbone the "unknown" fallback.
        assert "functional_coupling" not in alpha
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_CONSTRUCT_NAME
        assert alpha["backbone"] == "unknown"

    def test_human_slug_with_non_human_ortholog_sets_coupling_and_warns(self, monkeypatch):
        # Family-only set holding BOTH a correct human slug and a non-human
        # ortholog: the model's family-consistent human slug carries the
        # functional coupling AND the non-human-ortholog review still fires --
        # both true simultaneously (the value is the best determination; the
        # warning routes the species/mapping question to a human).
        chim = _success(
            family="Gs",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_FAMILY_ONLY,
            candidate_set=["gnas2_human", "gnas2_bovin"],
            score=11,
            a5_tail="QRMHLRQYELL",
            backbone_slug="gnas2_human",
            backbone_family="Gs",
        )
        report, alpha = _alpha_after(chim, "gnas2_human", monkeypatch)
        # Functional coupling IS set (model slug is family-consistent)...
        assert alpha["functional_coupling"] == "gnas2_human"
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_FAMILY_VERIFIED
        assert alpha["backbone"] == "gnas2_human"
        # ...and the non-human-ortholog warning fires at the same time.
        msg = " ".join(report["critical_warnings"])
        assert "non-human ortholog" in msg or "species" in msg

    def test_detector_resolved_overrides_model_same_family_vote(self, monkeypatch):
        # The alpha5 RESOLVED a unique subtype (gnas2_human) while the model voted
        # a DIFFERENT same-family slug (gnal_human, also Gs). The stored functional
        # coupling follows the DETECTOR (the reliable source), and the
        # [TIE-BREAKER OVERRIDE] conflict still fires to gate the disagreement.
        chim = _success(
            family="Gs",
            family_confident=True,
            subtype="gnas2_human",
            subtype_resolution=CHIMERA_SUBTYPE_RESOLVED,
            candidate_set=["gnas2_human"],
            score=11,
            a5_tail="QRMHLRQYELL",
        )
        report, alpha = _alpha_after(chim, "gnal_human", monkeypatch)
        # Stored value is the detector's resolved subtype, NOT the model's vote.
        assert alpha["functional_coupling"] == "gnas2_human"
        assert alpha["subtype_basis"] == SUBTYPE_BASIS_RESOLVED
        # The model/detector split is still surfaced for confirmation.
        assert any(
            "TIE-BREAKER OVERRIDE" in c and "gnal_human" in c for c in report["algo_conflicts"]
        )


class TestDetectReviewSignalsRouted:
    """Detect REVIEW signals reach the curator as critical warnings -- the
    production consumer of the detect review route. The chimeric kind is the
    one exception: the aggregator re-derives it from its own alpha5 analysis,
    so routing the detect copy too would duplicate / override that."""

    def _report_with_signals(self, signals, monkeypatch):
        monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
        monkeypatch.setattr("gpcr_tools.aggregator.runner.load_detect_signals", lambda pdb: signals)
        return _build_validation_report("X", {}, {}, [], {}, None)

    def test_review_signal_reaches_critical_warnings(self, monkeypatch):
        sig = DetectSignal(
            kind="some_review_kind",
            target_ref="ligands",
            summary="a human must look at this",
            severity=SEVERITY_REVIEW,
        )
        report = self._report_with_signals([sig], monkeypatch)
        assert any("a human must look at this" in w for w in report["critical_warnings"])

    def test_advisory_signal_not_surfaced_as_warning(self, monkeypatch):
        sig = DetectSignal(
            kind="some_kind",
            target_ref="ligands",
            summary="advisory evidence only",
            severity=SEVERITY_ADVISORY,
        )
        report = self._report_with_signals([sig], monkeypatch)
        assert not any("advisory evidence only" in w for w in report["critical_warnings"])

    def test_chimeric_review_signal_not_double_surfaced(self, monkeypatch):
        sig = DetectSignal(
            kind=SIGNAL_CHIMERIC_GPROTEIN,
            target_ref="signaling_partners.g_protein.alpha_subunit",
            summary="alpha5 cannot distinguish the subtype",
            severity=SEVERITY_REVIEW,
        )
        report = self._report_with_signals([sig], monkeypatch)
        assert not any("cannot distinguish the subtype" in w for w in report["critical_warnings"])


class TestVotingLogAlwaysWritten:
    """The voting log is written for every PDB, including those with no
    discrepancies, so aggregation always leaves an audit trace of the vote."""

    def test_log_written_when_no_discrepancies(self, configure_paths):
        report = {"critical_warnings": [], "algo_conflicts": [], "detector_notes": []}
        result = _write_outputs("TEST1", {"receptor_info": {}}, [], report)
        assert result.voting_log_path is not None
        assert result.voting_log_path.is_file()
        # A clean PDB's log is an explicit empty list (not a missing file), which
        # yields an empty controversy map downstream -- so it never gates accept-all.
        assert json.loads(result.voting_log_path.read_text()) == []

    def test_log_carries_discrepancy_records_when_present(self, configure_paths):
        report = {"critical_warnings": [], "algo_conflicts": [], "detector_notes": []}
        discrepancies = [
            {
                "path": "ligands[RET]",
                "best_run_value": None,
                "majority_vote_value": {"chem_comp_id": "RET"},
                "all_votes": {"role": {"agonist": 2}},
                "needs_review": True,
            }
        ]
        result = _write_outputs("TEST1", {"receptor_info": {}}, discrepancies, report)
        assert result.voting_log_path is not None
        logged = json.loads(result.voting_log_path.read_text())
        assert logged == discrepancies


def _lig(**overrides):
    """Minimal ligand record; overrides set the fields under test."""
    base = {"chem_comp_id": "XXX", "name": "ligand", "validation_status": None}
    base.update(overrides)
    return base


def _comp_ids(best):
    return [lig.get("chem_comp_id") for lig in best["ligands"]]


class TestPruneExcludedBufferLigands:
    """Excluded-buffer ligands (detergents / cryo-additives / matrix lipids such
    as BOG / NAG) are dropped from the aggregated record at the aggregation layer,
    so they never reach the curator or the CSV -- while a genuinely-functional
    incidental lipid the model judged real survives."""

    def test_excluded_buffer_dropped(self):
        # 5WKT-shaped: BOG / NAG tagged EXCLUDED_BUFFER, no role check, role
        # Cofactor -> both dropped.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="BOG",
                    validation_status=VALIDATION_EXCLUDED_BUFFER,
                    pharmacological_role_check=None,
                    role={"value": "Cofactor"},
                ),
                _lig(
                    chem_comp_id="NAG",
                    validation_status=VALIDATION_EXCLUDED_BUFFER,
                    pharmacological_role_check=None,
                    role={"value": "Cofactor"},
                ),
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == []

    def test_functional_incidental_lipid_survives(self):
        # PLM the model judged a real functional ligand (is_functional_ligand True)
        # is rescued and kept even though it carries the EXCLUDED_BUFFER tag.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="PLM",
                    validation_status=VALIDATION_EXCLUDED_BUFFER,
                    pharmacological_role_check={"is_functional_ligand": True},
                )
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert _comp_ids(best) == ["PLM"]

    def test_non_functional_incidental_lipid_dropped(self):
        # 2HPY / 3PQR-shaped: PLM tagged EXCLUDED_BUFFER with the model's verdict
        # is_functional_ligand False -> the `is True` rescue does NOT fire, so it
        # is dropped (palmitoylation PTM, not a bound ligand).
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="PLM",
                    validation_status=VALIDATION_EXCLUDED_BUFFER,
                    pharmacological_role_check={"is_functional_ligand": False},
                )
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == []

    def test_matched_small_molecule_not_dropped(self):
        # A matched lipid carries MATCHED_SMALL_MOLECULE (NOT EXCLUDED_BUFFER), so
        # the status-only predicate leaves it untouched -- a component-id check
        # would have wrongly dropped it.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="CLR",
                    validation_status=VALIDATION_MATCHED_SMALL_MOLECULE,
                    pharmacological_role_check=None,
                )
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert _comp_ids(best) == ["CLR"]

    def test_null_and_missing_prc_both_dropped(self):
        # The rescue is an `is True` identity test, not truthiness: an explicit
        # null role check and an entirely absent one both fail to rescue.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="BOG",
                    validation_status=VALIDATION_EXCLUDED_BUFFER,
                    pharmacological_role_check=None,
                ),
                _lig(chem_comp_id="NAG", validation_status=VALIDATION_EXCLUDED_BUFFER),
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == []

    def test_normal_matched_small_molecule_survives(self):
        # A bona-fide drug ligand (MATCHED_SMALL_MOLECULE) is never touched.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="ZMA",
                    name="Adenosine antagonist",
                    validation_status=VALIDATION_MATCHED_SMALL_MOLECULE,
                )
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert _comp_ids(best) == ["ZMA"]

    def test_multi_copy_alert_pruned_for_dropped_buffer(self):
        # A dropped multi-copy buffer's MULTI_COPY_LIGAND alert must be pruned from
        # oligomer_analysis["alerts"] so no alert dangles at a ligand that no
        # longer exists; an alert for a surviving ligand stays.
        best = {
            "ligands": [
                _lig(chem_comp_id="BOG", validation_status=VALIDATION_EXCLUDED_BUFFER),
                _lig(chem_comp_id="ZMA", validation_status=VALIDATION_MATCHED_SMALL_MOLECULE),
            ],
            "oligomer_analysis": {
                "alerts": [
                    {
                        "type": ALERT_MULTI_COPY_LIGAND,
                        "message": (
                            f"[{ALERT_MULTI_COPY_LIGAND}] at 'ligands[BOG]': modelled in "
                            "5 copies (instances E, F, G, H, I); human review recommended."
                        ),
                    },
                    {
                        "type": ALERT_MULTI_COPY_LIGAND,
                        "message": (
                            f"[{ALERT_MULTI_COPY_LIGAND}] at 'ligands[ZMA]': modelled in "
                            "2 copies (instances A, B); human review recommended."
                        ),
                    },
                ]
            },
        }
        _prune_excluded_buffer_ligands(best)
        assert _comp_ids(best) == ["ZMA"]
        alerts = best["oligomer_analysis"]["alerts"]
        messages = " ".join(a["message"] for a in alerts)
        assert "ligands[BOG]" not in messages
        assert "ligands[ZMA]" in messages

    def test_multi_copy_alert_pruning_guards_absent_oligomer(self):
        # No oligomer_analysis block at all -> dropping still works, no crash.
        best = {"ligands": [_lig(chem_comp_id="BOG", validation_status=VALIDATION_EXCLUDED_BUFFER)]}
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == []

    def test_no_ligands_key_no_crash(self):
        best: dict = {"receptor_info": {}}
        _prune_excluded_buffer_ligands(best)
        # No ligands list present -> nothing added, nothing crashed.
        assert "ligands" not in best or best["ligands"] == []

    def test_empty_ligands_list_no_crash(self):
        best: dict = {"ligands": []}
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == []

    def test_all_dropped_becomes_empty_list(self):
        best = {
            "ligands": [
                _lig(chem_comp_id="BOG", validation_status=VALIDATION_EXCLUDED_BUFFER),
                _lig(chem_comp_id="NAG", validation_status=VALIDATION_EXCLUDED_BUFFER),
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == []

    def test_non_dict_ligand_entry_preserved(self):
        # A malformed ligands list (a None and a stray string mixed in with real
        # dicts) must not crash: the isinstance(lig, dict) guard means non-dict
        # entries are never inspected for the buffer predicate, so they are kept
        # as-is while the EXCLUDED_BUFFER dict is dropped and the normal dict
        # survives.
        normal = _lig(chem_comp_id="ZMA", validation_status=VALIDATION_MATCHED_SMALL_MOLECULE)
        best = {
            "ligands": [
                None,
                _lig(chem_comp_id="BOG", validation_status=VALIDATION_EXCLUDED_BUFFER),
                "stray",
                normal,
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert best["ligands"] == [None, "stray", normal]
