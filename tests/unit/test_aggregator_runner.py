"""Unit tests for aggregator runner helpers."""

from __future__ import annotations

import copy
import json

import pytest

from gpcr_tools.aggregator.runner import (
    _build_validation_report,
    _coupling_protomer,
    _prune_excluded_buffer_ligands,
    _rebuild_small_molecule_rows_from_per_copy,
    _write_outputs,
)
from gpcr_tools.aggregator.voting import find_discrepancies
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
    SITE_REF_ALLOSTERIC_7TM,
    SITE_REF_INTRACELLULAR,
    SITE_REF_MEMBRANE_FACING,
    SITE_REF_ORTHOSTERIC,
    SITE_REF_UNKNOWN,
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
        # A chimeric G protein cannot be resolved from sequence alone, so it
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
        # The algorithm positively found no G protein: the hallucination branch
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
        # ...but the hallucination IS surfaced (AI named a G protein, algo found none).
        assert any(
            "NO G protein" in c or "no g protein" in c.lower() for c in report["algo_conflicts"]
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


# ---------------------------------------------------------------------------
# Per-copy small-molecule row rebuild
# ---------------------------------------------------------------------------


def _pc(copy_id, site, role="Cofactor", confidence="Medium"):
    """A per-copy vote row (as it appears in majority_votes['ligand_copies'])."""
    return {"copy_id": copy_id, "site_ref": site, "role": {"value": role}, "confidence": confidence}


def _inst(auth_asym, auth_seq, label):
    """A nonpolymer instance-index entry (author chain / residue / mmCIF label)."""
    return {"auth_asym_id": auth_asym, "auth_seq_id": auth_seq, "label_asym_id": label}


def _sm_lig(comp, site, is_functional=None, **overrides):
    """A small-molecule ligand row with a component id, site, and functional verdict."""
    prc = None if is_functional is None else {"is_functional_ligand": is_functional}
    return _lig(
        chem_comp_id=comp,
        site_ref=site,
        validation_status=VALIDATION_MATCHED_SMALL_MOLECULE,
        pharmacological_role_check=prc,
        role={"value": "Cofactor"},
        **overrides,
    )


def _rows(best):
    """(chem_comp_id, site_ref, is_functional) for each rebuilt ligand row."""
    out = []
    for lig in best["ligands"]:
        prc = lig.get("pharmacological_role_check") if isinstance(lig, dict) else None
        isf = prc.get("is_functional_ligand") if isinstance(prc, dict) else None
        out.append((lig.get("chem_comp_id"), lig.get("site_ref"), isf))
    return out


class TestRebuildSmallMoleculeRowsFromPerCopy:
    """Small-molecule ligand rows are re-derived from the aggregated per-copy site
    votes, so a physical copy is grouped under the site the runs agreed on rather
    than inheriting one outlier best run's copy list. Row set / membership / site
    come from the voted per-copy assignment; keep/drop and is_functional from the
    in-memory majority vote; chemistry (and existence) from the post-prune list."""

    def test_split_one_inflated_row_into_two_by_voted_site(self):
        # 7E2X-shaped: one CLR copy is a pocket PAM, the rest are membrane lipids.
        # The best run carried both rows but flipped the membrane row's verdict to
        # non-functional; the majority keeps it functional. The rebuild produces
        # both rows, each holding only the copies voted to its site.
        best = {
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False),  # best-run outlier
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F"), _inst("R", "603", "G"), _inst("R", "604", "H")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:603", SITE_REF_MEMBRANE_FACING),
                _pc("R:604", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),  # majority verdict
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [
            ("CLR", SITE_REF_ALLOSTERIC_7TM, True),
            ("CLR", SITE_REF_MEMBRANE_FACING, True),
        ]
        # The record now ships the voted per-copy list, so the CSV writer
        # re-partitions each row's copies from the same attribution.
        assert best["ligand_copies"] == mv["ligand_copies"]

    def test_majority_non_functional_row_stamped_false_and_copies_follow_out(self):
        # A structural lipid the MAJORITY judged non-functional is stamped False
        # (over the best run's outlier True). The row is still emitted (recorded for
        # the curator); the CSV writer then drops it AND follows its copies out --
        # they must not flood the surviving sibling row as tagged homeless copies.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),  # best-run outlier
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F"), _inst("R", "603", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:603", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False),  # majority verdict
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        # Both groups emitted; the membrane group now carries the majority False.
        assert _rows(best) == [
            ("CLR", SITE_REF_ALLOSTERIC_7TM, True),
            ("CLR", SITE_REF_MEMBRANE_FACING, False),
        ]
        rows = transform_for_csv("XXXX", best)["ligands.csv"]
        by_site = {r["Site"]: r for r in rows}
        # The non-functional membrane row is dropped, and its copy follows it out:
        # the surviving allosteric row keeps only its own copy, no "(?)" inflation.
        assert set(by_site) == {SITE_REF_ALLOSTERIC_7TM}
        assert by_site[SITE_REF_ALLOSTERIC_7TM]["Residue_seq_id"] == "R:602"

    def test_resurrects_row_the_best_run_wrongly_dropped(self):
        # 8Y69-shaped silent drop: the only CLR row carries the best run's False
        # verdict (the CSV writer would drop it, leaving no sibling row). The
        # majority verdict is True, so the rebuild stamps True and the row survives.
        best = {
            "ligands": [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False)],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("D", "901", "I"), _inst("H", "301", "J")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("D:901", SITE_REF_MEMBRANE_FACING),
                _pc("H:301", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [("CLR", SITE_REF_MEMBRANE_FACING, True)]

    def test_null_verdict_site_with_no_post_prune_row_is_not_fabricated(self):
        # The per-copy votes place the copies at a site (membrane) that has NO
        # post-prune row of this component and NO majority-True call -- so that
        # site must NOT be fabricated into a shipped row by borrowing the
        # component's chemistry from another site. The compound-level allosteric
        # row (which no per-copy group reached) is kept (never silently deleted),
        # and the membrane copies leave via a dropped follow-out marker instead of
        # flooding the surviving allosteric row.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                _sm_lig("C8E", SITE_REF_ALLOSTERIC_7TM, is_functional=None, name="detergent")
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {"C8E": [_inst("A", "1", "L"), _inst("A", "2", "N")]}
            },
        }
        mv = {
            "ligand_copies": [
                _pc("A:1", SITE_REF_MEMBRANE_FACING),
                _pc("A:2", SITE_REF_MEMBRANE_FACING),
            ],
            # No C8E:membrane_facing ligand vote -> "not assessed" (null), and there
            # is no post-prune membrane row -> the site is not fabricated.
            "ligands": [_sm_lig("C8E", SITE_REF_ALLOSTERIC_7TM, is_functional=None)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        # The kept allosteric anchor row (uncovered by any group -> no derived chain)
        # plus the dropped membrane follow-out marker (never shipped).
        assert _rows(best) == [
            ("C8E", SITE_REF_ALLOSTERIC_7TM, None),
            ("C8E", SITE_REF_MEMBRANE_FACING, False),
        ]
        # Chemistry is carried over from the component's post-prune row.
        assert best["ligands"][0]["name"] == "detergent"
        rows = transform_for_csv("XXXX", best)["ligands.csv"]
        # Only the allosteric row ships; the membrane marker is dropped and takes its
        # two copies with it -- no "(?)" inflation on the surviving row.
        assert [r["Site"] for r in rows] == [SITE_REF_ALLOSTERIC_7TM]
        assert rows[0]["Residue_seq_id"] == ""
        assert not any("(?)" in r["Residue_seq_id"] for r in rows)

    def test_orphan_component_pruned_from_list_builds_no_row(self):
        # PLM is on both the incidental roster and the exclude list, so the buffer
        # prune removed it from the ligand list while its copies remain in the
        # per-copy roster. With no surviving chemistry template it builds no row --
        # and its homeless copies never fabricate one. CLR is unaffected.
        best = {
            "ligands": [_sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True)],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F")],
                    "PLM": [_inst("R", "606", "J")],  # pruned from ligands
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:606", SITE_REF_MEMBRANE_FACING),  # PLM copy, no home component
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("PLM", SITE_REF_MEMBRANE_FACING, is_functional=False),
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [("CLR", SITE_REF_ALLOSTERIC_7TM, True)]

    def test_all_unknown_component_passes_through_unchanged(self):
        # A real drug whose copies all voted 'unknown' has no real-site group, so
        # it is passed through untouched -- never dropped for want of a placed copy.
        drug = _sm_lig("ZMA", SITE_REF_ORTHOSTERIC, is_functional=None, name="antagonist")
        best = {
            "ligands": [drug],
            "oligomer_analysis": {"nonpolymer_instance_index": {"ZMA": [_inst("A", "500", "B")]}},
        }
        mv = {
            "ligand_copies": [_pc("A:500", SITE_REF_UNKNOWN)],
            "ligands": [_sm_lig("ZMA", SITE_REF_ORTHOSTERIC, is_functional=None)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert best["ligands"] == [drug]  # identical object, untouched

    def test_partial_unknown_keeps_real_site_and_leaves_unknown_to_surface(self):
        # Some copies voted a real site, one voted 'unknown'. The real-site row is
        # rebuilt; the unknown copy is not dropped -- it stays in the shipped
        # per-copy list for the CSV writer's homeless bucket to surface.
        best = {
            "ligands": [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True)],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F"), _inst("R", "603", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_MEMBRANE_FACING),
                _pc("R:603", SITE_REF_UNKNOWN),
            ],
            "ligands": [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [("CLR", SITE_REF_MEMBRANE_FACING, True)]
        assert best["ligand_copies"] == mv["ligand_copies"]  # unknown copy retained

    def test_keyless_ligand_passed_through_in_place(self):
        # A keyless entity (peptide / apo, no chem_comp_id) is passed through
        # untouched and keeps its position relative to a rebuilt small molecule.
        peptide = _lig(chem_comp_id=None, name="Stalk peptide", validation_status="MATCHED_POLYMER")
        best = {
            "ligands": [
                peptide,
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),
            ],
            "oligomer_analysis": {"nonpolymer_instance_index": {"CLR": [_inst("R", "602", "F")]}},
        }
        mv = {
            "ligand_copies": [_pc("R:602", SITE_REF_MEMBRANE_FACING)],
            "ligands": [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert best["ligands"][0] is peptide
        assert _rows(best)[1] == ("CLR", SITE_REF_MEMBRANE_FACING, True)

    def test_no_per_copy_votes_leaves_record_untouched(self):
        # Pre-feature data (no per-copy list) -> the whole ligand list ships as-is.
        original = [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False)]
        best = {"ligands": list(original), "oligomer_analysis": {"nonpolymer_instance_index": {}}}
        _rebuild_small_molecule_rows_from_per_copy(best, {"ligand_copies": []})
        assert best["ligands"] == original
        assert "ligand_copies" not in best

    def test_missing_instance_index_leaves_record_untouched(self):
        # Without a nonpolymer instance index there is no copy->component map, so
        # the rebuild cannot run and the list ships unchanged.
        original = [_sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False)]
        best = {"ligands": list(original), "oligomer_analysis": {}}
        mv = {"ligand_copies": [_pc("R:602", SITE_REF_MEMBRANE_FACING)]}
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert best["ligands"] == original

    def test_combines_with_csv_writer_repartition_no_inflation(self):
        # End-to-end: the rebuilt rows plus the CSV writer's per-site partition
        # must compose -- each site row lists only its own copies, with no homeless
        # "(?)" inflation on a sibling row.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False),
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [
                        _inst("R", "602", "F"),
                        _inst("R", "603", "G"),
                        _inst("R", "604", "H"),
                    ]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:603", SITE_REF_MEMBRANE_FACING),
                _pc("R:604", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        rows = transform_for_csv("XXXX", best)["ligands.csv"]
        by_site = {r["Site"]: r for r in rows}
        assert set(by_site) == {SITE_REF_ALLOSTERIC_7TM, SITE_REF_MEMBRANE_FACING}
        assert by_site[SITE_REF_ALLOSTERIC_7TM]["Residue_seq_id"] == "R:602"
        assert by_site[SITE_REF_MEMBRANE_FACING]["Residue_seq_id"] == "R:603, R:604"
        assert not any("(?)" in r["Residue_seq_id"] for r in rows)

    def test_best_run_false_row_with_null_majority_stays_dropped(self):
        # 7D77-shaped: a post-prune row the best run judged non-functional (False)
        # whose majority verdict is "not assessed" (null). A null majority must NOT
        # overwrite the False, so the row stays dropped and its copies follow it out
        # -- the surviving sibling keeps only its own copy, no "(?)" inflation.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False),
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F"), _inst("R", "603", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:603", SITE_REF_MEMBRANE_FACING),
            ],
            # No membrane majority vote -> null -> must not lift the best-run False.
            "ligands": [_sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [
            ("CLR", SITE_REF_ALLOSTERIC_7TM, True),
            ("CLR", SITE_REF_MEMBRANE_FACING, False),  # stayed dropped, not lifted to null
        ]
        rows = transform_for_csv("XXXX", best)["ligands.csv"]
        assert [r["Site"] for r in rows] == [SITE_REF_ALLOSTERIC_7TM]
        assert rows[0]["Residue_seq_id"] == "R:602"
        assert "(?)" not in rows[0]["Residue_seq_id"]

    def test_site_pruned_from_component_that_survives_elsewhere_not_revived(self):
        # 7D76/7D77-shaped: the component survives post-prune at one site
        # (intracellular, an incidental lipid rescued as functional) but was
        # buffer-pruned at another (membrane). The per-copy votes place copies at
        # the pruned membrane site with only a null majority -- that site must NOT be
        # revived as a shipped row by borrowing the surviving site's template. Its
        # copies leave via a dropped follow-out marker (no "(?)" inflation), and the
        # marker's chain_id is derived from those copies, not the template chain.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                # PLM survives post-prune only at intracellular (rescued as functional);
                # its membrane site was buffer-pruned, so it is absent from the list.
                _sm_lig("PLM", SITE_REF_INTRACELLULAR, is_functional=True, chain_id="A")
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "PLM": [_inst("A", "401", "E"), _inst("R", "602", "G"), _inst("R", "603", "H")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("A:401", SITE_REF_INTRACELLULAR),
                _pc("R:602", SITE_REF_MEMBRANE_FACING),
                _pc("R:603", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [
                _sm_lig("PLM", SITE_REF_INTRACELLULAR, is_functional=True),
                # membrane: no majority verdict -> null; no post-prune row either.
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [
            ("PLM", SITE_REF_INTRACELLULAR, True),
            ("PLM", SITE_REF_MEMBRANE_FACING, False),  # dropped follow-out marker
        ]
        rows = transform_for_csv("XXXX", best)["ligands.csv"]
        # Only the surviving intracellular row ships; the pruned membrane site is not
        # revived, and its copies leave with the marker instead of piling on as (?).
        assert [r["Site"] for r in rows] == [SITE_REF_INTRACELLULAR]
        assert rows[0]["Residue_seq_id"] == "A:401"
        assert "(?)" not in rows[0]["Residue_seq_id"]

    def test_majority_true_row_with_no_per_copy_support_is_kept(self):
        # 8XQL/GOQ-shaped: two majority-True functional rows (allosteric + a second
        # site), but the per-copy votes only reach the second site (the allosteric
        # copy the best run filed there is, by majority, at the other site). The
        # allosteric functional row must NOT be silently deleted for want of a
        # placed copy; it is kept (with chain derived from its -- empty -- copy set).
        best = {
            "ligands": [
                _sm_lig("GOQ", SITE_REF_ALLOSTERIC_7TM, is_functional=True, chain_id="R"),
                _sm_lig("GOQ", SITE_REF_INTRACELLULAR, is_functional=True, chain_id="R"),
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "GOQ": [_inst("R", "501", "F"), _inst("R", "502", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:501", SITE_REF_INTRACELLULAR),
                _pc("R:502", SITE_REF_INTRACELLULAR),
            ],
            "ligands": [
                _sm_lig("GOQ", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("GOQ", SITE_REF_INTRACELLULAR, is_functional=True),
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert _rows(best) == [
            ("GOQ", SITE_REF_ALLOSTERIC_7TM, True),  # kept, not deleted
            ("GOQ", SITE_REF_INTRACELLULAR, True),
        ]
        by_site = {lig["site_ref"]: lig for lig in best["ligands"]}
        # The covered row's chain is derived from its two copies; the uncovered row
        # has no placed copy, so its chain is empty (matching its empty residues).
        assert by_site[SITE_REF_INTRACELLULAR]["chain_id"] == "R"
        assert by_site[SITE_REF_ALLOSTERIC_7TM]["chain_id"] == ""

    def test_rebuilt_row_chain_id_derived_from_copy_chains(self):
        # 4WW3-shaped: the post-prune template carries a single-chain chain_id, but
        # the copies grouped onto the site span two chains. The rebuilt row's
        # chain_id is re-derived from the copies' author chains (sorted, de-duped),
        # not inherited from the template.
        best = {
            "ligands": [_sm_lig("TWT", SITE_REF_MEMBRANE_FACING, is_functional=True, chain_id="B")],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "TWT": [_inst("A", "301", "F"), _inst("B", "301", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("B:301", SITE_REF_MEMBRANE_FACING),
                _pc("A:301", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [_sm_lig("TWT", SITE_REF_MEMBRANE_FACING, is_functional=True)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert best["ligands"][0]["chain_id"] == "A, B"

    def test_flagship_7e2x_cholesterol_row_contract(self):
        # 7E2X flagship: 10 cholesterol copies, one an allosteric PAM and the rest
        # membrane structural lipid. The best run split them into two rows but the
        # majority judges the membrane row non-functional. After the rebuild + CSV
        # partition the membrane row is dropped and takes its copies with it, so a
        # single cholesterol row ships (the allosteric PAM) with exactly its one
        # copy and zero "(?)" inflation.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        membrane_copies = [_inst("R", str(602 + i), chr(ord("F") + i)) for i in range(10)]
        best = {
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=None),
            ],
            "oligomer_analysis": {"nonpolymer_instance_index": {"CLR": membrane_copies}},
        }
        mv = {
            "ligand_copies": [_pc("R:602", SITE_REF_ALLOSTERIC_7TM)]
            + [_pc(f"R:{603 + i}", SITE_REF_MEMBRANE_FACING) for i in range(9)],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False),  # majority
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        rows = transform_for_csv("7E2X", best)["ligands.csv"]
        clr_rows = [r for r in rows if r["Name"] == "CLR"]
        assert len(clr_rows) == 1
        assert clr_rows[0]["Site"] == SITE_REF_ALLOSTERIC_7TM
        assert clr_rows[0]["Residue_seq_id"] == "R:602"
        assert not any("(?)" in r["Residue_seq_id"] for r in rows)

    def test_revived_row_role_and_narrative_come_from_majority_not_sibling(self):
        # 7V3Z / 7XBX-shaped: the best run kept ONE cholesterol row (allosteric_7tm,
        # a PAM with pocket-specific justification prose). The majority attributes
        # extra copies to a second site (membrane_facing) whose voted role is a
        # different value (Cofactor) and whose narrative fields are un-votable (null).
        # The revived membrane row must borrow only CHEMISTRY from the surviving
        # allosteric template -- its role.value, evidence, and site_ref_justification
        # must come from the membrane majority vote (Cofactor / cleared), never the
        # allosteric sibling's "PAM" + "groove between TM2-4" prose.
        allosteric_template = _sm_lig(
            "CLR",
            SITE_REF_ALLOSTERIC_7TM,
            is_functional=True,
            name="CHOLESTEROL",
            SMILES="C(sibling)",
            InChIKey="HVYWMOMLDIMFJA-DPAQBDIFSA-N",
            site_ref_justification="Binds a discrete pocket (groove between TM2-4).",
        )
        allosteric_template["role"] = {
            "value": "PAM",
            "confidence": "High",
            "evidence": {"reasoning": "acts as a PAM", "source": "Paper"},
        }
        allosteric_template["pharmacological_role_check"] = {
            "is_functional_ligand": True,
            "confidence": "High",
            "evidence": "described as a functional PAM",
        }
        best = {
            "ligands": [allosteric_template],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("A", "601", "F"), _inst("A", "602", "G")]
                }
            },
        }
        # The membrane majority: a DIFFERENT role value, un-votable soft fields
        # nulled by the voting stage.
        mv_membrane = _sm_lig(
            "CLR", SITE_REF_MEMBRANE_FACING, is_functional=True, site_ref_justification=None
        )
        mv_membrane["role"] = {"value": "Cofactor", "confidence": None, "evidence": None}
        mv_membrane["pharmacological_role_check"] = {
            "is_functional_ligand": True,
            "confidence": None,
            "evidence": None,
        }
        mv = {
            "ligand_copies": [
                _pc("A:601", SITE_REF_ALLOSTERIC_7TM, role="PAM"),
                _pc("A:602", SITE_REF_MEMBRANE_FACING, role="Cofactor"),
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                mv_membrane,
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        by_site = {lig["site_ref"]: lig for lig in best["ligands"]}
        membrane = by_site[SITE_REF_MEMBRANE_FACING]
        # P1: the revived row's decision fields are the membrane majority's, NOT the
        # allosteric sibling template's.
        assert membrane["role"]["value"] == "Cofactor"
        assert membrane["pharmacological_role_check"]["is_functional_ligand"] is True
        # P2: the sibling site's explanatory prose does NOT ride along.
        assert membrane["role"]["evidence"] is None
        assert membrane.get("site_ref_justification") is None
        assert membrane["pharmacological_role_check"]["evidence"] is None
        assert "TM2-4" not in json.dumps(membrane)
        # Chemistry (component-intrinsic) still comes from the surviving template.
        assert membrane["name"] == "CHOLESTEROL"
        assert membrane["SMILES"] == "C(sibling)"
        assert membrane["InChIKey"] == "HVYWMOMLDIMFJA-DPAQBDIFSA-N"
        # The surviving anchor row is untouched (its own site's decision + prose).
        assert by_site[SITE_REF_ALLOSTERIC_7TM]["role"]["value"] == "PAM"

    def test_dropped_marker_narrative_cleared_when_no_majority_entry(self):
        # A dropped follow-out marker built for a site with NO majority ligand entry
        # (a structural lipid the prune removed) must not display the borrowed
        # sibling site's role/prose on the curator panel: role is cleared to None and
        # every narrative field is empty, even though it borrows the sibling chemistry.
        template = _sm_lig(
            "PLM",
            SITE_REF_INTRACELLULAR,
            is_functional=True,
            name="PALMITATE",
            site_ref_justification="Buried in the intracellular cavity.",
        )
        template["role"] = {
            "value": "Cofactor",
            "evidence": {"reasoning": "coordinates the pocket"},
        }
        best = {
            "ligands": [template],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "PLM": [_inst("A", "401", "E"), _inst("R", "602", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("A:401", SITE_REF_INTRACELLULAR),
                _pc("R:602", SITE_REF_MEMBRANE_FACING),
            ],
            # No PLM:membrane_facing vote -> null -> the marker gets cleared narrative.
            "ligands": [_sm_lig("PLM", SITE_REF_INTRACELLULAR, is_functional=True)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        marker = {lig["site_ref"]: lig for lig in best["ligands"]}[SITE_REF_MEMBRANE_FACING]
        assert marker["pharmacological_role_check"]["is_functional_ligand"] is False  # dropped
        assert marker["role"] is None
        assert marker.get("site_ref_justification") is None
        assert "intracellular cavity" not in json.dumps(marker)
        assert marker["name"] == "PALMITATE"  # chemistry still borrowed

    def test_rebuild_does_not_mutate_majority_votes(self):
        # The rebuild consumes majority_votes to source per-site decisions but must
        # NEVER mutate it: the same vote structure feeds the discrepancy pass /
        # voting log, and an in-place edit here would corrupt that shared state.
        best = {
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=False),  # outlier
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F"), _inst("R", "603", "G")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:602", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:603", SITE_REF_MEMBRANE_FACING),
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),  # revives sibling
            ],
        }
        mv_before = copy.deepcopy(mv)
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        assert mv == mv_before  # untouched, deep-equal

    def test_discrepancy_gate_survives_rebuild_value_flip(self):
        # P3 core: mirror the runner's step 9 -> step 10c order. The best run judged
        # a ligand non-functional (is_functional False); the majority says True. The
        # discrepancy is computed FIRST (step 9). Then the rebuild (step 10c) stamps
        # the shipped row to the majority True -- making the shipped value AGREE with
        # the majority. The already-computed discrepancy must still be present and
        # unchanged: the review gate must NOT vanish just because the rebuild
        # reconciled the shipped value.
        best = {
            "ligands": [
                # best-run outlier verdict: non-functional (role matches majority so
                # the ONLY discrepancy is the is_functional verdict).
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=False)
            ],
            "oligomer_analysis": {"nonpolymer_instance_index": {"CLR": [_inst("R", "602", "F")]}},
        }
        mv = {
            "ligand_copies": [_pc("R:602", SITE_REF_ALLOSTERIC_7TM)],
            "ligands": [
                # majority verdict disagrees with the best run: functional.
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True)
            ],
        }

        # Step 9: discrepancies computed on the PRE-rebuild best run.
        discrepancies = find_discrepancies(best, mv, {})
        isfunc_disc = [d for d in discrepancies if d["path"].endswith("is_functional_ligand")]
        assert len(isfunc_disc) == 1
        assert isfunc_disc[0]["best_run_value"] is False
        assert isfunc_disc[0]["majority_vote_value"] is True
        discrepancies_snapshot = copy.deepcopy(discrepancies)

        # Step 10c: the rebuild reconciles the shipped row to the majority verdict.
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        prc = best["ligands"][0]["pharmacological_role_check"]
        assert prc["is_functional_ligand"] is True  # shipped value now agrees

        # The gate is unchanged: the disagreement the curator must see is still there.
        assert discrepancies == discrepancies_snapshot
        assert [d for d in discrepancies if d["path"].endswith("is_functional_ligand")]
