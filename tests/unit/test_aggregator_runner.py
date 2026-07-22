"""Unit tests for aggregator runner helpers."""

from __future__ import annotations

import copy
import json

import pytest

from gpcr_tools.aggregator.runner import (
    _assert_boundary_matcher_sound,
    _build_validation_report,
    _coupling_protomer,
    _mark_multi_copy_ligand_gating,
    _multi_copy_alert_component,
    _multi_copy_site_divergence,
    _path_covered,
    _prune_excluded_buffer_ligands,
    _rebuild_small_molecule_rows_from_per_copy,
    _reconcile_source_discrepancies,
    _shipped_base_prefixes,
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


def _report_native(chimera_result, ai_uniprot, monkeypatch, *, is_chimeric=None):
    """Run the report with a G protein whose alpha subunit carries the model's
    slug (and optionally the model's is_chimeric flag), returning the report."""
    monkeypatch.setattr("gpcr_tools.aggregator.runner.validate_all", lambda *a, **k: [])
    _no_detect_signals(monkeypatch)
    g_protein: dict = {"alpha_subunit": {"uniprot_entry_name": ai_uniprot}}
    if is_chimeric is not None:
        g_protein["is_chimeric"] = is_chimeric
    best = {"signaling_partners": {"g_protein": g_protein}}
    return _build_validation_report("X", best, {}, [], chimera_result, None)


class TestGAlphaSubtypeDowngrade:
    """A native, family-consistent G protein whose subtype is structurally
    inseparable (identical alpha5) is advisory, not a gating chimera review.
    The downgrade is keyed on deterministic signals only."""

    def _native_gi(self, **overrides):
        base = dict(
            family="Gi/o",
            family_confident=True,
            subtype_resolution=CHIMERA_SUBTYPE_INSEPARABLE_SET,
            candidate_set=["gnai1_human", "gnai2_human"],
            score=11,
            a5_tail="IKNNLKDCGLF",
            backbone_slug="gnai1_human",
            backbone_family="Gi/o",
            is_alpha5_graft=False,
        )
        base.update(overrides)
        return _success(**base)

    def test_native_family_verified_downgrades_to_advisory_note(self, monkeypatch):
        from gpcr_tools.config import ALERT_PREFIX_GALPHA_SUBTYPE_UNRESOLVED

        chim = self._native_gi()
        report = _report_native(chim, "gnai1_human", monkeypatch)
        # Advisory note, never a gating critical warning.
        assert any(ALERT_PREFIX_GALPHA_SUBTYPE_UNRESOLVED in n for n in report["detector_notes"])
        assert not any("[CHIMERIC G PROTEIN]" in w for w in report["critical_warnings"])
        # The path anchor is preserved for downstream routing.
        note = next(n for n in report["detector_notes"] if "SUBTYPE UNRESOLVED" in n)
        assert "signaling_partners.g_protein.alpha_subunit" in note
        # Nothing else gates this fixture.
        assert report["critical_warnings"] == []
        assert report["algo_conflicts"] == []

    def test_downgrade_blocked_by_model_chimera_flag(self, monkeypatch):
        # A model-declared chimera stays a gating review even when family-verified
        # and backbone-consistent: the downgrade is never keyed on the AI flag alone.
        chim = self._native_gi()
        report = _report_native(chim, "gnai1_human", monkeypatch, is_chimeric=True)
        assert any(
            "[CHIMERIC G PROTEIN]" in w and "cannot distinguish the subtype" in w
            for w in report["critical_warnings"]
        )
        assert not any("SUBTYPE UNRESOLVED" in n for n in report["detector_notes"])

    def test_downgrade_blocked_when_no_deposited_backbone_slug(self, monkeypatch):
        # No attached G-alpha slug on the entity -> no single family-consistent
        # deposited backbone -> stays gating.
        chim = self._native_gi(backbone_slug=None, backbone_family=None)
        report = _report_native(chim, "gnai1_human", monkeypatch)
        assert any("[CHIMERIC G PROTEIN]" in w for w in report["critical_warnings"])
        assert not any("SUBTYPE UNRESOLVED" in n for n in report["detector_notes"])

    def test_downgrade_blocked_when_backbone_family_differs(self, monkeypatch):
        # The deposited scaffold is a different family (alpha5-graft) -> stays
        # gating; the functional identity still follows the alpha5.
        chim = self._native_gi(
            backbone_slug="gnas2_human", backbone_family="Gs", is_alpha5_graft=True
        )
        report = _report_native(chim, "gnai1_human", monkeypatch)
        assert any("[CHIMERIC G PROTEIN]" in w for w in report["critical_warnings"])
        assert not any("SUBTYPE UNRESOLVED" in n for n in report["detector_notes"])

    def test_construct_name_only_stays_gated(self, monkeypatch):
        # Family is NOT verified (no model slug to confirm it) -> the family
        # review stays gating regardless of a consistent backbone.
        chim = self._native_gi()
        report = _report_native(chim, None, monkeypatch)
        assert any(
            "[CHIMERIC G PROTEIN]" in w and "cannot distinguish the subtype" in w
            for w in report["critical_warnings"]
        )
        assert not any("SUBTYPE UNRESOLVED" in n for n in report["detector_notes"])

    def test_non_human_ortholog_uses_species_prefix_and_gates(self, monkeypatch):
        from gpcr_tools.config import ALERT_PREFIX_GALPHA_SPECIES_UNVERIFIED

        chim = self._native_gi(
            subtype_resolution=CHIMERA_SUBTYPE_FAMILY_ONLY,
            candidate_set=["gnai1_human", "gnai1_rat"],
        )
        report = _report_native(chim, "gnai1_human", monkeypatch)
        # Renamed prefix, still a gating critical warning; anchor preserved.
        assert any(
            ALERT_PREFIX_GALPHA_SPECIES_UNVERIFIED in w
            and "signaling_partners.g_protein.alpha_subunit" in w
            for w in report["critical_warnings"]
        )
        assert not any("[CHIMERIC G PROTEIN]" in w for w in report["critical_warnings"])
        assert not any("SUBTYPE UNRESOLVED" in n for n in report["detector_notes"])


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
    so they never reach the curator or the CSV -- unless the model explicitly
    judged one a functional ligand, in which case it is rescued. That rescue is a
    defensive branch: the exclude list and the incidental roster are disjoint, so
    an excluded buffer carrying a functional verdict is not expected today."""

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

    def test_excluded_buffer_with_functional_verdict_survives(self):
        # Defensive rescue branch: an excluded buffer the model judged a real
        # functional ligand (is_functional_ligand True) is kept despite its
        # EXCLUDED_BUFFER tag. No molecule triggers this in production today (the
        # exclude list and the incidental roster are disjoint), so it guards the
        # invariant rather than a live case.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="LMT",
                    validation_status=VALIDATION_EXCLUDED_BUFFER,
                    pharmacological_role_check={"is_functional_ligand": True},
                )
            ]
        }
        _prune_excluded_buffer_ligands(best)
        assert _comp_ids(best) == ["LMT"]

    def test_excluded_buffer_without_functional_verdict_dropped(self):
        # An excluded buffer whose role check does not affirm a functional ligand
        # (is_functional_ligand False) is NOT rescued -> the `is True` identity
        # check gates the rescue, so the buffer prune drops it.
        best = {
            "ligands": [
                _lig(
                    chem_comp_id="LMT",
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


def _pc(copy_id, site, role="unknown", confidence="Medium"):
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
        # role 'unknown' is a neutral vehicle for the per-copy is_functional logic:
        # it survives a null verdict and leaves ligands on prc=False, without the
        # role=Cofactor short-circuit to aux that would mask the verdict under test.
        role={"value": "unknown"},
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
        # PLM has a copy in the per-copy roster but no row in the shipped ligand
        # list, so there is no surviving chemistry template for it. It therefore
        # builds no row -- and its homeless copy never fabricates one. CLR is
        # unaffected.
        best = {
            "ligands": [_sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True)],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "602", "F")],
                    "PLM": [_inst("R", "606", "J")],  # no row in the ligand list
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

    def test_partial_unknown_gets_its_own_clean_unknown_row(self):
        # Some copies voted a real site, one voted 'unknown'. The real-site row is
        # rebuilt; the un-sited copy is gathered into ONE honest Site=unknown row of
        # the same compound (chemistry from the template, per-site decision cleared to
        # "not assessed", role from the copies' majority, chain from the copies)
        # rather than tagged onto the sibling. Its residue stays clean in the CSV.
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
        assert _rows(best) == [
            ("CLR", SITE_REF_MEMBRANE_FACING, True),
            ("CLR", SITE_REF_UNKNOWN, None),  # honest unknown-site row, not assessed
        ]
        unknown_row = best["ligands"][1]
        assert unknown_row["chain_id"] == "R"  # from the un-sited copy, not the template
        assert (unknown_row.get("role") or {}).get("value") == "unknown"  # copies' majority
        assert best["ligand_copies"] == mv["ligand_copies"]  # unknown copy retained

    def test_unknown_copies_reuse_existing_unknown_row(self):
        # The best run already emitted a Site=unknown row for the compound (carrying a
        # STALE sibling role "PAM"). The un-sited copies attach to THAT row rather than
        # spawning a duplicate; its chain is re-derived from them AND its role/soft
        # fields are recomputed from the copies' own majority (here "Cofactor"),
        # exactly as the freshly-built branch does -- no stale sibling prose survives.
        best = {
            "ligands": [
                _sm_lig("CA", SITE_REF_ORTHOSTERIC, is_functional=True, chain_id="A"),
                _lig(
                    chem_comp_id="CA",
                    site_ref=SITE_REF_UNKNOWN,
                    validation_status=VALIDATION_MATCHED_SMALL_MOLECULE,
                    pharmacological_role_check=None,
                    chain_id="",
                    role={"value": "PAM"},  # stale sibling role that must be overwritten
                    site_ref_justification="borrowed prose",
                ),
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CA": [_inst("A", "1", "K"), _inst("B", "2", "L"), _inst("C", "3", "M")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("A:1", SITE_REF_ORTHOSTERIC),
                _pc("B:2", SITE_REF_UNKNOWN, role="Cofactor"),
                _pc("C:3", SITE_REF_UNKNOWN, role="Cofactor"),
            ],
            "ligands": [
                _sm_lig("CA", SITE_REF_ORTHOSTERIC, is_functional=True),
                _sm_lig("CA", SITE_REF_UNKNOWN, is_functional=None),
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        # Exactly one unknown row (reused, not duplicated).
        assert _rows(best).count(("CA", SITE_REF_UNKNOWN, None)) == 1
        by_site = {lig["site_ref"]: lig for lig in best["ligands"]}
        unknown_row = by_site[SITE_REF_UNKNOWN]
        assert unknown_row["chain_id"] == "B, C"
        # Role now = the copies' majority, NOT the stale "PAM"; soft prose cleared.
        assert (unknown_row.get("role") or {}).get("value") == "Cofactor"
        assert unknown_row.get("site_ref_justification") is None

    def test_unknown_copies_build_clean_unknown_row_end_to_end(self):
        # 7M3F CaSR-shaped: 4 calcium copies -- 2 voted a real site, 2 voted
        # 'unknown'. After the rebuild + CSV partition the two un-sited copies sit on
        # a clean Site=unknown row (chemistry from the template, residue tokens clean,
        # no "(?)"), NOT tagged onto the real-site row.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                _sm_lig("CA", SITE_REF_ORTHOSTERIC, is_functional=None, name="Calcium ion")
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CA": [
                        _inst("A", "907", "L"),
                        _inst("B", "908", "W"),
                        _inst("A", "908", "M"),
                        _inst("A", "909", "N"),
                    ]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("A:907", SITE_REF_ORTHOSTERIC),
                _pc("B:908", SITE_REF_ORTHOSTERIC),
                _pc("A:908", SITE_REF_UNKNOWN),
                _pc("A:909", SITE_REF_UNKNOWN),
            ],
            "ligands": [_sm_lig("CA", SITE_REF_ORTHOSTERIC, is_functional=None)],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        rows = transform_for_csv("7M3F", best)["ligands.csv"]
        by_site = {r["Site"]: r for r in rows}
        assert set(by_site) == {SITE_REF_ORTHOSTERIC, SITE_REF_UNKNOWN}
        assert by_site[SITE_REF_ORTHOSTERIC]["Residue_seq_id"] == "A:907, B:908"
        assert by_site[SITE_REF_UNKNOWN]["Residue_seq_id"] == "A:908, A:909"
        assert by_site[SITE_REF_UNKNOWN]["Name"] == "CA"  # chemistry from the template
        assert by_site[SITE_REF_UNKNOWN]["ChainID"] == "A"
        assert not any("(?)" in r["Residue_seq_id"] for r in rows)

    def test_all_nonfunctional_component_emits_no_unknown_row(self):
        # 6D26/SIN-shaped: a buffer the majority judged non-functional at every real
        # site has NO surviving row. Its copies voted to those (dropped) sites follow
        # them out, and -- crucially -- the copies voted 'unknown' do NOT reappear as
        # a lone Site=unknown row: with no surviving row, the whole molecule (and its
        # un-sited copies) drops, consistent with "non-functional -> not emitted".
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                _sm_lig("SIN", SITE_REF_MEMBRANE_FACING, is_functional=False),
                _sm_lig("SIN", SITE_REF_INTRACELLULAR, is_functional=False),
                _sm_lig("FSY", SITE_REF_ORTHOSTERIC, is_functional=None, name="agonist"),
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "SIN": [
                        _inst("A", "2411", "L"),
                        _inst("A", "2412", "M"),
                        _inst("A", "2414", "O"),
                        _inst("A", "2415", "P"),
                    ],
                    "FSY": [_inst("A", "2401", "B")],
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("A:2401", SITE_REF_ORTHOSTERIC),
                _pc("A:2411", SITE_REF_MEMBRANE_FACING),
                _pc("A:2412", SITE_REF_INTRACELLULAR),
                _pc("A:2414", SITE_REF_UNKNOWN),
                _pc("A:2415", SITE_REF_UNKNOWN),
            ],
            "ligands": [
                _sm_lig("SIN", SITE_REF_MEMBRANE_FACING, is_functional=False),
                _sm_lig("SIN", SITE_REF_INTRACELLULAR, is_functional=False),
                _sm_lig("FSY", SITE_REF_ORTHOSTERIC, is_functional=None),
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        # No SIN unknown row is fabricated; SIN keeps only its two dropped markers.
        assert ("SIN", SITE_REF_UNKNOWN, None) not in _rows(best)
        rows = transform_for_csv("6D26", best)["ligands.csv"]
        # In the CSV, SIN vanishes entirely (all rows dropped, un-sited copies with
        # it); only the surviving FSY row ships, with no "(?)" anywhere.
        assert [r["Name"] for r in rows] == ["FSY"]
        assert not any("(?)" in r["Residue_seq_id"] for r in rows)

    def test_unknown_row_built_from_surviving_template_never_silently_dropped(self):
        # The component's FIRST post-prune row is a GHOST (dropped by the CSV), but it
        # also has a surviving functional row. The unknown row must borrow chemistry
        # from the SURVIVING row -- not the ghost -- so the new unknown row is not
        # itself eaten by ligand_row_dropped, and its un-sited copy is never lost.
        from gpcr_tools.config import VALIDATION_GHOST_LIGAND
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        ghost = _lig(
            chem_comp_id="CLR",
            site_ref=SITE_REF_ALLOSTERIC_7TM,
            validation_status=VALIDATION_GHOST_LIGAND,
            role={"value": "Cofactor"},
            pharmacological_role_check={"is_functional_ligand": True},
        )
        best = {
            "ligands": [
                ghost,  # first row of the component -> the naive chemistry template
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),
            ],
            "oligomer_analysis": {
                "nonpolymer_instance_index": {
                    "CLR": [_inst("R", "601", "F"), _inst("R", "602", "G"), _inst("R", "603", "H")]
                }
            },
        }
        mv = {
            "ligand_copies": [
                _pc("R:601", SITE_REF_ALLOSTERIC_7TM),
                _pc("R:602", SITE_REF_MEMBRANE_FACING),
                _pc("R:603", SITE_REF_UNKNOWN),
            ],
            "ligands": [
                _sm_lig("CLR", SITE_REF_ALLOSTERIC_7TM, is_functional=True),
                _sm_lig("CLR", SITE_REF_MEMBRANE_FACING, is_functional=True),
            ],
        }
        _rebuild_small_molecule_rows_from_per_copy(best, mv)
        unknown_rows = [lig for lig in best["ligands"] if lig.get("site_ref") == SITE_REF_UNKNOWN]
        assert len(unknown_rows) == 1
        # Built from the surviving row's chemistry, so NOT a ghost -> survives the CSV.
        assert unknown_rows[0].get("validation_status") != VALIDATION_GHOST_LIGAND
        rows = transform_for_csv("XXXX", best)["ligands.csv"]
        by_site = {r["Site"]: r for r in rows}
        assert by_site[SITE_REF_UNKNOWN]["Residue_seq_id"] == "R:603"  # copy not lost
        assert not any("(?)" in r["Residue_seq_id"] for r in rows)

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
        # 7D76/7D77-shaped: the component has a row at one site (intracellular, a
        # matched lipid the model kept as functional) but no row at another
        # (membrane). The per-copy votes place copies at the membrane site
        # with only a null majority -- that site must NOT be revived as a shipped
        # row by borrowing the surviving site's template. Its copies leave via a
        # dropped follow-out marker (no "(?)" inflation), and the marker's chain_id
        # is derived from those copies, not the template chain.
        from gpcr_tools.csv_generator.csv_writer import transform_for_csv

        best = {
            "ligands": [
                # PLM has a row only at intracellular (a matched lipid kept as
                # functional); the membrane site has no row in the list.
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
        # (a lipid absent from the surviving ligand list) must not display the borrowed
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


class TestSourceSideReconcile:
    """A gating controversy no shipped entity owns is downgraded to advisory; a
    controversy a shipped entity does cover keeps gating (reconciled at the
    base-compound level so a still-shipping compound whose site changed keeps its
    gate), and the bracket-safe prefix test never mis-slices a peptide name that
    contains brackets.
    """

    def test_unreachable_gating_controversy_downgraded(self) -> None:
        best = {"ligands": [{"chem_comp_id": "ATP", "role": {"value": "agonist"}}]}
        discrepancies = [
            {"path": "ligands[ATP].role.value", "gating": True},
            {"path": "ligands[GONE].role.value", "gating": True},
        ]
        _reconcile_source_discrepancies(best, discrepancies)
        by_path = {d["path"]: d for d in discrepancies}
        # The shipped ATP row keeps its gate; the vanished GONE row has nothing to gate.
        assert by_path["ligands[ATP].role.value"].get("gating", True) is True
        assert by_path["ligands[GONE].role.value"]["gating"] is False

    def test_same_compound_new_site_keeps_gating(self) -> None:
        # The must-fix scenario: a step-10c rebuild rewrote a still-shipping
        # compound's site_ref (HEM shipped now at 'allosteric'), while the gating
        # site_ref controversy was recorded PRE-rebuild against the old site
        # (HEM:orthosteric). The compound still ships at a site, so the contested
        # site is a genuine shipped error and MUST keep gating -- reconciling on the
        # full site-qualified identity would wrongly read HEM as "dropped".
        best = {"ligands": [{"chem_comp_id": "HEM", "site_ref": "allosteric"}]}
        discrepancies = [{"path": "ligands[HEM:orthosteric].site_ref", "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0].get("gating", True) is True

    def test_compound_dropped_entirely_downgraded(self) -> None:
        # A controversy on a compound that no longer ships at ANY site is downgraded.
        best = {"ligands": [{"chem_comp_id": "HEM", "site_ref": "allosteric"}]}
        discrepancies = [{"path": "ligands[ATP:orthosteric].site_ref", "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0]["gating"] is False

    def test_substring_compound_id_not_confused(self) -> None:
        # HEM shipped must not cover a controversy on the longer id HEME (the base
        # id is a strict prefix but the boundary char after it is a name char).
        best = {"ligands": [{"chem_comp_id": "HEM", "site_ref": "orthosteric"}]}
        discrepancies = [{"path": "ligands[HEME:orthosteric].site_ref", "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0]["gating"] is False

    def test_bracketed_peptide_name_prefix_not_downgraded(self) -> None:
        # A peptide whose name contains brackets ([Sar1,Ile8]-Angiotensin II) is a
        # keyless ligand; its identity carries those brackets. A gating controversy
        # on it must NOT be downgraded (a naive bracket-strip would wrongly clear it).
        name = "[Sar1,Ile8]-Angiotensin II"
        best = {"ligands": [{"chem_comp_id": "None", "name": name, "type": "peptide"}]}
        (open_prefix,) = _shipped_base_prefixes(best)  # exactly one shipped ligand
        path = f"{open_prefix}].pubchem_id"
        discrepancies = [{"path": path, "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0].get("gating", True) is True

    def test_bracketed_peptide_sibling_downgraded(self) -> None:
        # The lookalike sibling (Angiotensin III) does not ship, so a controversy
        # on it is downgraded -- proving II and III are told apart despite the
        # shared bracketed prefix.
        best = {
            "ligands": [
                {"chem_comp_id": "None", "name": "[Sar1,Ile8]-Angiotensin II", "type": "peptide"}
            ]
        }
        (open_prefix,) = _shipped_base_prefixes(best)
        sibling_path = open_prefix.replace("angiotensin ii", "angiotensin iii") + "].pubchem_id"
        discrepancies = [{"path": sibling_path, "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0]["gating"] is False

    def test_path_covered_boundary_semantics(self) -> None:
        prefixes = {"ligands[HEM"}
        # Site suffix (':') and identity close (']') are boundaries; a name char is not.
        assert _path_covered("ligands[HEM:allosteric].site_ref", prefixes) is True
        assert _path_covered("ligands[HEM].site_ref", prefixes) is True
        assert _path_covered("ligands[HEME:allosteric].site_ref", prefixes) is False

    def test_fail_closed_self_check_catches_broken_matcher(self) -> None:
        # The self-check must be able to FAIL: a matcher that ignores the boundary
        # (matches on bare startswith) would clear a real gate, so the guard trips.
        prefixes = {"ligands[HEM"}
        # A sound matcher passes the guard.
        _assert_boundary_matcher_sound(prefixes)
        import gpcr_tools.aggregator.runner as runner_mod

        original = runner_mod._path_covered
        runner_mod._path_covered = lambda path, ps: any(path.startswith(p) for p in ps)  # type: ignore[assignment]
        try:
            with pytest.raises(AssertionError):
                _assert_boundary_matcher_sound(prefixes)
        finally:
            runner_mod._path_covered = original  # type: ignore[assignment]

    def test_self_check_non_vacuous_when_nothing_shipped(self) -> None:
        # The guard must not go vacuous when the shipped set is empty -- that is
        # exactly when every list-path controversy is downgraded, so a broken
        # matcher there would silently clear real gates. With no shipped prefixes,
        # a sound matcher still passes and a bare-startswith matcher still trips.
        _assert_boundary_matcher_sound(set())  # sound matcher, empty set: passes
        import gpcr_tools.aggregator.runner as runner_mod

        original = runner_mod._path_covered
        runner_mod._path_covered = lambda path, ps: any(path.startswith(p) for p in ps)  # type: ignore[assignment]
        try:
            with pytest.raises(AssertionError):
                _assert_boundary_matcher_sound(set())
        finally:
            runner_mod._path_covered = original  # type: ignore[assignment]

    def test_self_check_catches_site_boundary_blind_matcher(self) -> None:
        # The load-bearing case: a matcher that accepts the closing ']' boundary but
        # is BLIND to the ':' (site-qualified) boundary. It would pass a self-check
        # that never probes ':', yet it would read a still-shipping compound whose
        # site was rewritten by the step-10c rebuild (HEM:orthosteric under a
        # shipped HEM) as "dropped" and clear a genuine site conflict. The guard
        # must trip on it -- for the empty set and for a shipped prefix alike.
        import gpcr_tools.aggregator.runner as runner_mod

        def _site_blind(path: str, ps) -> bool:
            # Accepts only the ']' boundary; drops the ':' continuation.
            return any(path.startswith(p) and path[len(p) : len(p) + 1] == "]" for p in ps)

        original = runner_mod._path_covered
        runner_mod._path_covered = _site_blind  # type: ignore[assignment]
        try:
            with pytest.raises(AssertionError):
                _assert_boundary_matcher_sound(set())
            with pytest.raises(AssertionError):
                _assert_boundary_matcher_sound({"ligands[HEM"})
        finally:
            runner_mod._path_covered = original  # type: ignore[assignment]

    def test_pre_vs_post_rebuild_site_conflict_kept_gating_end_to_end(self) -> None:
        # The scenario the ':' guard protects, exercised through the public entry
        # point: HEM ships at 'allosteric' (rewritten by the rebuild) while the
        # gating site_ref controversy was recorded PRE-rebuild against the old site
        # (HEM:orthosteric). The compound still ships, so the contested site is a
        # genuine shipped error and must KEEP gating.
        best = {"ligands": [{"chem_comp_id": "HEM", "site_ref": "allosteric"}]}
        discrepancies = [{"path": "ligands[HEM:orthosteric].site_ref", "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0].get("gating", True) is True

    def test_advisory_controversy_left_untouched(self) -> None:
        # A record already advisory (gating=False) is not re-examined.
        best: dict = {"ligands": []}
        discrepancies = [{"path": "ligands[GONE].role.value", "gating": False}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0]["gating"] is False

    def test_non_list_paths_untouched(self) -> None:
        # A scalar/top-level controversy is out of scope and keeps its verdict.
        best: dict = {"ligands": []}
        discrepancies = [{"path": "structure_info.state.value", "gating": True}]
        _reconcile_source_discrepancies(best, discrepancies)
        assert discrepancies[0].get("gating", True) is True


def _mc_alert(comp):
    """A MULTI_COPY_LIGAND oligomer alert carrying the ligands[<comp>] anchor path."""
    return {
        "type": ALERT_MULTI_COPY_LIGAND,
        "message": (
            f"[MULTI_COPY_LIGAND] at 'ligands[{comp}]': modelled in 2 copies "
            f"(instances D, E); one annotation row may hide copies at distinct "
            f"sites or with distinct roles. Human review recommended."
        ),
    }


class TestMultiCopySiteDivergence:
    """The pure per-component decision: a multi-copy ligand gates only when its
    joined copies sit at more than one distinct binding site. Fail-closed on thin
    or unattributable evidence."""

    def test_same_site_across_copies_is_advisory(self):
        assert (
            _multi_copy_site_divergence([SITE_REF_MEMBRANE_FACING, SITE_REF_MEMBRANE_FACING])
            is False
        )

    def test_distinct_sites_gate(self):
        assert _multi_copy_site_divergence([SITE_REF_ORTHOSTERIC, SITE_REF_INTRACELLULAR]) is True

    def test_known_versus_empty_gates(self):
        # Fail-closed: an absent/blank site is its own distinct value, so a
        # known-vs-blank pair is treated as divergent.
        assert _multi_copy_site_divergence([SITE_REF_ORTHOSTERIC, None]) is True
        assert _multi_copy_site_divergence([SITE_REF_ORTHOSTERIC, ""]) is True

    def test_all_empty_is_one_value_advisory(self):
        # Every copy blank -> a single distinct value -> advisory (None and "" fold
        # to the same blank token).
        assert _multi_copy_site_divergence([None, ""]) is False
        assert _multi_copy_site_divergence([None, None]) is False

    def test_fewer_than_two_copies_fail_closed(self):
        assert _multi_copy_site_divergence([SITE_REF_ORTHOSTERIC]) is True
        assert _multi_copy_site_divergence([]) is True
        assert _multi_copy_site_divergence(None) is True

    def test_three_copies_one_outlier_gate(self):
        assert (
            _multi_copy_site_divergence(
                [SITE_REF_MEMBRANE_FACING, SITE_REF_MEMBRANE_FACING, SITE_REF_ORTHOSTERIC]
            )
            is True
        )


class TestMultiCopyAlertComponent:
    """The component id is recovered from the alert's ligands[<comp>] path."""

    def test_extracts_component(self):
        assert _multi_copy_alert_component(_mc_alert("CLR")) == "CLR"
        assert _multi_copy_alert_component(_mc_alert("A1AEI")) == "A1AEI"

    def test_no_path_returns_none(self):
        assert _multi_copy_alert_component({"type": ALERT_MULTI_COPY_LIGAND, "message": ""}) is None
        assert _multi_copy_alert_component({"message": "no path here"}) is None


class TestMarkMultiCopyLigandGating:
    """The marker stamps each MULTI_COPY_LIGAND alert's ``gating`` flag from the
    aggregated per-copy site attribution, joined through the oligomer roster. It
    runs after the per-copy rebuild, so it reads the final ``ligand_copies``."""

    def _best(self, comp, insts, copies, extra_alerts=None):
        alerts = [_mc_alert(comp)]
        if extra_alerts:
            alerts.extend(extra_alerts)
        return {
            "oligomer_analysis": {
                "alerts": alerts,
                "nonpolymer_instance_index": {comp: insts},
            },
            "ligand_copies": copies,
        }

    def _flag(self, best, comp="CLR"):
        for a in best["oligomer_analysis"]["alerts"]:
            if a.get("type") == ALERT_MULTI_COPY_LIGAND and _multi_copy_alert_component(a) == comp:
                return a.get("gating")
        raise AssertionError("alert not found")

    def test_same_site_marked_advisory(self):
        best = self._best(
            "CLR",
            [_inst("A", "1201", "D"), _inst("A", "1202", "E")],
            [
                _pc("A:1201", SITE_REF_MEMBRANE_FACING),
                _pc("A:1202", SITE_REF_MEMBRANE_FACING),
            ],
        )
        _mark_multi_copy_ligand_gating(best)
        assert self._flag(best) is False

    def test_distinct_sites_marked_gating(self):
        best = self._best(
            "BU1",
            [_inst("A", "409", "J"), _inst("A", "410", "K")],
            [
                _pc("A:409", SITE_REF_INTRACELLULAR),
                _pc("A:410", SITE_REF_MEMBRANE_FACING),
            ],
        )
        _mark_multi_copy_ligand_gating(best)
        assert self._flag(best, "BU1") is True

    def test_only_one_joinable_copy_fail_closed(self):
        # The alert saw two copies, but only one per-copy row joins back -> gate.
        best = self._best(
            "CLR",
            [_inst("A", "1201", "D"), _inst("A", "1202", "E")],
            [_pc("A:1201", SITE_REF_MEMBRANE_FACING)],
        )
        _mark_multi_copy_ligand_gating(best)
        assert self._flag(best) is True

    def test_unmappable_component_fail_closed(self):
        # No instance-index entry maps to the ligand copies -> no joinable copies.
        best = {
            "oligomer_analysis": {
                "alerts": [_mc_alert("CLR")],
                "nonpolymer_instance_index": {},
            },
            "ligand_copies": [
                _pc("A:1201", SITE_REF_MEMBRANE_FACING),
                _pc("A:1202", SITE_REF_MEMBRANE_FACING),
            ],
        }
        _mark_multi_copy_ligand_gating(best)
        assert self._flag(best) is True

    def test_known_versus_unattributed_site_gates(self):
        best = self._best(
            "CLR",
            [_inst("A", "1201", "D"), _inst("A", "1202", "E")],
            [
                _pc("A:1201", SITE_REF_ORTHOSTERIC),
                _pc("A:1202", None),
            ],
        )
        _mark_multi_copy_ligand_gating(best)
        assert self._flag(best) is True

    def test_non_multi_copy_alerts_untouched(self):
        other = {"type": "HALLUCINATION", "message": "[HALLUCINATION] at 'receptor_info': x"}
        best = self._best(
            "CLR",
            [_inst("A", "1201", "D"), _inst("A", "1202", "E")],
            [
                _pc("A:1201", SITE_REF_MEMBRANE_FACING),
                _pc("A:1202", SITE_REF_MEMBRANE_FACING),
            ],
            extra_alerts=[other],
        )
        _mark_multi_copy_ligand_gating(best)
        assert "gating" not in other

    def test_no_oligomer_analysis_is_noop(self):
        best = {"ligand_copies": []}
        _mark_multi_copy_ligand_gating(best)  # must not raise
        assert "oligomer_analysis" not in best
