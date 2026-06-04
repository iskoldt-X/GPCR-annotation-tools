"""Tests for the pre-annotation detect stage: the signal contract, the
G-protein identity detector, and the stage runner (persist + reload)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from gpcr_tools.config import FULL_G_ALPHA_CANDIDATES, reset_config
from gpcr_tools.detector.gprotein import G_PROTEIN_LOCUS, detect_g_protein_identity
from gpcr_tools.detector.ligands import detect_incidental_candidates
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    SIGNAL_INCIDENTAL_CANDIDATE,
    DetectSignal,
    to_critical_warnings,
)
from gpcr_tools.detector.stage import load_detect_signals, run_detect, run_detect_stage
from gpcr_tools.validator.cache import SequenceCache

TRANSDUCIN_A5 = "IKENLKDCGLF"
DISTINCT_A5 = "WWWWWWWWWWW"


def _mock_refs(tail_by_slug: dict[str, str], default_tail: str = DISTINCT_A5) -> Any:
    def _fetch(accession: str, cache: Any) -> str | None:
        slug = FULL_G_ALPHA_CANDIDATES.get(accession)
        tail = tail_by_slug.get(slug, default_tail) if slug else default_tail
        return "GGGGG" + tail

    return _fetch


def _galpha_entry(sequence: str) -> dict[str, Any]:
    return {
        "polymer_entities": [
            {
                "rcsb_polymer_entity": {"pdbx_description": "G alpha subunit"},
                "entity_poly": {"pdbx_seq_one_letter_code_can": sequence},
            }
        ]
    }


def _nonpoly_entry(comp_ids: list[str]) -> dict[str, Any]:
    return {
        "nonpolymer_entities": [{"nonpolymer_comp": {"chem_comp": {"id": c}}} for c in comp_ids]
    }


_TRANSDUCIN_TAILS = dict.fromkeys(("gnat1_human", "gnat2_human", "gnat3_human"), TRANSDUCIN_A5)


class TestIncidentalCandidateLigandDetector:
    def test_incidental_candidate_molecules_emit_advisory(self) -> None:
        sigs = detect_incidental_candidates("X", _nonpoly_entry(["CLR", "PLM", "HOH"]))
        assert {s.payload["comp_id"] for s in sigs} == {"CLR", "PLM"}
        assert all(s.kind == SIGNAL_INCIDENTAL_CANDIDATE for s in sigs)
        assert all(s.severity == SEVERITY_ADVISORY for s in sigs)  # prompt routing, not review

    def test_non_incidental_candidate_ligand_not_flagged(self) -> None:
        assert detect_incidental_candidates("X", _nonpoly_entry(["RET", "HOH"])) == []

    def test_no_nonpolymer_no_signal(self) -> None:
        assert detect_incidental_candidates("X", {"polymer_entities": []}) == []


class TestDetectSignal:
    def test_dict_roundtrip(self) -> None:
        s = DetectSignal(
            kind="k", target_ref="a.b", summary="hi", payload={"x": 1}, severity=SEVERITY_REVIEW
        )
        assert DetectSignal.from_dict(s.to_dict()) == s

    def test_to_critical_warnings_only_review(self) -> None:
        sigs = [
            DetectSignal("k1", "loc1", "advisory one", severity=SEVERITY_ADVISORY),
            DetectSignal("k2", "loc2", "review two", severity=SEVERITY_REVIEW),
        ]
        assert to_critical_warnings(sigs) == ["k2 at 'loc2': review two"]


class TestSeverityFailSafe:
    """Anything that is not an explicit, recognised advisory becomes review, so
    an unclassified signal is surfaced to a human, never fed to the model."""

    def test_default_severity_is_review(self) -> None:
        # A detector that forgets to classify its signal must NOT silently get
        # advisory routing (which would feed the model prompt).
        assert DetectSignal(kind="k", target_ref="a", summary="b").severity == SEVERITY_REVIEW

    def test_unrecognised_severity_coerced_to_review(self) -> None:
        sig = DetectSignal("k", "a", "b", severity="totally-bogus")
        assert sig.severity == SEVERITY_REVIEW

    def test_empty_severity_coerced_to_review(self) -> None:
        assert DetectSignal("k", "a", "b", severity="").severity == SEVERITY_REVIEW

    def test_explicit_advisory_preserved(self) -> None:
        sig = DetectSignal("k", "a", "b", severity=SEVERITY_ADVISORY)
        assert sig.severity == SEVERITY_ADVISORY

    def test_from_dict_missing_severity_is_review(self) -> None:
        # A serialised signal from a malformed / future version with no severity
        # key must fail safe to review rather than default to advisory.
        sig = DetectSignal.from_dict({"kind": "k", "target_ref": "a", "summary": "b"})
        assert sig.severity == SEVERITY_REVIEW

    def test_from_dict_unrecognised_severity_is_review(self) -> None:
        sig = DetectSignal.from_dict(
            {"kind": "k", "target_ref": "a", "summary": "b", "severity": "weird"}
        )
        assert sig.severity == SEVERITY_REVIEW

    def test_from_dict_roundtrip_preserves_advisory(self) -> None:
        original = DetectSignal("k", "a", "b", payload={"x": 1}, severity=SEVERITY_ADVISORY)
        assert DetectSignal.from_dict(original.to_dict()) == original


class TestGProteinDetector:
    def test_transducin_emits_review_signal(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        entry = _galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs = detect_g_protein_identity("9IIX", entry, cache)
        assert len(sigs) == 1
        s = sigs[0]
        assert s.kind == SIGNAL_CHIMERIC_GPROTEIN
        assert s.target_ref == G_PROTEIN_LOCUS
        assert s.severity == SEVERITY_REVIEW
        assert s.payload["family"] == "Gi/o"
        assert "Gi/o" in s.summary

    def test_resolved_subtype_is_advisory(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        entry = _galpha_entry("MMMMMMMMMM" + target)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": target}),
        ):
            sigs = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_ADVISORY
        assert sigs[0].payload["subtype"] == "gnas2_human"

    def test_no_g_protein_no_signal(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        entry = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "Dopamine receptor D2"},
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MMMMMMMMMMMM"},
                }
            ]
        }
        assert detect_g_protein_identity("X", entry, cache) == []

    def test_low_confidence_emits_weak_review(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        entry = _galpha_entry("A" * 25)  # matches no real alpha5
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({}),  # every ref gets DISTINCT_A5 (no 'A')
        ):
            sigs = detect_g_protein_identity("X", entry, cache)
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_REVIEW
        assert "too weak" in sigs[0].summary


class TestDetectStage:
    @pytest.fixture
    def ws(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        workspace = tmp_path / "ws"
        for sub in ("enriched", "detect", "cache"):
            (workspace / sub).mkdir(parents=True)
        monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
        reset_config()
        yield workspace
        reset_config()

    def test_run_detect_persists_and_reloads(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs = run_detect("9IIX")
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_REVIEW
        assert (ws / "detect" / "9IIX.json").is_file()
        assert load_detect_signals("9IIX") == sigs

    def test_run_detect_unwraps_data_entry_envelope(self, ws: Path) -> None:
        # enriched files may be wrapped as {"data": {"entry": {...}}}.
        enveloped = {"data": {"entry": _galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5)}}
        (ws / "enriched" / "9IIX.json").write_text(json.dumps(enveloped))
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            sigs = run_detect("9IIX")
        assert len(sigs) == 1
        assert sigs[0].severity == SEVERITY_REVIEW

    def test_run_detect_unwraps_envelope_for_ligand_detector(self, ws: Path) -> None:
        # The envelope unwrap must reach the ligand detector too, not just gprotein.
        enveloped = {"data": {"entry": _nonpoly_entry(["PLM"])}}
        (ws / "enriched" / "9XYZ.json").write_text(json.dumps(enveloped))
        sigs = run_detect("9XYZ", skip_api_checks=True)
        # PLM is an incidental_candidate molecule; the unwrap must reach that ligand detector.
        assert SIGNAL_INCIDENTAL_CANDIDATE in {s.kind for s in sigs}

    def test_run_detect_missing_enriched(self, ws: Path) -> None:
        assert run_detect("NOPE") == []
        assert not (ws / "detect" / "NOPE.json").is_file()  # no file when no input

    def test_skip_api_checks_writes_empty_signal_file(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        sigs = run_detect("9IIX", skip_api_checks=True)
        assert sigs == []
        assert (ws / "detect" / "9IIX.json").is_file()  # stage output always present

    def test_metadata_detector_runs_under_skip_api(self, ws: Path) -> None:
        # The excluded-ligand detector is metadata-only, so it runs even when
        # sequence-based detectors are skipped.
        (ws / "enriched" / "9XYZ.json").write_text(json.dumps(_nonpoly_entry(["PLM"])))
        sigs = run_detect("9XYZ", skip_api_checks=True)
        # PLM fires the metadata-only incidental_candidate detector even under skip_api.
        assert SIGNAL_INCIDENTAL_CANDIDATE in {s.kind for s in sigs}

    def test_provided_cache_is_not_saved_by_run_detect(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A caller running many PDBs owns the shared cache; run_detect must not
        # save a cache it did not create.
        from gpcr_tools.config import get_config
        from gpcr_tools.validator.cache import SequenceCache

        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        cache = SequenceCache(get_config().cache_dir / "uniprot_sequence_cache.json")
        saves: list[int] = []
        monkeypatch.setattr(cache, "save", lambda: saves.append(1))
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            run_detect("9IIX", cache=cache)
        assert saves == []

    def test_run_detect_stage_persists_cache(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(_TRANSDUCIN_TAILS),
        ):
            run_detect_stage("9IIX")
        assert (ws / "cache" / "uniprot_sequence_cache.json").is_file()
