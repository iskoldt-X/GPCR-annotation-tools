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
from gpcr_tools.detector.signals import (
    SEVERITY_ADVISORY,
    SEVERITY_REVIEW,
    SIGNAL_CHIMERIC_GPROTEIN,
    DetectSignal,
    to_critical_warnings,
)
from gpcr_tools.detector.stage import load_detect_signals, run_detect
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


_TRANSDUCIN_TAILS = dict.fromkeys(("gnat1_human", "gnat2_human", "gnat3_human"), TRANSDUCIN_A5)


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

    def test_run_detect_missing_enriched(self, ws: Path) -> None:
        assert run_detect("NOPE") == []

    def test_skip_api_checks_writes_empty_signal_file(self, ws: Path) -> None:
        (ws / "enriched" / "9IIX.json").write_text(
            json.dumps(_galpha_entry("MMMMMMMMMM" + TRANSDUCIN_A5))
        )
        sigs = run_detect("9IIX", skip_api_checks=True)
        assert sigs == []
        assert (ws / "detect" / "9IIX.json").is_file()  # stage output always present
