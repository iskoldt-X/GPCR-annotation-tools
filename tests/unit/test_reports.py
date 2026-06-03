"""Tests for operational reports."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpcr_tools import reports
from gpcr_tools.config import get_config, reset_config


@pytest.fixture()
def cfg(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    return get_config()


def _write_download_log(cfg: Any, mapping: dict) -> None:
    cfg.download_log_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.download_log_file.write_text(json.dumps(mapping), encoding="utf-8")


def _write_validation(cfg: Any, pdb: str, **fields: Any) -> None:
    vdir = cfg.aggregated_dir / "validation_logs"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / f"{pdb}_validation.json").write_text(json.dumps(fields), encoding="utf-8")


class TestPdfCoverage:
    def test_empty(self, cfg: Any) -> None:
        assert "no download log" in reports.report_pdf_coverage()

    def test_groups_by_status(self, cfg: Any) -> None:
        _write_download_log(
            cfg,
            {
                "A": {"status": "success_pdf_downloaded"},
                "B": {"status": "success_pdf_downloaded"},
                "C": {"status": "fallback_paywalled"},
            },
        )
        out = reports.report_pdf_coverage()
        assert "3 PDB" in out
        assert "success_pdf_downloaded" in out
        assert "fallback_paywalled" in out


class TestFullAudit:
    def test_empty(self, cfg: Any) -> None:
        assert "no validation logs" in reports.report_full_audit()

    def test_summarises_warnings_and_conflicts(self, cfg: Any) -> None:
        _write_validation(
            cfg, "AAA", critical_warnings=["w1"], algo_conflicts=[], chimera_status="success"
        )
        _write_validation(
            cfg, "BBB", critical_warnings=[], algo_conflicts=["c1"], chimera_status="skipped"
        )
        out = reports.report_full_audit()
        assert "2 PDB" in out
        assert "AAA" in out  # has a critical warning
        assert "BBB" in out  # has an algo conflict
        assert "success" in out
        assert "skipped" in out


class TestTailAnalysis:
    def test_empty(self, cfg: Any) -> None:
        assert "no validation logs" in reports.report_tail_analysis()

    def test_score_distribution_and_flagging(self, cfg: Any) -> None:
        _write_validation(cfg, "AAA", chimera_score=11, chimera_status="success")  # clean
        _write_validation(cfg, "BBB", chimera_score=5, chimera_status="success")  # sub-anchor
        _write_validation(cfg, "CCC", chimera_score=11, chimera_status="conflict")  # non-ok
        out = reports.report_tail_analysis()
        assert "3 PDB" in out
        assert "score 11" in out
        assert "score 5" in out
        assert "Flagged for review (non-success or score < 8): 2" in out
        assert "BBB" in out
        assert "CCC" in out


def test_pdf_coverage_handles_non_dict_log(cfg: Any) -> None:
    """A corrupt-but-parseable (non-dict) download log must not crash the report."""
    _write_download_log(cfg, ["not", "a", "dict"])
    assert "no download log" in reports.report_pdf_coverage()  # degrades, no crash


def test_pdf_coverage_none_status_renders_unknown(cfg: Any) -> None:
    """A null status renders as 'unknown', not the literal string 'None'."""
    _write_download_log(cfg, {"A": {"status": None}})
    assert "unknown" in reports.report_pdf_coverage()


def test_tail_analysis_accepts_float_score(cfg: Any) -> None:
    """A float chimera score below 4 is shown and flagged, not silently ignored."""
    _write_validation(cfg, "FLT", chimera_score=3.5, chimera_status="partial")
    out = reports.report_tail_analysis()
    assert "3.5" in out
    assert "FLT" in out
