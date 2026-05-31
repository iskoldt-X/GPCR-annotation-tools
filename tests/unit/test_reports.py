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
        _write_validation(cfg, "AAA", chimera_score=4, chimera_status="success")
        _write_validation(cfg, "BBB", chimera_score=2, chimera_status="success")  # low -> flagged
        _write_validation(cfg, "CCC", chimera_score=4, chimera_status="conflict")  # non-ok -> flag
        out = reports.report_tail_analysis()
        assert "3 PDB" in out
        assert "score 4" in out
        assert "score 2" in out
        assert "Flagged for review (non-success or score < 4): 2" in out
        assert "BBB" in out
        assert "CCC" in out
