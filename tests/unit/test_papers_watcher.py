"""Tests for papers/watcher.py — filesystem watcher for paywalled papers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpcr_tools.papers.watcher import (
    _get_pending_paywalled,
    _ingest_if_ready,
    _is_valid_pdf,
    _match_pdf_to_pdb,
    _wait_for_stability,
    run_watcher,
)


class TestIsValidPdf:
    def test_valid_pdf(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 some content")
        assert _is_valid_pdf(f) is True

    def test_invalid(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"not a pdf")
        assert _is_valid_pdf(f) is False

    def test_missing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.pdf"
        assert _is_valid_pdf(f) is False


class TestGetPendingPaywalled:
    def test_filters_paywalled(self) -> None:
        log: dict[str, Any] = {
            "7W55": {"status": "success_pdf_downloaded"},
            "8ABC": {"status": "fallback_paywalled", "doi": "10.1038/test"},
            "9XYZ": {"status": "fallback_paywalled", "doi": None},
        }
        pending = _get_pending_paywalled(log)
        assert "7W55" not in pending
        assert "8ABC" in pending
        assert "9XYZ" in pending

    def test_empty_log(self) -> None:
        assert _get_pending_paywalled({}) == {}


class TestMatchPdfToPdb:
    def test_exact_filename_match(self, tmp_path: Path) -> None:
        pdf = tmp_path / "8ABC.pdf"
        pending: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled"},
            "9XYZ": {"status": "fallback_paywalled"},
        }
        assert _match_pdf_to_pdb(pdf, pending) == "8ABC"

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        pdf = tmp_path / "8abc.pdf"
        pending: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled"},
        }
        assert _match_pdf_to_pdb(pdf, pending) == "8ABC"

    def test_no_match(self, tmp_path: Path) -> None:
        pdf = tmp_path / "random_paper.pdf"
        pending: dict[str, Any] = {
            "8ABC": {"status": "fallback_paywalled"},
        }
        assert _match_pdf_to_pdb(pdf, pending) is None


class TestWaitForStability:
    def test_stable_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4" + b"x" * 1000)
        # File is already stable (not being written)
        assert _wait_for_stability(f) is True

    def test_missing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.pdf"
        assert _wait_for_stability(f) is False


class TestWatcherDoesNotAbandonPapers:
    def test_run_watcher_ingests_preexisting_pdf(self, tmp_path: Path, monkeypatch) -> None:
        """A paper already sitting in papers/ before the watch starts must be
        ingested as manual_user_provided — not snapshotted away and abandoned as
        skipped_no_paper."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import get_config, reset_config

        reset_config()
        cfg = get_config()
        cfg.papers_dir.mkdir(parents=True, exist_ok=True)
        (cfg.papers_dir / "8ABC.pdf").write_bytes(b"%PDF-1.4" + b"x" * 500)

        # All pending resolved by the startup scan, so the watch loop is never
        # entered (no hang) and the paper is recorded as provided.
        matched = run_watcher({"8ABC": {"status": "fallback_paywalled", "doi": "10.x/y"}})

        assert matched == 1
        log = json.loads(cfg.download_log_file.read_text())
        assert log["8ABC"]["status"] == "manual_user_provided"

    def test_ingest_if_ready_rejects_invalid_pdf_without_consuming_it(self, tmp_path: Path) -> None:
        """A stable-but-invalid file is left in place (returns False) so a later
        good copy can still be picked up — not renamed or logged."""
        bad = tmp_path / "raw.pdf"
        bad.write_bytes(b"<html>not a pdf</html>" + b"x" * 500)
        assert _ingest_if_ready(bad, "8ABC", {"8ABC": {"doi": None}}, tmp_path) is False
        assert not (tmp_path / "8ABC.pdf").exists()
        assert bad.exists()
