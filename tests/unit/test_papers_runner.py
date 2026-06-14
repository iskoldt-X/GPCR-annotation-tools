"""Tests for papers/runner.py — the fetch-papers orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from gpcr_tools.papers.runner import run_fetch_papers


class TestWatchOnly:
    def test_watch_only_skips_download_and_watches_paywalled(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """--watch-only goes straight to the watcher on the log's paywalled entries,
        WITHOUT re-running the auto-download retry (no network, no email needed)."""
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        # Email intentionally NOT set -- watch-only must not require it.
        monkeypatch.delenv("GPCR_EMAIL_FOR_APIS", raising=False)
        from gpcr_tools.config import get_config, reset_config

        reset_config()
        cfg = get_config()
        cfg.download_log_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.download_log_file.write_text(
            json.dumps(
                {
                    "7W55": {"status": "success_pdf_downloaded"},
                    "8ABC": {"status": "fallback_paywalled", "doi": "10.x/y"},
                    "9XYZ": {"status": "fallback_paywalled", "doi": None},
                }
            )
        )

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "gpcr_tools.papers.runner.run_watcher",
            lambda log: seen.setdefault("log", log) or 0,
        )

        def _fail_download(*_a, **_k):
            raise AssertionError("download must not run in watch-only mode")

        monkeypatch.setattr("gpcr_tools.papers.runner.download_paper_for_pdb", _fail_download)

        run_fetch_papers(watch_only=True)

        # The watcher receives the FULL log (it groups by DOI across all PDBs); it is
        # entered only because there ARE paywalled entries to resolve.
        assert set(seen["log"]) == {"7W55", "8ABC", "9XYZ"}

    def test_watch_only_no_paywalled_returns_cleanly(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
        from gpcr_tools.config import get_config, reset_config

        reset_config()
        cfg = get_config()
        cfg.download_log_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.download_log_file.write_text(json.dumps({"7W55": {"status": "success_pdf_downloaded"}}))

        called = {"watch": False}
        monkeypatch.setattr(
            "gpcr_tools.papers.runner.run_watcher",
            lambda paywalled: called.__setitem__("watch", True),
        )
        run_fetch_papers(watch_only=True)
        assert called["watch"] is False  # nothing to watch -> watcher not entered
