"""Tests for end-to-end pipeline orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from gpcr_tools import pipeline
from gpcr_tools.config import get_config, reset_config


@pytest.fixture()
def cfg(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    c = get_config()
    c.enriched_dir.mkdir(parents=True, exist_ok=True)
    return c


def _patch_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr("gpcr_tools.fetcher.runner.run_fetch", lambda **k: calls.append("fetch"))
    monkeypatch.setattr(
        "gpcr_tools.papers.runner.run_fetch_papers", lambda **k: calls.append("fetch-papers")
    )
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.run_annotation_stage", lambda **k: calls.append("annotate")
    )
    monkeypatch.setattr(
        "gpcr_tools.aggregator.runner.aggregate_all",
        lambda **k: (calls.append("aggregate"), [])[1],
    )


def test_dry_run_executes_nothing(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(dry_run=True)
    assert calls == []


def test_runs_stages_in_dependency_order(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (cfg.enriched_dir / "7W55.json").write_text("{}")  # fetch prerequisite
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline()
    assert calls == ["fetch", "fetch-papers", "annotate", "aggregate"]


def test_batch_mode_stops_after_submit(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (cfg.enriched_dir / "7W55.json").write_text("{}")
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(batch=True)
    # Batch results arrive asynchronously -> do not aggregate in the same run.
    assert calls == ["fetch", "fetch-papers", "annotate"]


def test_skip_fetch_papers(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (cfg.enriched_dir / "7W55.json").write_text("{}")
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline(skip_fetch_papers=True)
    assert calls == ["fetch", "annotate", "aggregate"]


def test_stops_when_no_enriched_data(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # enriched dir is empty -> nothing to annotate; stop after fetch.
    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    pipeline.run_pipeline()
    assert calls == ["fetch"]


def test_dry_run_logs_the_plan(
    cfg: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    calls: list[str] = []
    _patch_stages(monkeypatch, calls)
    with caplog.at_level(logging.INFO, logger="gpcr_tools.pipeline"):
        pipeline.run_pipeline(dry_run=True)
    assert "would run" in caplog.text  # the preview is emitted (not swallowed)
    assert calls == []
