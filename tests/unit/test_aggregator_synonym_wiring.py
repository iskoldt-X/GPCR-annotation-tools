"""Wiring guard: the PubChem synonym check must stay connected to aggregation.

The synonym check is opt-in -- ``validate_and_enrich_ligands`` only runs it when
given a ``synonym_cache``. These tests assert ``aggregate_pdb`` threads the cache
through when API checks are on, and suppresses it (passes ``None``) under
``--skip-api-checks``. Without them, a future edit that drops the ``synonym_cache=``
keyword or flips the gate would silently return the check to dormant and every
other test would still pass.
"""

from __future__ import annotations

import gpcr_tools.aggregator.runner as runner


class _StopAfterLigandStepError(Exception):
    """Raised by the spy to short-circuit the steps after ligand validation."""


def _stub_to_ligand_step(monkeypatch, recorder):
    """Stub everything up to and including step 5 so ``aggregate_pdb`` reaches the
    ligand-validation step, and replace ``validate_and_enrich_ligands`` with a spy
    that records the ``synonym_cache`` it received, then short-circuits the rest of
    the pipeline (the spy raises; ``aggregate_pdb`` catches it) so the test makes no
    network or disk calls.
    """
    monkeypatch.setattr(runner, "get_config", lambda: object())
    monkeypatch.setattr("gpcr_tools.workspace.validate_contract", lambda cfg: None)
    monkeypatch.setattr(runner, "load_ai_runs", lambda pdb_id: [{"ligands": []}])
    monkeypatch.setattr(runner, "get_majority_votes", lambda runs: ({}, {}))
    monkeypatch.setattr(runner, "select_best_run", lambda runs, votes: (0, {"ligands": []}))
    monkeypatch.setattr(runner, "load_enriched_data", lambda pdb_id: {})
    monkeypatch.setattr(runner, "inject_ground_truth", lambda *a, **k: None)

    def _spy(pdb_id, best_run_data, enriched, *, synonym_cache=None):
        recorder["called"] = True
        recorder["synonym_cache"] = synonym_cache
        raise _StopAfterLigandStepError

    monkeypatch.setattr(runner, "validate_and_enrich_ligands", _spy)


def test_synonym_cache_threaded_when_api_checks_on(monkeypatch):
    recorder: dict = {}
    _stub_to_ligand_step(monkeypatch, recorder)
    sentinel_cache = object()

    runner.aggregate_pdb("XXXX", skip_api_checks=False, synonym_cache=sentinel_cache)

    assert recorder.get("called") is True
    assert recorder["synonym_cache"] is sentinel_cache


def test_synonym_cache_suppressed_under_skip_api_checks(monkeypatch):
    recorder: dict = {}
    _stub_to_ligand_step(monkeypatch, recorder)
    sentinel_cache = object()

    runner.aggregate_pdb("XXXX", skip_api_checks=True, synonym_cache=sentinel_cache)

    assert recorder.get("called") is True
    assert recorder["synonym_cache"] is None
