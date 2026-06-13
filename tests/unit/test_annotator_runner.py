import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from gpcr_tools.annotator import runner
from gpcr_tools.config import (
    get_config,
    get_gemini_model_name,
    model_run_subdir,
    reset_config,
)
from gpcr_tools.detector.signals import SEVERITY_ADVISORY, SIGNAL_INCIDENTAL_CANDIDATE, DetectSignal


def test_run_single_pdb_skips_if_done(tmp_path, monkeypatch):
    """Test that runner fast-exits if all run files exist."""

    # Mock config via env vars
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()

    pdb_id = "7W55"
    out_dir = config.ai_results_dir / pdb_id / model_run_subdir(get_gemini_model_name())
    out_dir.mkdir(parents=True)

    # Create 2 runs
    for n in range(1, 3):
        (out_dir / f"run_{n}.json").write_text("{}")

    # Attempting to run 2 times should be skipped
    # If it wasn't skipped, it would crash calling the unset Client.
    runner.run_single_pdb(pdb_id, {}, "Prompt", Path("dummy.pdf"), num_runs=2)

    # Output runs still the exact 2
    assert len(list(out_dir.glob("run_*.json"))) == 2


def test_build_and_submit_batch(tmp_path, monkeypatch):
    """Test building a JSONL and submitting it to the Gemini batch API."""
    monkeypatch.setenv("GPCR_ENRICHED_PATH", str(tmp_path / "enriched"))
    monkeypatch.setenv("GPCR_PAPERS_PATH", str(tmp_path / "papers"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    reset_config()
    config = get_config()

    (tmp_path / "state").mkdir()
    config.enriched_dir.mkdir()
    config.papers_dir.mkdir()

    # Setup dummy target files
    pdb_id = "7W55"
    (config.enriched_dir / f"{pdb_id}.json").write_text("{}")
    (config.papers_dir / f"{pdb_id}.pdf").write_text("%PDF")

    # Mock get_client
    mock_client = MagicMock()
    mock_files = MagicMock()
    mock_uploaded = MagicMock()
    mock_uploaded.uri = "http://mock.uri"
    mock_uploaded.name = "mock/uploaded_file"
    mock_files.upload.return_value = mock_uploaded
    mock_client.files = mock_files

    mock_batches = MagicMock()
    mock_batch_job = MagicMock()
    mock_batch_job.name = "batchJobs/mock_job_name"
    mock_batches.create.return_value = mock_batch_job
    mock_client.batches = mock_batches

    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: mock_client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.compress_pdf_if_needed", lambda a, b: a)

    runner.build_and_submit_batch([pdb_id], "Prompt", num_runs=1)

    # Verification
    assert mock_files.upload.call_count == 2  # Once for PDF, once for JSONL
    assert mock_batches.create.call_count == 1

    # Check registry updated (now stores the URI with an upload timestamp so a
    # stale cached upload can be detected and re-uploaded).
    registry = json.loads(config.uploaded_files_registry_file.read_text())
    assert registry[pdb_id]["uri"] == "http://mock.uri"
    assert "uploaded_at" in registry[pdb_id]

    # Check job updated
    assert config.current_batch_job_file.read_text() == "batchJobs/mock_job_name"


def test_recover_batch(tmp_path, monkeypatch):
    """Test recovering JSONL responses into run_n.json files."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.pipeline_runs_dir.mkdir(parents=True)

    raw_output = config.pipeline_runs_dir / "raw_output_testjob.jsonl"

    # Construct a mock batch response lines
    mock_resp = {
        "key": "7W55__run_01",
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "annotate_gpcr_db_structure",
                                    "args": {
                                        # To post processor
                                        "receptor_info": {"uniprot_entry_name": "OPSD_BOVIN"}
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        },
    }
    raw_output.write_text(json.dumps(mock_resp) + "\n")

    # Mock post_processor so it doesn't try to sanitize empty structure
    def mock_post_process(args):
        return {"sanitized": True, "receptor_info": args.get("receptor_info")}

    monkeypatch.setattr("gpcr_tools.annotator.runner.post_process_annotation", mock_post_process)

    runner.recover_batch()

    out_file = config.ai_results_dir / "7W55" / model_run_subdir(None) / "run_1.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert data["sanitized"] is True
    assert data["receptor_info"]["uniprot_entry_name"] == "OPSD_BOVIN"


def test_run_single_pdb_writes_provenance(tmp_path, monkeypatch):
    """Each single-mode result records which model/prompt/run produced it."""
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    monkeypatch.setenv("GPCR_CODE_VERSION", "abc1234")
    reset_config()
    config = get_config()

    mock_client = MagicMock()
    mock_client.files.upload.return_value = MagicMock(uri="u", name="f")
    fc = MagicMock()
    fc.name = runner.ANNOTATOR_FUNCTION_NAME
    fc.args = {"receptor_info": {"uniprot_entry_name": "OPSD_BOVIN"}}
    mock_response = MagicMock()
    mock_response.function_calls = [fc]
    mock_response.model_version = "gemini-2.5-pro-002"
    mock_client.models.generate_content.return_value = mock_response

    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: mock_client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.compress_pdf_if_needed", lambda a, b: a)
    monkeypatch.setattr("gpcr_tools.annotator.runner.build_prompt_parts", lambda *a, **k: ["ctx"])
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.post_process_annotation",
        lambda args: {"receptor_info": args.get("receptor_info")},
    )
    # An advisory signal must be recorded in provenance.
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.load_detect_signals",
        lambda pdb_id: [
            DetectSignal(
                kind=SIGNAL_INCIDENTAL_CANDIDATE,
                target_ref="ligands",
                summary="PLM incidental_candidate",
                payload={"comp_id": "PLM"},
                severity=SEVERITY_ADVISORY,
            )
        ],
    )

    runner.run_single_pdb(
        "7W55",
        {},
        "Prompt",
        Path("dummy.pdf"),
        num_runs=1,
        model_name="gemini-2.5-pro",
        prompt_id="v5",
    )

    # Large PDFs upload via the resumable path, which needs an explicit mime type.
    assert mock_client.files.upload.call_args.kwargs["config"]["mime_type"] == "application/pdf"

    data = json.loads(
        (
            config.ai_results_dir / "7W55" / model_run_subdir("gemini-2.5-pro") / "run_1.json"
        ).read_text()
    )
    prov = data["_provenance"]
    assert prov["model_requested"] == "gemini-2.5-pro"
    assert prov["model_served"] == "gemini-2.5-pro-002"
    assert prov["prompt"] == "v5"
    assert prov["code_version"] == "abc1234"
    assert prov["detect_advisory"] == [SIGNAL_INCIDENTAL_CANDIDATE]
    assert prov["run"] == 1
    assert prov["mode"] == "single"


def test_recover_batch_writes_provenance(tmp_path, monkeypatch):
    """Recovered batch results record model/prompt from the submission sidecar."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.pipeline_runs_dir.mkdir(parents=True)

    (config.pipeline_runs_dir / "_batch_provenance.json").write_text(
        json.dumps({"model_requested": "gemini-2.5-pro", "prompt": "v5", "code_version": "abc1234"})
    )

    raw_output = config.pipeline_runs_dir / "raw_output_testjob.jsonl"
    mock_resp = {
        "key": "7W55__run_01",
        "response": {
            "modelVersion": "gemini-2.5-pro-002",
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "annotate_gpcr_db_structure",
                                    "args": {"receptor_info": {}},
                                }
                            }
                        ]
                    }
                }
            ],
        },
    }
    raw_output.write_text(json.dumps(mock_resp) + "\n")
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.post_process_annotation", lambda args: {"ok": True}
    )

    runner.recover_batch()

    data = json.loads(
        (
            config.ai_results_dir / "7W55" / model_run_subdir("gemini-2.5-pro") / "run_1.json"
        ).read_text()
    )
    prov = data["_provenance"]
    assert prov["mode"] == "batch"
    assert prov["model_requested"] == "gemini-2.5-pro"
    assert prov["prompt"] == "v5"
    assert prov["code_version"] == "abc1234"
    assert prov["model_served"] == "gemini-2.5-pro-002"
    assert prov["run"] == 1


def test_recover_batch_uses_per_job_provenance(tmp_path, monkeypatch):
    """Two completed batches with different models must not cross-contaminate:
    each raw output is stamped from its OWN per-job provenance sidecar, never a
    single shared file that the latest submission overwrote."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.pipeline_runs_dir.mkdir(parents=True)

    def _resp(key, served):
        return {
            "key": key,
            "response": {
                "modelVersion": served,
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "annotate_gpcr_db_structure",
                                        "args": {"receptor_info": {}},
                                    }
                                }
                            ]
                        }
                    }
                ],
            },
        }

    # Job A (model-pro) and job B (model-flash): each with its own raw output
    # and its own provenance sidecar.
    (config.pipeline_runs_dir / "raw_output_jobA.jsonl").write_text(
        json.dumps(_resp("1ABC__run_01", "pro-002")) + "\n"
    )
    (config.pipeline_runs_dir / "_batch_provenance_jobA.json").write_text(
        json.dumps({"model_requested": "model-pro", "prompt": "v5"})
    )
    (config.pipeline_runs_dir / "raw_output_jobB.jsonl").write_text(
        json.dumps(_resp("2XYZ__run_01", "flash-002")) + "\n"
    )
    (config.pipeline_runs_dir / "_batch_provenance_jobB.json").write_text(
        json.dumps({"model_requested": "model-flash", "prompt": "v5"})
    )
    # A stale shared file from the most recent submission must NOT win over the
    # per-job sidecars — that overwrite is exactly what cross-contaminated runs.
    (config.pipeline_runs_dir / "_batch_provenance.json").write_text(
        json.dumps({"model_requested": "model-flash", "prompt": "v5"})
    )

    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.post_process_annotation", lambda args: {"ok": True}
    )

    runner.recover_batch()

    pro = config.ai_results_dir / "1ABC" / model_run_subdir("model-pro") / "run_1.json"
    flash = config.ai_results_dir / "2XYZ" / model_run_subdir("model-flash") / "run_1.json"
    assert pro.exists(), "job A must be stamped model-pro, not the shared file's model-flash"
    assert flash.exists()
    assert json.loads(pro.read_text())["_provenance"]["model_requested"] == "model-pro"
    assert json.loads(flash.read_text())["_provenance"]["model_requested"] == "model-flash"


def _setup_batch_state(tmp_path, monkeypatch, job_name="batchJobs/mock_job"):
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    (tmp_path / "state").mkdir(exist_ok=True)
    config.current_batch_job_file.write_text(job_name)
    # Start each setup from a clean registry so the legacy-adoption path (the
    # only thing the single-file pointer drives now) is exercised consistently,
    # and a loop reusing one tmp_path doesn't carry a job's status across rounds.
    config.batch_jobs_registry_file.unlink(missing_ok=True)
    return config


def _mock_batch_job(state_name, file_name="files/result.jsonl", error=None):
    job = MagicMock()
    job.state.name = state_name
    if file_name is None:
        job.dest = None
    else:
        job.dest.file_name = file_name
    job.error = error
    return job


def test_check_batch_status_downloads_on_success(tmp_path, monkeypatch):
    """A succeeded job downloads its result file via the SDK and recovers it."""
    config = _setup_batch_state(tmp_path, monkeypatch)
    client = MagicMock()
    client.batches.get.return_value = _mock_batch_job("JOB_STATE_SUCCEEDED")
    client.files.download.return_value = b'{"id": "x"}\n'
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.recover_batch", lambda: None)

    runner.check_batch_status()

    client.files.download.assert_called_once_with(file="files/result.jsonl")
    raw = config.pipeline_runs_dir / "raw_output_batchJobs_mock_job.jsonl"
    assert raw.read_bytes() == b'{"id": "x"}\n'


def test_check_batch_status_partial_success_also_downloads(tmp_path, monkeypatch):
    _setup_batch_state(tmp_path, monkeypatch)
    client = MagicMock()
    client.batches.get.return_value = _mock_batch_job("JOB_STATE_PARTIALLY_SUCCEEDED")
    client.files.download.return_value = b"{}\n"
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.recover_batch", lambda: None)

    runner.check_batch_status()

    client.files.download.assert_called_once()


def test_check_batch_status_no_download_on_terminal_failure(tmp_path, monkeypatch):
    """Failed / cancelled / expired jobs carry no results -- do not attempt download."""
    for state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        _setup_batch_state(tmp_path, monkeypatch)
        client = MagicMock()
        client.batches.get.return_value = _mock_batch_job(state, error="boom")
        monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda c=client: c)

        runner.check_batch_status()

        client.files.download.assert_not_called()


def test_check_batch_status_no_download_while_pending(tmp_path, monkeypatch):
    _setup_batch_state(tmp_path, monkeypatch)
    client = MagicMock()
    client.batches.get.return_value = _mock_batch_job("JOB_STATE_PENDING")
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)

    runner.check_batch_status()

    client.files.download.assert_not_called()


def test_run_outputs_namespaced_by_model(tmp_path, monkeypatch):
    """Annotating one PDB with two models writes to separate subdirectories
    instead of overwriting."""
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    pdb_id = "7W55"

    # Pre-seed model-a's run so any collision would be visible.
    a_dir = config.ai_results_dir / pdb_id / model_run_subdir("model-a")
    a_dir.mkdir(parents=True)
    (a_dir / "run_1.json").write_text('{"from": "a"}')

    mock_client = MagicMock()
    fc = MagicMock()
    fc.name = runner.ANNOTATOR_FUNCTION_NAME
    fc.args = {"receptor_info": {}}
    mock_response = MagicMock()
    mock_response.function_calls = [fc]
    mock_response.model_version = "model-b-001"
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: mock_client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.compress_pdf_if_needed", lambda a, b: a)
    monkeypatch.setattr("gpcr_tools.annotator.runner.build_prompt_parts", lambda *a, **k: ["ctx"])
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.post_process_annotation", lambda args: {"ok": True}
    )

    runner.run_single_pdb(pdb_id, {}, "Prompt", Path("dummy.pdf"), num_runs=1, model_name="model-b")

    b_file = config.ai_results_dir / pdb_id / model_run_subdir("model-b") / "run_1.json"
    assert b_file.exists()
    # model-a's run is untouched.
    assert json.loads((a_dir / "run_1.json").read_text()) == {"from": "a"}


def test_discover_annotation_targets_is_model_aware(tmp_path, monkeypatch):
    """A PDB complete for one model is excluded for that model but still pending
    for another (runs are namespaced per model)."""
    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    config = get_config()
    config.enriched_dir.mkdir(parents=True)
    (config.enriched_dir / "AAA.json").write_text("{}")
    (config.enriched_dir / "BBB.json").write_text("{}")
    # AAA has both runs under model-x; BBB has none.
    done_dir = config.ai_results_dir / "AAA" / model_run_subdir("model-x")
    done_dir.mkdir(parents=True)
    (done_dir / "run_1.json").write_text("{}")
    (done_dir / "run_2.json").write_text("{}")

    assert runner.discover_annotation_targets(2, "model-x") == ["BBB"]
    assert "AAA" in runner.discover_annotation_targets(2, "model-y")


def test_run_annotation_stage_fails_fast_on_stale_contract(tmp_path, monkeypatch):
    """The expensive AI annotate stage must fail fast on a stale storage
    contract -- before resolving targets or making any model call.
    """
    import pytest

    monkeypatch.setenv("GPCR_WORKSPACE", str(tmp_path))
    reset_config()
    config = get_config()
    config.contract_file.parent.mkdir(parents=True, exist_ok=True)
    # Unsupported (stale) version recorded in the workspace.
    config.contract_file.write_text(json.dumps({"storage_contract_version": 1}))

    # Targets / model discovery must NOT be reached if the gate fires first.
    def _boom(*args, **kwargs):
        raise AssertionError("expensive work ran despite a stale contract")

    monkeypatch.setattr(runner, "discover_annotation_targets", _boom)

    with pytest.raises(SystemExit):
        runner.run_annotation_stage()


def test_registry_fresh_uri_expiry():
    """A cached upload URI is reused only within the Files-API TTL; a stale or
    legacy entry returns None so the caller re-uploads."""
    now = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    fresh = {"uri": "files/abc", "uploaded_at": (now - timedelta(hours=1)).isoformat()}
    stale = {"uri": "files/old", "uploaded_at": (now - timedelta(hours=50)).isoformat()}

    assert runner._registry_fresh_uri(fresh, now) == "files/abc"
    assert runner._registry_fresh_uri(stale, now) is None
    assert runner._registry_fresh_uri("files/legacy-bare-string", now) is None
    assert runner._registry_fresh_uri(None, now) is None
    assert runner._registry_fresh_uri({"uri": "files/x"}, now) is None  # no timestamp


def test_chunk_request_groups_packs_and_never_splits_a_pdb():
    """Groups that can't combine under the cap each become their own chunk,
    intact -- a single PDB's runs are never split across jobs."""
    groups = [[1, 2, 3], [4, 5, 6]]  # two PDBs, 3 runs each
    assert runner._chunk_request_groups(groups, 4) == [[1, 2, 3], [4, 5, 6]]


def test_chunk_request_groups_combines_small_groups():
    groups = [[1, 2], [3], [4, 5, 6]]
    assert runner._chunk_request_groups(groups, 3) == [[1, 2, 3], [4, 5, 6]]


def test_chunk_request_groups_oversized_group_is_its_own_chunk():
    """A single PDB whose runs exceed the cap is kept whole in its own chunk
    rather than being divided."""
    groups = [[1, 2, 3, 4, 5], [6]]
    assert runner._chunk_request_groups(groups, 3) == [[1, 2, 3, 4, 5], [6]]


def test_chunk_request_groups_empty():
    assert runner._chunk_request_groups([], 3) == []


def _setup_multi_pdb_batch(tmp_path, monkeypatch, pdbs):
    monkeypatch.setenv("GPCR_ENRICHED_PATH", str(tmp_path / "enriched"))
    monkeypatch.setenv("GPCR_PAPERS_PATH", str(tmp_path / "papers"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    reset_config()
    config = get_config()
    (tmp_path / "state").mkdir()
    config.enriched_dir.mkdir()
    config.papers_dir.mkdir()
    for p in pdbs:
        (config.enriched_dir / f"{p}.json").write_text("{}")
        (config.papers_dir / f"{p}.pdf").write_text("%PDF")

    upl = MagicMock()
    upl.uri = "u"
    upl.name = "files/src"
    client = MagicMock()
    client.files.upload.return_value = upl
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.compress_pdf_if_needed", lambda a, b: a)
    return config, client


def test_build_and_submit_batch_shards_into_multiple_jobs(tmp_path, monkeypatch):
    """A submission larger than the per-job cap is split into multiple jobs,
    each registered in the job registry; the single-file pointer alone could
    not have tracked them."""
    config, client = _setup_multi_pdb_batch(tmp_path, monkeypatch, ["1ABC", "2DEF", "3GHI"])
    # cap 2; 3 PDBs x 1 run = 3 requests, never splitting a PDB -> 2 jobs.
    monkeypatch.setattr("gpcr_tools.annotator.runner.GEMINI_BATCH_MAX_REQUESTS", 2)
    job0, job1 = MagicMock(), MagicMock()
    job0.name = "batchJobs/j0"
    job1.name = "batchJobs/j1"
    client.batches.create.side_effect = [job0, job1]

    runner.build_and_submit_batch(["1ABC", "2DEF", "3GHI"], "Prompt", num_runs=1)

    assert client.batches.create.call_count == 2
    registry = json.loads(config.batch_jobs_registry_file.read_text())
    assert set(registry["jobs"]) == {"batchJobs/j0", "batchJobs/j1"}
    assert registry["jobs"]["batchJobs/j0"]["chunk_count"] == 2
    assert registry["jobs"]["batchJobs/j0"]["status"] == "submitted"
    # The deprecated single-file pointer mirrors the most recent job.
    assert config.current_batch_job_file.read_text() == "batchJobs/j1"


def test_build_and_submit_batch_chunk_failure_isolated(tmp_path, monkeypatch):
    """If one chunk fails to submit, earlier chunks stay registered and the
    call does not crash."""
    config, client = _setup_multi_pdb_batch(tmp_path, monkeypatch, ["1ABC", "2DEF", "3GHI"])
    monkeypatch.setattr("gpcr_tools.annotator.runner.GEMINI_BATCH_MAX_REQUESTS", 2)
    job0 = MagicMock()
    job0.name = "batchJobs/j0"
    client.batches.create.side_effect = [job0, RuntimeError("boom")]

    runner.build_and_submit_batch(["1ABC", "2DEF", "3GHI"], "Prompt", num_runs=1)  # no raise

    assert client.batches.create.call_count == 2
    registry = json.loads(config.batch_jobs_registry_file.read_text())
    assert set(registry["jobs"]) == {"batchJobs/j0"}


def test_check_batch_status_polls_all_pending_jobs(tmp_path, monkeypatch):
    """All submitted jobs in the registry are polled and downloaded, not just
    one (the single-file pointer could track only one)."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.batch_jobs_registry_file.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "batchJobs/a": {"job_name": "batchJobs/a", "status": "submitted"},
                    "batchJobs/b": {"job_name": "batchJobs/b", "status": "submitted"},
                },
            }
        )
    )
    client = MagicMock()
    client.batches.get.side_effect = lambda name: _mock_batch_job("JOB_STATE_SUCCEEDED")
    client.files.download.return_value = b"{}\n"
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.recover_batch", lambda: None)

    runner.check_batch_status()

    assert client.files.download.call_count == 2
    registry = json.loads(config.batch_jobs_registry_file.read_text())
    assert registry["jobs"]["batchJobs/a"]["status"] == "downloaded"
    assert registry["jobs"]["batchJobs/b"]["status"] == "downloaded"


def test_check_batch_status_adopts_legacy_job(tmp_path, monkeypatch):
    """A pre-registry workspace with only current_batch_job.txt has its
    in-flight job adopted into the registry and polled."""
    config = _setup_batch_state(tmp_path, monkeypatch, job_name="batchJobs/legacy")
    client = MagicMock()
    client.batches.get.return_value = _mock_batch_job("JOB_STATE_SUCCEEDED")
    client.files.download.return_value = b"{}\n"
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)
    monkeypatch.setattr("gpcr_tools.annotator.runner.recover_batch", lambda: None)

    runner.check_batch_status()

    registry = json.loads(config.batch_jobs_registry_file.read_text())
    assert "batchJobs/legacy" in registry["jobs"]
    client.batches.get.assert_called_once_with(name="batchJobs/legacy")
    client.files.download.assert_called_once()


def test_recover_batch_skips_existing_run(tmp_path, monkeypatch):
    """recover never clobbers an already-recovered run (idempotent; immune to a
    stale re-downloaded raw file)."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.pipeline_runs_dir.mkdir(parents=True)

    out_dir = config.ai_results_dir / "7W55" / model_run_subdir(None)
    out_dir.mkdir(parents=True)
    (out_dir / "run_1.json").write_text('{"keep": true}')

    raw_output = config.pipeline_runs_dir / "raw_output_testjob.jsonl"
    raw_output.write_text(
        json.dumps(
            {
                "key": "7W55__run_01",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "annotate_gpcr_db_structure",
                                            "args": {"receptor_info": {}},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        )
        + "\n"
    )

    called = {"n": 0}

    def _pp(args):
        called["n"] += 1
        return {"overwritten": True}

    monkeypatch.setattr("gpcr_tools.annotator.runner.post_process_annotation", _pp)

    runner.recover_batch()

    # The existing run is untouched and post-processing was never invoked.
    assert json.loads((out_dir / "run_1.json").read_text()) == {"keep": True}
    assert called["n"] == 0


def test_recover_batch_uses_registry_attribution(tmp_path, monkeypatch):
    """When a raw output has a registry entry, its model/prompt come from the
    entry -- not a stale shared sidecar that names a different model."""
    monkeypatch.setenv("GPCR_STATE_PATH", str(tmp_path / "state"))
    monkeypatch.setenv("GPCR_AI_RESULTS_PATH", str(tmp_path / "ai_results"))
    reset_config()
    config = get_config()
    config.pipeline_runs_dir.mkdir(parents=True)

    raw_path = config.pipeline_runs_dir / "raw_output_batchJobs_jx.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "key": "1ABC__run_01",
                "response": {
                    "modelVersion": "model-pro-002",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "annotate_gpcr_db_structure",
                                            "args": {"receptor_info": {}},
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                },
            }
        )
        + "\n"
    )
    config.batch_jobs_registry_file.write_text(
        json.dumps(
            {
                "version": 1,
                "jobs": {
                    "batchJobs/jx": {
                        "job_name": "batchJobs/jx",
                        "status": "downloaded",
                        "model_requested": "model-pro",
                        "prompt": "v5",
                        "code_version": "zzz",
                        "detect_advisory": {},
                        "raw_output_file": str(raw_path),
                    }
                },
            }
        )
    )
    # A stale shared sidecar names a different model; the registry must win.
    (config.pipeline_runs_dir / "_batch_provenance.json").write_text(
        json.dumps({"model_requested": "model-flash", "prompt": "old"})
    )
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.post_process_annotation", lambda args: {"ok": True}
    )

    runner.recover_batch()

    pro = config.ai_results_dir / "1ABC" / model_run_subdir("model-pro") / "run_1.json"
    flash = config.ai_results_dir / "1ABC" / model_run_subdir("model-flash") / "run_1.json"
    assert pro.exists()
    assert not flash.exists()
    prov = json.loads(pro.read_text())["_provenance"]
    assert prov["model_requested"] == "model-pro"
    assert prov["prompt"] == "v5"
    assert prov["code_version"] == "zzz"
    registry = json.loads(config.batch_jobs_registry_file.read_text())
    assert registry["jobs"]["batchJobs/jx"]["status"] == "recovered"


def test_check_batch_status_legacy_recover_uses_sidecar_model(tmp_path, monkeypatch):
    """End-to-end migration: an in-flight job adopted from current_batch_job.txt
    recovers under the model recorded in its legacy sidecar -- not the adopted
    entry's null model, which would mis-file the result to default/."""
    config = _setup_batch_state(tmp_path, monkeypatch, job_name="batchJobs/old1")
    config.pipeline_runs_dir.mkdir(parents=True, exist_ok=True)
    # Legacy per-job sidecar holds the real model (written by the old code path).
    (config.pipeline_runs_dir / "_batch_provenance_batchJobs_old1.json").write_text(
        json.dumps({"model_requested": "gemini-2.5-pro", "prompt": "v5", "code_version": "abc1234"})
    )

    raw_line = (
        json.dumps(
            {
                "key": "7W55__run_01",
                "response": {
                    "modelVersion": "gemini-2.5-pro-002",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "annotate_gpcr_db_structure",
                                            "args": {"receptor_info": {}},
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                },
            }
        ).encode()
        + b"\n"
    )

    client = MagicMock()
    client.batches.get.return_value = _mock_batch_job("JOB_STATE_SUCCEEDED")
    client.files.download.return_value = raw_line
    monkeypatch.setattr("gpcr_tools.annotator.runner.get_client", lambda: client)
    monkeypatch.setattr(
        "gpcr_tools.annotator.runner.post_process_annotation", lambda args: {"ok": True}
    )

    runner.check_batch_status()  # adopt -> download -> recover, all real

    correct = config.ai_results_dir / "7W55" / model_run_subdir("gemini-2.5-pro") / "run_1.json"
    wrong = config.ai_results_dir / "7W55" / model_run_subdir(None) / "run_1.json"
    assert correct.exists(), "recover must use the sidecar model, not default/"
    assert not wrong.exists()
    assert json.loads(correct.read_text())["_provenance"]["model_requested"] == "gemini-2.5-pro"


def _capture_batch_jsonl(client):
    """Patch the client's file upload to capture the batch JSONL content."""
    captured = {}

    def _upload(file=None, config=None):
        if str(file).endswith(".jsonl"):
            captured["jsonl"] = Path(file).read_text()
        m = MagicMock()
        m.name = "files/src"
        m.uri = "u"
        return m

    client.files.upload.side_effect = _upload
    return captured


def test_build_and_submit_batch_injects_temperature(tmp_path, monkeypatch):
    """A given temperature is emitted as per-request generationConfig in the JSONL."""
    _, client = _setup_multi_pdb_batch(tmp_path, monkeypatch, ["7W55"])
    job = MagicMock()
    job.name = "batchJobs/j0"
    client.batches.create.return_value = job
    captured = _capture_batch_jsonl(client)

    runner.build_and_submit_batch(["7W55"], "Prompt", num_runs=1, temperature=0.7)

    lines = [json.loads(line) for line in captured["jsonl"].splitlines() if line.strip()]
    assert lines[0]["request"]["generationConfig"] == {"temperature": 0.7}


def test_build_and_submit_batch_omits_temperature_by_default(tmp_path, monkeypatch):
    """Without a temperature, no generationConfig is sent -- behaviour unchanged."""
    _, client = _setup_multi_pdb_batch(tmp_path, monkeypatch, ["7W55"])
    job = MagicMock()
    job.name = "batchJobs/j0"
    client.batches.create.return_value = job
    captured = _capture_batch_jsonl(client)

    runner.build_and_submit_batch(["7W55"], "Prompt", num_runs=1)

    lines = [json.loads(line) for line in captured["jsonl"].splitlines() if line.strip()]
    assert "generationConfig" not in lines[0]["request"]


def test_build_tool_config_temperature_override():
    """The inline generation config leaves temperature unset by default (so the
    model default applies) and honours an explicit override."""
    from gpcr_tools.annotator.detect_orchestrator import build_tool_config

    assert build_tool_config([]).temperature is None
    assert build_tool_config([], temperature=0.7).temperature == 0.7
