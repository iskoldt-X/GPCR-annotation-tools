"""Annotation runner -- single-PDB, batch submission, and recovery."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from google.genai.errors import APIError

from gpcr_tools.annotator.detect_orchestrator import build_tool_config, build_tool_for_signals
from gpcr_tools.annotator.gemini_client import get_client
from gpcr_tools.annotator.pdf_compressor import compress_pdf_if_needed
from gpcr_tools.annotator.post_processor import post_process_annotation
from gpcr_tools.annotator.prompt_builder import build_prompt_parts
from gpcr_tools.annotator.schema import ANNOTATION_TOOL
from gpcr_tools.code_version import get_code_version
from gpcr_tools.config import (
    ANNOTATOR_FUNCTION_NAME,
    BATCH_REGISTRY_VERSION,
    BATCH_STATUS_DOWNLOADED,
    BATCH_STATUS_FAILED,
    BATCH_STATUS_RECOVERED,
    BATCH_STATUS_SUBMITTED,
    GEMINI_BASE_BACKOFF,
    GEMINI_BATCH_MAX_REQUESTS,
    GEMINI_DEFAULT_RUNS,
    GEMINI_FILE_TTL_HOURS,
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_WORKERS,
    SLEEP_GEMINI_429,
    get_config,
    get_gemini_model_name,
    model_run_subdir,
)
from gpcr_tools.detector.signals import SEVERITY_ADVISORY
from gpcr_tools.detector.stage import load_detect_signals

logger = logging.getLogger(__name__)


def _registry_fresh_uri(entry: Any, now: datetime) -> str | None:
    """Return a still-valid cached upload URI from a registry *entry*, or None.

    Entries are ``{"uri": ..., "uploaded_at": <iso>}``. A URI older than the
    Files-API TTL — or a legacy bare-string entry whose age is unknown — is
    treated as expired so the caller re-uploads instead of embedding a dead
    fileUri into the batch request.
    """
    if isinstance(entry, dict):
        uri = entry.get("uri")
        uploaded_at = entry.get("uploaded_at")
        if uri and uploaded_at:
            try:
                age = now - datetime.fromisoformat(uploaded_at)
            except ValueError:
                return None
            if age < timedelta(hours=GEMINI_FILE_TTL_HOURS):
                return str(uri)
    return None


def _safe_job_name(job_name: str) -> str:
    """Filesystem-safe token for a provider job name (names contain '/')."""
    return job_name.replace("/", "_")


def _load_job_registry(config: Any) -> dict[str, Any]:
    """Return the batch-job registry, or a fresh empty one.

    The registry (``state/batch_jobs.json``) tracks every submitted batch job
    keyed by job name, so a sharded submission's multiple jobs are all
    trackable and recoverable. Tolerant: a missing or corrupt file yields an
    empty registry rather than raising.
    """
    reg_file = config.batch_jobs_registry_file
    if reg_file.exists():
        loaded: Any = None
        try:
            loaded = json.loads(reg_file.read_text())
        except (json.JSONDecodeError, OSError):
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("jobs"), dict):
            return loaded
    return {"version": BATCH_REGISTRY_VERSION, "jobs": {}}


def _save_job_registry(config: Any, registry: dict[str, Any]) -> None:
    """Persist the job registry atomically (tmp + os.replace)."""
    reg_file = config.batch_jobs_registry_file
    os.makedirs(reg_file.parent, exist_ok=True)
    tmp = reg_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp, reg_file)


def _register_job(config: Any, entry: dict[str, Any]) -> None:
    """Insert or replace one job entry, keyed by ``entry['job_name']``."""
    registry = _load_job_registry(config)
    registry["jobs"][entry["job_name"]] = entry
    _save_job_registry(config, registry)


def _update_job_status(config: Any, job_name: str, **fields: Any) -> None:
    """Patch fields on one job entry (no-op if the job is unknown)."""
    registry = _load_job_registry(config)
    entry = registry["jobs"].get(job_name)
    if entry is None:
        return
    entry.update(fields)
    _save_job_registry(config, registry)


def _chunk_request_groups(
    groups: list[list[dict[str, Any]]], max_per_chunk: int
) -> list[list[dict[str, Any]]]:
    """Pack per-PDB request *groups* into chunks of at most *max_per_chunk*.

    A single PDB's runs are never split across chunks; a group larger than the
    cap becomes its own (over-cap) chunk rather than being divided.
    """
    chunks: list[list[dict[str, Any]]] = []
    chunk: list[dict[str, Any]] = []
    for reqs in groups:
        if chunk and len(chunk) + len(reqs) > max_per_chunk:
            chunks.append(chunk)
            chunk = []
        chunk.extend(reqs)
        if len(chunk) >= max_per_chunk:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _submit_batch_chunk(
    config: Any,
    client: Any,
    *,
    model_name: str,
    prompt_id: str | None,
    chunk_requests: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
    detect_advisory_by_pdb: dict[str, list[str]],
    created_at: str,
) -> str:
    """Submit one chunk as a batch job and register it; return the job name.

    Raises on upload/create failure so the caller can isolate a single chunk's
    failure; the temp JSONL is always cleaned up.
    """
    os.makedirs(config.pipeline_runs_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for req in chunk_requests:
            f.write(json.dumps(req) + "\n")
        tmp_jsonl = Path(f.name)

    try:
        batch_src_file = client.files.upload(
            file=str(tmp_jsonl), config={"mime_type": "application/jsonl"}
        )
        if not batch_src_file.name:
            raise ValueError("Uploaded file has no name")

        batch_job = client.batches.create(model=model_name, src=batch_src_file.name)
        if not batch_job.name:
            raise ValueError("Created batch job has no name")
        logger.info(
            "Batch chunk %d/%d submitted: %s (%d requests)",
            chunk_index + 1,
            chunk_count,
            batch_job.name,
            len(chunk_requests),
        )

        # The registry carries the provenance (model / prompt / code_version /
        # advisories) that recover_batch stamps onto each result, so a job's
        # results are always attributed to the model that produced them, with
        # no dependence on a shared sidecar a later submission could overwrite.
        _register_job(
            config,
            {
                "job_name": batch_job.name,
                "status": BATCH_STATUS_SUBMITTED,
                "model_requested": model_name,
                "prompt": prompt_id,
                "code_version": get_code_version(),
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "request_count": len(chunk_requests),
                "created_at": created_at,
                "raw_output_file": None,
                "recovered_at": None,
                "detect_advisory": detect_advisory_by_pdb,
            },
        )

        # Back-compat: keep the single-file pointer to the most recent job so a
        # rollback to pre-registry code still finds a job to poll. The registry
        # is authoritative; this mirror is deprecated.
        tmp_job_file = config.current_batch_job_file.with_suffix(".tmp")
        with open(tmp_job_file, "w") as fj:
            fj.write(batch_job.name)
        os.replace(tmp_job_file, config.current_batch_job_file)
        return str(batch_job.name)
    finally:
        if tmp_jsonl.exists():
            os.remove(tmp_jsonl)


def run_single_pdb(
    pdb_id: str,
    enriched_data: dict,
    prompt_text: str,
    pdf_path: Path,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
    prompt_id: str | None = None,
    temperature: float | None = None,
) -> None:
    """Run annotation for a single PDB entry using parallel Gemini calls.

    Uploads the PDF once, then fans out *num_runs* independent generation
    requests via a thread pool.  Completed runs are persisted atomically
    so the process is safely resumable.
    """
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    out_dir = config.ai_results_dir / pdb_id / model_run_subdir(model_name)

    # Check resumability
    os.makedirs(out_dir, exist_ok=True)
    completed_runs = 0
    for n in range(1, num_runs + 1):
        if (out_dir / f"run_{n}.json").exists():
            completed_runs += 1

    if completed_runs >= num_runs:
        logger.info("[%s] All %d runs already completed. Skipping.", pdb_id, num_runs)
        return

    client = get_client()

    # Compress PDF if needed
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pdf = Path(tmp_dir) / f"{pdb_id}_compressed.pdf"
        try:
            actual_pdf = compress_pdf_if_needed(pdf_path, tmp_pdf)
        except Exception as e:
            logger.error("[%s] PDF compression failed: %s", pdb_id, e)
            return

        # Upload PDF
        try:
            uploaded_file = client.files.upload(
                file=str(actual_pdf), config={"mime_type": "application/pdf"}
            )
        except Exception as e:
            logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
            return

        try:
            # Detect signals route advisory evidence into the prompt + tool; with
            # none (or only review signals) the prompt and config are unchanged.
            detect_signals = load_detect_signals(pdb_id)
            advisory_kinds = sorted(
                {s.kind for s in detect_signals if s.severity == SEVERITY_ADVISORY}
            )
            parts = build_prompt_parts(
                pdb_id, enriched_data, prompt_text, detect_signals=detect_signals
            )
            run_config = build_tool_config(detect_signals, temperature=temperature)
            contents: list[Any] = [*parts, uploaded_file]

            def do_run(run_num: int) -> None:
                out_file = out_dir / f"run_{run_num}.json"
                if out_file.exists():
                    return

                retries = 0
                while retries < GEMINI_MAX_RETRIES:
                    try:
                        # get a potentially rotated client
                        run_client = get_client()
                        response = run_client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=run_config,
                        )

                        if not response.function_calls:
                            raise ValueError("No function calls returned by the model")

                        # Extract the first function call
                        fc = response.function_calls[0]
                        if fc.name != ANNOTATOR_FUNCTION_NAME:
                            raise ValueError(f"Unexpected function call: {fc.name}")

                        args = fc.args
                        if args is None:
                            raise ValueError("Function call missing arguments")

                        # Process and save
                        final_data = post_process_annotation(args)
                        final_data["_provenance"] = {
                            "model_requested": model_name,
                            "model_served": getattr(response, "model_version", None),
                            "prompt": prompt_id,
                            "code_version": get_code_version(),
                            "detect_advisory": advisory_kinds,
                            "run": run_num,
                            "mode": "single",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }

                        # Atomic write
                        tmp_out = out_file.with_suffix(".tmp")
                        with open(tmp_out, "w") as f:
                            json.dump(final_data, f, indent=2)
                        os.replace(tmp_out, out_file)
                        logger.info("[%s] Run %d complete.", pdb_id, run_num)
                        return

                    except APIError as e:
                        retries += 1
                        if e.code == 429:
                            # Rate-limited — longer sleep before retry
                            time.sleep(SLEEP_GEMINI_429 * (2 ** (retries - 1)))
                        else:
                            time.sleep(GEMINI_BASE_BACKOFF * (2 ** (retries - 1)))
                    except Exception as exc:
                        logger.warning(
                            "[%s] Run %d attempt %d failed: %s",
                            pdb_id,
                            run_num,
                            retries + 1,
                            exc,
                        )
                        retries += 1
                        time.sleep(GEMINI_BASE_BACKOFF * (2 ** (retries - 1)))

                logger.error(
                    "[%s] Run %d failed after %d retries.", pdb_id, run_num, GEMINI_MAX_RETRIES
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(num_runs, GEMINI_MAX_WORKERS)
            ) as executor:
                # Consume the iterator so any exception escaping a worker
                # surfaces here instead of being silently discarded.
                list(executor.map(do_run, range(1, num_runs + 1)))

        finally:
            import contextlib

            with contextlib.suppress(Exception):
                if uploaded_file.name:
                    client.files.delete(name=uploaded_file.name)


def build_and_submit_batch(
    targets: list[str],
    prompt_text: str,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
    prompt_id: str | None = None,
    temperature: float | None = None,
) -> None:
    """Build a JSONL payload for all *targets* and submit it to the Gemini Batch API."""
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    client = get_client()

    # Prepare batch requests
    now = datetime.now(UTC)
    request_groups: list[list[dict[str, Any]]] = []
    registry = {}

    # Check if uploaded files registry exists
    reg_file = config.uploaded_files_registry_file
    if reg_file.exists():
        try:
            with open(reg_file) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            pass

    # Per-PDB advisory signal kinds, recorded into the job provenance so
    # recover_batch can stamp each result with the advisories active at submit.
    detect_advisory_by_pdb: dict[str, list[str]] = {}
    for pdb_id in targets:
        enriched_file = config.enriched_dir / f"{pdb_id}.json"
        pdf_file = config.papers_dir / f"{pdb_id}.pdf"

        if not enriched_file.exists() or not pdf_file.exists():
            logger.warning("[%s] Missing enriched data or PDF, skipping batch prep.", pdb_id)
            continue

        try:
            with open(enriched_file) as f:
                enriched_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # One unreadable enriched file must not abort the whole batch.
            logger.warning("[%s] Skipping — unreadable enriched JSON: %s", pdb_id, exc)
            continue

        # Determine runs to do
        out_dir = config.ai_results_dir / pdb_id / model_run_subdir(model_name)
        os.makedirs(out_dir, exist_ok=True)
        runs_to_do = [n for n in range(1, num_runs + 1) if not (out_dir / f"run_{n}.json").exists()]

        if not runs_to_do:
            continue

        # Reuse a cached upload only if it is still within the Files-API TTL.
        pdf_uri = _registry_fresh_uri(registry.get(pdb_id), now)
        if not pdf_uri:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_pdf = Path(tmp_dir) / f"{pdb_id}_compressed.pdf"
                try:
                    actual_pdf = compress_pdf_if_needed(pdf_file, tmp_pdf)
                    uploaded_file = client.files.upload(
                        file=str(actual_pdf), config={"mime_type": "application/pdf"}
                    )
                    pdf_uri = uploaded_file.uri
                    registry[pdb_id] = {"uri": pdf_uri, "uploaded_at": now.isoformat()}
                    logger.info("[%s] Uploaded PDF to %s", pdb_id, pdf_uri)
                except Exception as e:
                    logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
                    continue

        detect_signals = load_detect_signals(pdb_id)
        detect_advisory_by_pdb[pdb_id] = sorted(
            {s.kind for s in detect_signals if s.severity == SEVERITY_ADVISORY}
        )
        parts = build_prompt_parts(
            pdb_id, enriched_data, prompt_text, detect_signals=detect_signals
        )
        tool_for_pdb = build_tool_for_signals(ANNOTATION_TOOL, detect_signals)

        # We need to construct the request dict for the batch API.
        # The schema for the batch API contents is identical to generate_content.
        # Requests are grouped per PDB so chunking never splits one structure's
        # runs across batch jobs.
        pdb_requests: list[dict[str, Any]] = []
        for n in runs_to_do:
            req_id = f"{pdb_id}__run_{n:02d}"

            # Construct the contents array. The File Data needs a specific format.
            contents_batch: list[dict[str, Any]] = []
            for part in parts:
                if isinstance(part, str):
                    contents_batch.append({"parts": [{"text": part}]})
            contents_batch.append(
                {"parts": [{"fileData": {"fileUri": pdf_uri, "mimeType": "application/pdf"}}]}
            )

            # The tool schema must be provided as a dict (per-PDB: augmented when
            # an incidental-candidate signal is present, identical to base otherwise).
            assert tool_for_pdb.function_declarations is not None
            fn_decl = tool_for_pdb.function_declarations[0]
            tool_dict = {
                "functionDeclarations": [
                    {
                        "name": fn_decl.name,
                        "description": fn_decl.description,
                        "parameters": fn_decl.parameters.model_dump(exclude_none=True)
                        if fn_decl.parameters
                        else {},
                    }
                ]
            }

            # Per-request payload. The model is set once at the batch-job level;
            # repeating it per request is rejected as a mismatch. A temperature,
            # however, is per-request (generationConfig) -- omitted entirely when
            # not set, so the default behaviour is unchanged.
            request_payload: dict[str, Any] = {
                "contents": contents_batch,
                "tools": [tool_dict],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            }
            if temperature is not None:
                request_payload["generationConfig"] = {"temperature": temperature}

            # "key" (not "id") is echoed back in the output for correlation.
            pdb_requests.append({"key": req_id, "request": request_payload})

        if pdb_requests:
            request_groups.append(pdb_requests)

    # Save updated registry
    tmp_reg = reg_file.with_suffix(".tmp")
    with open(tmp_reg, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_reg, reg_file)

    total_requests = sum(len(group) for group in request_groups)
    if not total_requests:
        logger.info("No batch requests to submit. All done!")
        return

    # Shard into jobs of at most GEMINI_BATCH_MAX_REQUESTS requests so one
    # oversized submission can't be rejected wholesale or sit in the queue past
    # the provider's 48-hour expiry. Each PDB's runs stay within a single job.
    chunks = _chunk_request_groups(request_groups, GEMINI_BATCH_MAX_REQUESTS)
    submitted = 0
    for chunk_index, chunk in enumerate(chunks):
        try:
            _submit_batch_chunk(
                config,
                client,
                model_name=model_name,
                prompt_id=prompt_id,
                chunk_requests=chunk,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                detect_advisory_by_pdb=detect_advisory_by_pdb,
                created_at=now.isoformat(),
            )
            submitted += 1
        except Exception as exc:
            # One chunk's failure must not lose the chunks already submitted:
            # each is registered the moment it is created and recovered
            # independently, and the remaining outstanding runs are simply
            # re-chunked on the next run (completed runs are skipped).
            logger.error(
                "Batch chunk %d/%d failed to submit: %s", chunk_index + 1, len(chunks), exc
            )
    logger.info("Submitted %d/%d batch chunk(s).", submitted, len(chunks))


def check_batch_status() -> None:
    """Poll the Gemini Batch API for all tracked jobs and download finished ones."""
    config = get_config()
    client = get_client()

    registry = _load_job_registry(config)

    # Migration: a workspace from before the registry has only the single-file
    # pointer. Adopt that in-flight job so it is tracked and recovered like any
    # other (model/prompt are back-filled from its sidecar at recover time).
    if not registry["jobs"] and config.current_batch_job_file.exists():
        legacy_name = config.current_batch_job_file.read_text().strip()
        if legacy_name:
            _register_job(
                config,
                {
                    "job_name": legacy_name,
                    "status": BATCH_STATUS_SUBMITTED,
                    "model_requested": None,
                    "prompt": None,
                    "code_version": None,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "request_count": None,
                    "created_at": None,
                    "raw_output_file": None,
                    "recovered_at": None,
                    "detect_advisory": {},
                },
            )
            registry = _load_job_registry(config)

    pending = [e for e in registry["jobs"].values() if e.get("status") == BATCH_STATUS_SUBMITTED]
    if not pending:
        logger.info("No active batch job found in state.")
        return

    # The SDK exposes terminal states as JOB_STATE_* on ``job.state.name``.
    # Some terminal states carry results to download; others do not.
    succeeded_states = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")
    failed_states = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")

    downloaded_any = False
    for entry in pending:
        job_name = entry["job_name"]
        try:
            job = client.batches.get(name=job_name)
        except Exception as e:
            logger.error("Failed to get batch job %s: %s", job_name, e)
            continue

        state = job.state.name if job.state else ""
        logger.info("Batch Job %s is in state: %s", job_name, state)

        if state in failed_states:
            expired_note = (
                " -- it ran or waited past the provider's 48-hour limit; resubmit or split the batch"
                if state == "JOB_STATE_EXPIRED"
                else ""
            )
            logger.error(
                "Batch job %s ended without results (%s)%s: %s",
                job_name,
                state,
                expired_note,
                job.error,
            )
            _update_job_status(config, job_name, status=BATCH_STATUS_FAILED)
            continue

        if state not in succeeded_states:
            logger.info(
                "Batch job %s is not finished yet (state %s); try again later.", job_name, state
            )
            continue

        if not (job.dest and job.dest.file_name):
            logger.error("Batch job %s reported %s but exposed no result file.", job_name, state)
            continue

        try:
            os.makedirs(config.pipeline_runs_dir, exist_ok=True)
            raw_out_file = config.pipeline_runs_dir / f"raw_output_{_safe_job_name(job_name)}.jsonl"
            logger.info("Downloading %s to %s", job.dest.file_name, raw_out_file)
            content = client.files.download(file=job.dest.file_name)
            with open(raw_out_file, "wb") as f_out:
                f_out.write(content)
            _update_job_status(
                config, job_name, status=BATCH_STATUS_DOWNLOADED, raw_output_file=str(raw_out_file)
            )
            downloaded_any = True
        except Exception as e:  # surface any download failure without crashing
            logger.error("Failed to download batch results for %s: %s", job_name, e)

    # Recover whenever any downloaded-but-not-yet-recovered job exists -- not
    # only the ones downloaded this round -- so a crash between a download and
    # its recovery is healed on the next poll rather than stranding the result.
    registry = _load_job_registry(config)
    if downloaded_any or any(
        e.get("status") == BATCH_STATUS_DOWNLOADED for e in registry["jobs"].values()
    ):
        logger.info("Download(s) complete. Running recovery to parse results.")
        recover_batch()


def recover_batch() -> None:
    """Re-process raw JSONL batch output into individual per-run JSON files."""
    config = get_config()
    runs_dir = config.pipeline_runs_dir

    if not runs_dir.exists():
        logger.info("No pipeline runs directory found.")
        return

    registry = _load_job_registry(config)
    # Map a downloaded raw-output filename to its authoritative job entry, so
    # each result is attributed to the model/prompt of the job that produced
    # it -- not a shared sidecar a later submission may have overwritten.
    by_raw = {
        Path(e["raw_output_file"]).name: e
        for e in registry["jobs"].values()
        if e.get("raw_output_file")
    }

    def _load_provenance(raw_file: Path) -> dict:
        # Legacy/migration fallback for raw outputs with no registry entry
        # (downloaded before the registry existed): match the per-job sidecar
        # (raw_output_<job>.jsonl -> _batch_provenance_<job>.json), then the
        # legacy shared file. Without the per-job match, a stale raw file from
        # an earlier job would be stamped with a later job's model.
        job_suffix = raw_file.stem.removeprefix("raw_output_")
        for prov_file in (
            runs_dir / f"_batch_provenance_{job_suffix}.json",
            runs_dir / "_batch_provenance.json",
        ):
            if prov_file.exists():
                try:
                    loaded = json.loads(prov_file.read_text())
                except (json.JSONDecodeError, OSError):
                    return {}
                return loaded if isinstance(loaded, dict) else {}
        return {}

    for raw_file in runs_dir.glob("raw_output_*.jsonl"):
        entry = by_raw.get(raw_file.name)
        batch_meta: dict[str, Any]
        if entry is not None and entry.get("model_requested"):
            batch_meta = {
                "model_requested": entry.get("model_requested"),
                "prompt": entry.get("prompt"),
                "code_version": entry.get("code_version"),
                "detect_advisory": entry.get("detect_advisory") or {},
            }
        else:
            # No registry entry, OR a migration-adopted entry whose real model
            # lives only in the legacy per-job sidecar (the adopted entry has
            # model_requested=None). Resolve from the sidecar / shared file so
            # the result keeps its true model and per-model output directory.
            batch_meta = _load_provenance(raw_file)
        logger.info("Processing %s...", raw_file.name)
        with open(raw_file) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    req_id = data.get("key") or data.get("id")
                    if not req_id or "__run_" not in req_id:
                        continue

                    # rpartition (not split("__")) so a PDB id that itself
                    # contains "__" doesn't unpack-error and drop the run.
                    pdb_id, _, run_part = req_id.rpartition("__run_")
                    run_num = int(run_part)

                    out_dir = (
                        config.ai_results_dir
                        / pdb_id
                        / model_run_subdir(batch_meta.get("model_requested"))
                    )
                    out_file = out_dir / f"run_{run_num}.json"
                    # Resume-by-existence: never clobber an already-recovered
                    # run. Makes recover idempotent and immune to a stale or
                    # re-downloaded raw file overwriting good results.
                    if out_file.exists():
                        continue

                    response_obj = data.get("response", {})
                    candidates = response_obj.get("candidates") or []
                    if not candidates:
                        logger.warning(
                            "[%s] Run %d: no candidates in batch response (line %d)",
                            pdb_id,
                            run_num,
                            line_no,
                        )
                        continue

                    content = candidates[0].get("content") or {}
                    parts = content.get("parts") or []
                    matched = False
                    for part in parts:
                        fc = part.get("functionCall")
                        if fc and fc.get("name") == ANNOTATOR_FUNCTION_NAME:
                            args = fc.get("args")
                            if args is None:
                                logger.warning(
                                    "[%s] Run %d: function call has no args (line %d)",
                                    pdb_id,
                                    run_num,
                                    line_no,
                                )
                                break
                            final_data = post_process_annotation(args)
                            final_data["_provenance"] = {
                                "model_requested": batch_meta.get("model_requested"),
                                "model_served": response_obj.get("modelVersion"),
                                "prompt": batch_meta.get("prompt"),
                                # From the submission record: the code that built
                                # and submitted the batch, not the recovery run.
                                "code_version": batch_meta.get("code_version"),
                                "detect_advisory": (batch_meta.get("detect_advisory") or {}).get(
                                    pdb_id, []
                                ),
                                "run": run_num,
                                "mode": "batch",
                                "timestamp": datetime.now(UTC).isoformat(),
                            }

                            os.makedirs(out_dir, exist_ok=True)
                            tmp_out = out_file.with_suffix(".tmp")
                            with open(tmp_out, "w") as f_out:
                                json.dump(final_data, f_out, indent=2)
                            os.replace(tmp_out, out_file)
                            matched = True
                            break

                    if not matched:
                        logger.warning(
                            "[%s] Run %d: no matching function call in response (line %d)",
                            pdb_id,
                            run_num,
                            line_no,
                        )
                except Exception as e:
                    logger.error(
                        "Row-level Error Isolation: Failed to process line %d in %s: %s",
                        line_no,
                        raw_file.name,
                        e,
                    )
                    continue

        if entry is not None:
            _update_job_status(
                config,
                entry["job_name"],
                status=BATCH_STATUS_RECOVERED,
                recovered_at=datetime.now(UTC).isoformat(),
            )


def discover_annotation_targets(num_runs: int, model_name: str) -> list[str]:
    """Enriched PDB IDs that still need annotation runs for *model_name*.

    A PDB counts as done when its per-model run directory already holds
    *num_runs* run files; those are excluded. Model-aware so it matches the
    namespaced output layout written by the runners.
    """
    config = get_config()
    enriched_pdbs = {p.stem.upper() for p in config.enriched_dir.glob("*.json")}
    done: set[str] = set()
    if config.ai_results_dir.exists():
        for d in config.ai_results_dir.iterdir():
            if not d.is_dir():
                continue
            model_dir = d / model_run_subdir(model_name)
            completed = sum(
                1 for n in range(1, num_runs + 1) if (model_dir / f"run_{n}.json").exists()
            )
            if completed >= num_runs:
                done.add(d.name.upper())
    return sorted(enriched_pdbs - done)


def run_annotation_stage(
    pdb_id: str | None = None,
    targets_file: str | None = None,
    prompt_file: str | None = None,
    model: str | None = None,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    batch: bool = False,
    temperature: float | None = None,
) -> None:
    """Resolve targets / prompt / model and run annotation (single or batch).

    Shared by the ``annotate`` and ``pipeline`` commands. Auto-discovers
    enriched PDBs that still need runs when no explicit target is given.
    Raises ``FileNotFoundError`` when no prompt is available.
    """
    config = get_config()

    # Fail fast on a stale / missing storage contract BEFORE the expensive AI
    # calls. Previously only the interactive curate step validated the contract,
    # so a layout mismatch surfaced only after annotation had already run.
    from gpcr_tools.workspace import validate_contract

    validate_contract(config)

    model_name = model or get_gemini_model_name()

    if pdb_id:
        pdb_ids = [pdb_id.upper()]
    elif targets_file:
        from gpcr_tools.fetcher.targets import read_targets

        pdb_ids = read_targets(Path(targets_file))
    else:
        pdb_ids = discover_annotation_targets(num_runs, model_name)

    if prompt_file:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8")
        prompt_id = Path(prompt_file).stem
    elif config.default_prompt_file.exists():
        prompt_text = config.default_prompt_file.read_text(encoding="utf-8")
        prompt_id = config.default_prompt_file.stem
    else:
        raise FileNotFoundError(
            f"Default prompt file not found at {config.default_prompt_file}; "
            "create it or pass a prompt file."
        )

    if batch:
        build_and_submit_batch(
            pdb_ids,
            prompt_text,
            num_runs=num_runs,
            model_name=model_name,
            prompt_id=prompt_id,
            temperature=temperature,
        )
        return

    for pid in pdb_ids:
        enriched_path = config.enriched_dir / f"{pid}.json"
        if not enriched_path.exists():
            logger.warning("Skipping %s: no enriched data at %s", pid, enriched_path)
            continue
        try:
            with open(enriched_path, encoding="utf-8") as fh:
                enriched_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: unreadable enriched JSON: %s", pid, exc)
            continue
        pdf_path = config.papers_dir / f"{pid}.pdf"
        if not pdf_path.exists():
            logger.warning("Skipping %s: no PDF at %s", pid, pdf_path)
            continue
        run_single_pdb(
            pdb_id=pid,
            enriched_data=enriched_data,
            prompt_text=prompt_text,
            pdf_path=pdf_path,
            num_runs=num_runs,
            model_name=model_name,
            prompt_id=prompt_id,
            temperature=temperature,
        )
