"""Annotation runner -- single-PDB, batch submission, and recovery."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google.genai.errors import APIError

from gpcr_tools.annotator.gemini_client import get_client
from gpcr_tools.annotator.pdf_compressor import compress_pdf_if_needed
from gpcr_tools.annotator.post_processor import post_process_annotation
from gpcr_tools.annotator.prompt_builder import build_prompt_parts
from gpcr_tools.annotator.schema import ANNOTATION_TOOL, TOOL_CONFIG
from gpcr_tools.config import (
    ANNOTATOR_FUNCTION_NAME,
    GEMINI_BASE_BACKOFF,
    GEMINI_DEFAULT_RUNS,
    GEMINI_MAX_RETRIES,
    GEMINI_MAX_WORKERS,
    SLEEP_GEMINI_429,
    get_config,
    get_gemini_model_name,
    model_run_subdir,
)

logger = logging.getLogger(__name__)


def run_single_pdb(
    pdb_id: str,
    enriched_data: dict,
    prompt_text: str,
    pdf_path: Path,
    num_runs: int = GEMINI_DEFAULT_RUNS,
    model_name: str | None = None,
    prompt_id: str | None = None,
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
            parts = build_prompt_parts(pdb_id, enriched_data, prompt_text)
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
                            config=TOOL_CONFIG,
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
                executor.map(do_run, range(1, num_runs + 1))

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
) -> None:
    """Build a JSONL payload for all *targets* and submit it to the Gemini Batch API."""
    model_name = model_name or get_gemini_model_name()
    config = get_config()
    client = get_client()

    # Prepare batch requests
    requests = []
    registry = {}

    # Check if uploaded files registry exists
    reg_file = config.uploaded_files_registry_file
    if reg_file.exists():
        try:
            with open(reg_file) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            pass

    for pdb_id in targets:
        enriched_file = config.enriched_dir / f"{pdb_id}.json"
        pdf_file = config.papers_dir / f"{pdb_id}.pdf"

        if not enriched_file.exists() or not pdf_file.exists():
            logger.warning("[%s] Missing enriched data or PDF, skipping batch prep.", pdb_id)
            continue

        with open(enriched_file) as f:
            enriched_data = json.load(f)

        # Determine runs to do
        out_dir = config.ai_results_dir / pdb_id / model_run_subdir(model_name)
        os.makedirs(out_dir, exist_ok=True)
        runs_to_do = [n for n in range(1, num_runs + 1) if not (out_dir / f"run_{n}.json").exists()]

        if not runs_to_do:
            continue

        # Upload or get PDF
        pdf_uri = registry.get(pdb_id)
        if not pdf_uri:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_pdf = Path(tmp_dir) / f"{pdb_id}_compressed.pdf"
                try:
                    actual_pdf = compress_pdf_if_needed(pdf_file, tmp_pdf)
                    uploaded_file = client.files.upload(
                        file=str(actual_pdf), config={"mime_type": "application/pdf"}
                    )
                    pdf_uri = uploaded_file.uri
                    registry[pdb_id] = pdf_uri
                    logger.info("[%s] Uploaded PDF to %s", pdb_id, pdf_uri)
                except Exception as e:
                    logger.error("[%s] Failed to upload PDF: %s", pdb_id, e)
                    continue

        parts = build_prompt_parts(pdb_id, enriched_data, prompt_text)

        # We need to construct the request dict for the batch API
        # The schema for the batch API contents is identical to generate_content
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

            # The tool schema must be provided as a dict
            assert ANNOTATION_TOOL.function_declarations is not None
            fn_decl = ANNOTATION_TOOL.function_declarations[0]
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

            requests.append(
                {
                    # Per-request identifier echoed back in the output ("key", not
                    # "id"). The model is set once at the batch-job level below;
                    # repeating it per request is rejected as a mismatch.
                    "key": req_id,
                    "request": {
                        "contents": contents_batch,
                        "tools": [tool_dict],
                        "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
                    },
                }
            )

    # Save updated registry
    tmp_reg = reg_file.with_suffix(".tmp")
    with open(tmp_reg, "w") as f:
        json.dump(registry, f, indent=2)
    os.replace(tmp_reg, reg_file)

    if not requests:
        logger.info("No batch requests to submit. All done!")
        return

    # Write JSONL
    os.makedirs(config.pipeline_runs_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
        tmp_jsonl = Path(f.name)

    try:
        # Upload JSONL to Gemini
        batch_src_file = client.files.upload(
            file=str(tmp_jsonl), config={"mime_type": "application/jsonl"}
        )
        if not batch_src_file.name:
            raise ValueError("Uploaded file has no name")
        logger.info("Uploaded batch JSONL source: %s", batch_src_file.name)

        # Submit batch
        batch_job = client.batches.create(model=model_name, src=batch_src_file.name)
        if not batch_job.name:
            raise ValueError("Created batch job has no name")
        logger.info("Batch submitted successfully! Job Name: %s", batch_job.name)

        # Save job name
        tmp_job_file = config.current_batch_job_file.with_suffix(".tmp")
        with open(tmp_job_file, "w") as f:
            f.write(batch_job.name)
        os.replace(tmp_job_file, config.current_batch_job_file)

        # Record this job's provenance under a filename keyed to the job, so
        # recover_batch can stamp each result with the model/prompt of the job
        # that actually produced it. A single shared file would be overwritten
        # by the next submission and re-attribute an earlier job's results to
        # the wrong model (and the wrong per-model output directory).
        safe_name = batch_job.name.replace("/", "_")
        batch_prov_file = config.pipeline_runs_dir / f"_batch_provenance_{safe_name}.json"
        tmp_prov = batch_prov_file.with_suffix(".tmp")
        with open(tmp_prov, "w") as f:
            json.dump({"model_requested": model_name, "prompt": prompt_id}, f, indent=2)
        os.replace(tmp_prov, batch_prov_file)

    finally:
        if tmp_jsonl.exists():
            os.remove(tmp_jsonl)


def check_batch_status() -> None:
    """Poll the Gemini Batch API for the current job and download results when complete."""
    config = get_config()
    job_file = config.current_batch_job_file

    if not job_file.exists():
        logger.info("No active batch job found in state.")
        return

    with open(job_file) as f:
        job_name = f.read().strip()

    client = get_client()
    try:
        job = client.batches.get(name=job_name)
    except Exception as e:
        logger.error("Failed to get batch job %s: %s", job_name, e)
        return

    state = job.state.name if job.state else ""
    logger.info("Batch Job %s is in state: %s", job_name, state)

    # The SDK exposes terminal states as JOB_STATE_* on ``job.state.name``.
    # Some terminal states carry results to download; others do not.
    succeeded_states = ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")
    failed_states = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")

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
        return

    if state not in succeeded_states:
        logger.info(
            "Batch job %s is not finished yet (state %s); try again later.", job_name, state
        )
        return

    if not (job.dest and job.dest.file_name):
        logger.error("Batch job %s reported %s but exposed no result file.", job_name, state)
        return

    logger.info("Batch has completed. Downloading results...")
    try:
        os.makedirs(config.pipeline_runs_dir, exist_ok=True)
        safe_name = job_name.replace("/", "_")
        raw_out_file = config.pipeline_runs_dir / f"raw_output_{safe_name}.jsonl"
        logger.info("Downloading %s to %s", job.dest.file_name, raw_out_file)
        content = client.files.download(file=job.dest.file_name)
        with open(raw_out_file, "wb") as f_out:
            f_out.write(content)
        logger.info("Download complete. Running recovery to parse results.")
        recover_batch()
    except Exception as e:  # surface any download/parse failure without crashing
        logger.error("Failed to download batch results: %s", e)


def recover_batch() -> None:
    """Re-process raw JSONL batch output into individual per-run JSON files."""
    config = get_config()
    runs_dir = config.pipeline_runs_dir

    if not runs_dir.exists():
        logger.info("No pipeline runs directory found.")
        return

    def _load_provenance(raw_file: Path) -> dict:
        # Match each raw output to its own job's provenance
        # (raw_output_<job>.jsonl -> _batch_provenance_<job>.json), falling back
        # to the legacy shared file for outputs downloaded before per-job
        # provenance existed. Without the per-job match, a stale raw file from
        # an earlier job would be stamped with a later job's model.
        job_suffix = raw_file.stem.removeprefix("raw_output_")
        for prov_file in (
            runs_dir / f"_batch_provenance_{job_suffix}.json",
            runs_dir / "_batch_provenance.json",
        ):
            if prov_file.exists():
                try:
                    return json.loads(prov_file.read_text())
                except (json.JSONDecodeError, OSError):
                    return {}
        return {}

    for raw_file in runs_dir.glob("raw_output_*.jsonl"):
        batch_meta = _load_provenance(raw_file)
        logger.info("Processing %s...", raw_file.name)
        with open(raw_file) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    req_id = data.get("key") or data.get("id")
                    if not req_id or "__run_" not in req_id:
                        continue

                    pdb_id, run_part = req_id.split("__")
                    run_num = int(run_part.replace("run_", ""))

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
                                "run": run_num,
                                "mode": "batch",
                                "timestamp": datetime.now(UTC).isoformat(),
                            }

                            out_dir = (
                                config.ai_results_dir
                                / pdb_id
                                / model_run_subdir(batch_meta.get("model_requested"))
                            )
                            os.makedirs(out_dir, exist_ok=True)
                            out_file = out_dir / f"run_{run_num}.json"

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
) -> None:
    """Resolve targets / prompt / model and run annotation (single or batch).

    Shared by the ``annotate`` and ``pipeline`` commands. Auto-discovers
    enriched PDBs that still need runs when no explicit target is given.
    Raises ``FileNotFoundError`` when no prompt is available.
    """
    config = get_config()
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
        )
        return

    for pid in pdb_ids:
        enriched_path = config.enriched_dir / f"{pid}.json"
        if not enriched_path.exists():
            logger.warning("Skipping %s: no enriched data at %s", pid, enriched_path)
            continue
        with open(enriched_path, encoding="utf-8") as fh:
            enriched_data = json.load(fh)
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
        )
