"""Multi-tier PDF downloader for GPCR papers.

Resolves the DOI (recovering it from the citation table / a title search when
the primary citation lacks one), then tries candidate PDF URLs as a true
fallback chain -- a URL that resolves but yields a non-PDF (an HTML challenge)
or a 403/404 does not end the search, only an exhausted chain marks the paper
paywalled:
  CrossRef metadata — PMID/PMCID and a direct publisher PDF link
  Unpaywall — best OA PDF link
  PMC open-access S3 bucket — by PMCID (promoting a PMID to a PMCID first)
  Fallback — mark ``"fallback_paywalled"`` (left for manual download)

Reads enriched JSON from ``enriched/{pdb_id}.json``, writes PDFs to
``papers/{pdb_id}.pdf``, and updates ``state/download_log.json``
via atomic write after each PDB.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gpcr_tools.config import (
    CROSSREF_API_URL,
    DL_STATUS_FAILED_NO_DATA,
    DL_STATUS_FAILED_NO_DOI,
    DL_STATUS_PAYWALLED,
    DL_STATUS_SKIPPED_EXISTS,
    DL_STATUS_SKIPPED_NO_ENRICHED,
    DL_STATUS_SUCCESS,
    HTTP_RETRY_ALLOWED_METHODS,
    HTTP_RETRY_BACKOFF_FACTOR,
    HTTP_RETRY_CONNECT,
    HTTP_RETRY_READ,
    HTTP_RETRY_STATUS_FORCELIST,
    HTTP_RETRY_TOTAL,
    NCBI_IDCONV_URL,
    PDF_DOWNLOAD_CHUNK_SIZE,
    PDF_MIN_VALID_BYTES,
    PMC_S3_BASE_URL,
    SLEEP_NCBI_RATE_LIMIT,
    TIMEOUT_CROSSREF,
    TIMEOUT_NCBI_PMC_OA,
    TIMEOUT_PDF_DOWNLOAD,
    TIMEOUT_UNPAYWALL,
    UNPAYWALL_API_URL,
    get_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PDF_MAGIC = b"%PDF"


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------


def _build_session(email: str) -> requests.Session:
    """Build a requests Session with retry adapter and polite headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"LitFetcher/2.0 (mailto:{email})",
            "From": email,
        }
    )
    retry = Retry(
        total=HTTP_RETRY_TOTAL,
        read=HTTP_RETRY_READ,
        connect=HTTP_RETRY_CONNECT,
        backoff_factor=HTTP_RETRY_BACKOFF_FACTOR,
        status_forcelist=HTTP_RETRY_STATUS_FORCELIST,
        allowed_methods=list(HTTP_RETRY_ALLOWED_METHODS),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Download log (atomic read/write)
# ---------------------------------------------------------------------------


def _read_download_log() -> dict[str, Any]:
    """Read the download log, returning empty dict if absent or corrupt."""
    cfg = get_config()
    path = cfg.download_log_file
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read download log: %s", exc)
        return {}


def _update_download_log(pdb_id: str, entry: dict[str, Any]) -> None:
    """Atomic read-modify-write for the download log.

    Follows the same pattern as ``aggregator/runner._update_aggregate_log``.
    """
    cfg = get_config()
    path = cfg.download_log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    log_data = _read_download_log()
    log_data[pdb_id] = entry

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(path.parent),
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fd:
            tmp_path = fd.name
            json.dump(log_data, fd, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
        tmp_path = None
    except OSError as exc:
        logger.error("Failed to write download log: %s", exc)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tier API functions
# ---------------------------------------------------------------------------


def _fetch_crossref_metadata(doi: str, session: requests.Session) -> dict[str, str | None]:
    """Tier 0: CrossRef metadata -- PMID, PMCID, and a direct PDF link if present.

    Many gold/hybrid-OA publishers (notably Nature-brand journals) expose a direct
    ``application/pdf`` link in CrossRef even when Unpaywall carries no PDF URL, so
    capturing it here is the single largest DOI-only recovery.
    """
    url = f"{CROSSREF_API_URL}/{doi}"
    try:
        response = session.get(url, timeout=TIMEOUT_CROSSREF)
        if response.status_code == 200:
            data = response.json().get("message") or {}
            pmid = data.get("PMID")
            pmcid: str | None = None
            pdf_url: str | None = None
            for link in data.get("link") or []:
                link_url = link.get("URL") or ""
                content_type = link.get("content-type") or ""
                if pdf_url is None and (
                    "application/pdf" in content_type or link_url.lower().endswith(".pdf")
                ):
                    pdf_url = link_url
                if "www.ncbi.nlm.nih.gov/pmc/articles/PMC" in link_url:
                    match = re.search(r"PMC(\d+)", link_url)
                    if match:
                        pmcid = match.group(1)
            return {"pmid": pmid, "pmcid": pmcid, "crossref_pdf_url": pdf_url}
    except requests.exceptions.RequestException as exc:
        logger.warning("[CrossRef] Failed for DOI %s: %s", doi, exc)
    return {"pmid": None, "pmcid": None, "crossref_pdf_url": None}


def _fetch_unpaywall_pdf_url(
    doi: str,
    session: requests.Session,
    email: str | None = None,
) -> str | None:
    """Tier 1: Get OA PDF URL from Unpaywall."""
    url = f"{UNPAYWALL_API_URL}/{doi}"
    params: dict[str, str] = {}
    if email:
        params["email"] = email
    try:
        response = session.get(url, params=params, timeout=TIMEOUT_UNPAYWALL)
        if response.status_code == 200:
            data = response.json()
            oa_location = data.get("best_oa_location") or {}
            pdf_url = oa_location.get("url_for_pdf")
            if pdf_url:
                return pdf_url  # type: ignore[no-any-return]
    except requests.exceptions.RequestException as exc:
        logger.warning("[Unpaywall] Failed for DOI %s: %s", doi, exc)
    return None


def _fetch_pmc_s3_pdf_url(pmcid: str, session: requests.Session) -> str | None:
    """Resolve the article PDF in the PMC open-access S3 bucket.

    Replaces the retired NCBI OA FTP interface (the legacy oa_pdf paths now 404,
    and the web reader endpoint is behind a bot challenge). Confirms via the
    per-article metadata JSON that a PDF exists, then returns the canonical S3 PDF
    URL. ``None`` when the article is not in the open-access subset.
    """
    pmcid_norm = pmcid.upper()
    if not pmcid_norm.startswith("PMC"):
        pmcid_norm = f"PMC{pmcid_norm}"
    meta_url = f"{PMC_S3_BASE_URL}/metadata/{pmcid_norm}.1.json"
    try:
        response = session.get(meta_url, timeout=TIMEOUT_NCBI_PMC_OA)
        if response.status_code != 200:
            return None
        # ``pdf_url`` is the authoritative object path as an ``s3://bucket/key``
        # URI with an optional ``?md5=`` query; map it to the public HTTPS object
        # (this also carries the correct version, so no version is assumed). Null
        # when the article has no PDF in the open-access subset.
        pdf_field = (response.json() or {}).get("pdf_url")
        if not pdf_field:
            return None
        pdf_field = str(pdf_field)
        if pdf_field.startswith("s3://"):
            key = pdf_field.removeprefix("s3://").split("/", 1)[1].split("?", 1)[0]
            return f"{PMC_S3_BASE_URL}/{key}"
        if pdf_field.startswith("http"):
            return pdf_field.split("?", 1)[0]
        return f"{PMC_S3_BASE_URL}/{pdf_field.lstrip('/').split('?', 1)[0]}"
    except (requests.exceptions.RequestException, ValueError, IndexError) as exc:
        logger.warning("[PMC S3] Failed for PMCID %s: %s", pmcid, exc)
    return None


def _promote_pmid_to_pmcid(pmid: str, session: requests.Session) -> str | None:
    """Map a PMID to a PMCID via the NCBI ID Converter.

    Lets a DOI-only paper that nonetheless has a free PMC full text reach the PMC
    open-access route. Returns ``PMC...`` or ``None``.
    """
    params = {"ids": str(pmid), "format": "json", "tool": "litfetcher"}
    try:
        response = session.get(NCBI_IDCONV_URL, params=params, timeout=TIMEOUT_NCBI_PMC_OA)
        time.sleep(SLEEP_NCBI_RATE_LIMIT)  # the ID Converter rate-limits aggressively
        if response.status_code == 200:
            records = response.json().get("records") or []
            if records and records[0].get("pmcid"):
                return str(records[0]["pmcid"])
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("[ID Converter] Failed for PMID %s: %s", pmid, exc)
    return None


_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Boilerplate dropped before matching, so two distinct structural-biology titles
# do not look similar merely by sharing "structure of the ... receptor".
_TITLE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "to",
        "and",
        "or",
        "for",
        "with",
        "by",
        "structure",
        "structural",
        "crystal",
        "cryo",
        "em",
        "basis",
    }
)
_TITLE_CONTAINMENT_MIN = 0.85


def _title_tokens(text: str) -> set[str]:
    return {t for t in _TITLE_TOKEN_RE.findall(text.lower()) if t not in _TITLE_STOPWORDS}


def _fetch_doi_from_title(title: str, session: requests.Session) -> str | None:
    """Recover a DOI from a paper title via CrossRef bibliographic search.

    Accepts the top hit only when the two titles are near-identical in BOTH
    directions (symmetric containment >= 0.85 over content tokens, stopwords
    dropped). This rejects companion/series papers ("Structure of X" vs "Structure
    of X bound to Y"), because attaching the WRONG paper is worse than none.
    """
    want = _title_tokens(title)
    if len(want) < 3:  # too few content tokens to disambiguate safely
        return None
    params = {"query.bibliographic": title, "rows": "1"}
    try:
        response = session.get(CROSSREF_API_URL, params=params, timeout=TIMEOUT_CROSSREF)
        if response.status_code == 200:
            items = (response.json().get("message") or {}).get("items") or []
            if items:
                got = _title_tokens(" ".join(items[0].get("title") or []))
                overlap = len(want & got)
                if (
                    got
                    and overlap / len(want) >= _TITLE_CONTAINMENT_MIN
                    and overlap / len(got) >= _TITLE_CONTAINMENT_MIN
                ):
                    doi = items[0].get("DOI")
                    return str(doi) if doi else None
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("[CrossRef title] Failed for %r: %s", title[:60], exc)
    return None


def _recover_missing_doi(
    entry_data: dict[str, Any], pmid: Any, session: requests.Session
) -> tuple[str | None, Any]:
    """Best-effort DOI recovery when the primary citation carries none.

    Reads the already-fetched ``citation`` table (its DOI/PubMed fields), then
    falls back to a CrossRef title search. Returns ``(doi, pmid)``.
    """
    citations = [c for c in (entry_data.get("citation") or []) if isinstance(c, dict)]
    primary = next((c for c in citations if c.get("id") == "primary"), None)
    if primary is None and citations:
        primary = citations[0]
    if not primary:
        return None, pmid
    doi = primary.get("pdbx_database_id_DOI")
    pmid = pmid or primary.get("pdbx_database_id_PubMed")
    if doi:
        return str(doi), pmid
    title = primary.get("title")
    if title:
        return _fetch_doi_from_title(str(title), session), pmid
    return None, pmid


def _download_file(url: str, output_path: Path, session: requests.Session) -> bool:
    """Download a file from *url* to *output_path* with streaming."""
    try:
        response = session.get(url, timeout=TIMEOUT_PDF_DOWNLOAD, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=PDF_DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

        # Validate the downloaded content: real PDF magic AND a plausible size.
        # The magic check rejects HTML bot-challenge pages; the size floor rejects
        # tiny error stubs / a stream truncated after the first chunk (which would
        # otherwise be promoted to the final path and never retried).
        with open(output_path, "rb") as f:
            header = f.read(len(_PDF_MAGIC))
        size = output_path.stat().st_size
        if not header.startswith(_PDF_MAGIC) or size < PDF_MIN_VALID_BYTES:
            logger.warning(
                "Downloaded content from %s is not a valid PDF (header: %r, %d bytes)",
                url,
                header[:16],
                size,
            )
            with contextlib.suppress(OSError):
                output_path.unlink()
            return False

        return True
    except (requests.exceptions.RequestException, OSError) as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        with contextlib.suppress(OSError):
            output_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_paper_for_pdb(
    pdb_id: str,
    *,
    session: requests.Session | None = None,
    email: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download the paper for a single PDB ID.

    Return the download log entry dict.
    """
    cfg = get_config()
    pdb_id = pdb_id.upper()
    final_pdf = cfg.papers_dir / f"{pdb_id}.pdf"
    enriched_path = cfg.enriched_dir / f"{pdb_id}.json"

    now = datetime.now(UTC).isoformat()

    # Input guard
    if not enriched_path.exists():
        logger.warning("[%s] Enriched data not found, skipping", pdb_id)
        entry: dict[str, Any] = {
            "status": DL_STATUS_SKIPPED_NO_ENRICHED,
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Resumability
    if final_pdf.exists() and not force:
        logger.info("[%s] PDF already exists, skipping", pdb_id)
        entry = {
            "status": DL_STATUS_SKIPPED_EXISTS,
            "source": None,
            "file_path": str(final_pdf),
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Read enriched data
    try:
        with open(enriched_path, encoding="utf-8") as f:
            pdb_data: dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("[%s] Failed to read enriched JSON: %s", pdb_id, exc)
        entry = {
            "status": DL_STATUS_FAILED_NO_DATA,
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Resolve session
    resolved_email = email or os.environ.get("GPCR_EMAIL_FOR_APIS") or ""
    if not resolved_email:
        logger.error("GPCR_EMAIL_FOR_APIS is not set")
        entry = {
            "status": DL_STATUS_FAILED_NO_DATA,
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": None,
            "pmcid": None,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    sess = session or _build_session(resolved_email)

    # Extract identifiers from enriched data
    entry_data = (pdb_data.get("data") or {}).get("entry") or {}
    doi = (entry_data.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI")
    pmid = (entry_data.get("rcsb_entry_container_identifiers") or {}).get("pubmed_id")
    pmcid = (entry_data.get("pubmed") or {}).get("rcsb_pubmed_central_id")

    if not doi:
        # No DOI on the primary citation -- try the already-fetched citation[]
        # table and a CrossRef title search before giving up; many entries have a
        # published paper RCSB simply did not tag with a DOI.
        doi, pmid = _recover_missing_doi(entry_data, pmid, sess)

    if not doi:
        logger.info("[%s] No DOI found", pdb_id)
        entry = {
            "status": DL_STATUS_FAILED_NO_DOI,
            "source": None,
            "file_path": None,
            "doi": None,
            "pmid": pmid,
            "pmcid": pmcid,
            "timestamp": now,
        }
        _update_download_log(pdb_id, entry)
        return entry

    # Tier 0: CrossRef metadata (PMID/PMCID + a direct publisher PDF link).
    crossref = _fetch_crossref_metadata(doi, sess)
    pmid = crossref.get("pmid") or pmid
    pmcid = crossref.get("pmcid") or pmcid

    cfg.papers_dir.mkdir(parents=True, exist_ok=True)
    temp_pdf = cfg.papers_dir / f"{pdb_id}_temp.pdf"

    # Ordered candidate resolvers, tried as a TRUE fallback chain: a URL that
    # resolves but yields a non-PDF (e.g. an HTML bot challenge) or a 403/404 does
    # NOT end the search -- only an exhausted chain marks the paper paywalled.
    def _pmc_s3_url() -> str | None:
        nonlocal pmcid
        if not pmcid and pmid:
            pmcid = _promote_pmid_to_pmcid(str(pmid), sess)
        return _fetch_pmc_s3_pdf_url(str(pmcid), sess) if pmcid else None

    resolvers: list[tuple[str, Any]] = [
        ("crossref_pdf", lambda: crossref.get("crossref_pdf_url")),
        ("unpaywall_pdf", lambda: _fetch_unpaywall_pdf_url(doi, sess, email=resolved_email)),
        ("pmc_s3_pdf", _pmc_s3_url),
    ]
    for source, resolve in resolvers:
        pdf_url = resolve()
        if not pdf_url:
            continue
        if _download_file(pdf_url, temp_pdf, sess):
            os.replace(str(temp_pdf), str(final_pdf))
            logger.info("[%s] Downloaded PDF → %s (%s)", pdb_id, final_pdf, source)
            entry = {
                "status": DL_STATUS_SUCCESS,
                "source": source,
                "file_path": str(final_pdf),
                "doi": doi,
                "pmid": pmid,
                "pmcid": pmcid,
                "timestamp": now,
            }
            _update_download_log(pdb_id, entry)
            return entry
        with contextlib.suppress(OSError):
            temp_pdf.unlink()

    # Every open-access route exhausted. We deliberately do NOT fall back to an
    # abstract: the annotator needs the full paper, so the PDB is left paywalled
    # for manual download (the watcher ingests a hand-dropped PDF) and otherwise
    # skipped downstream.
    logger.info("[%s] No open-access PDF available, marking paywalled", pdb_id)
    entry = {
        "status": DL_STATUS_PAYWALLED,
        "source": None,
        "file_path": None,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "timestamp": now,
    }
    _update_download_log(pdb_id, entry)
    return entry
