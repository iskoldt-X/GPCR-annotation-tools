"""Tests for papers/downloader.py — multi-tier PDF download logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gpcr_tools.papers.downloader import (
    _fetch_crossref_metadata,
    _fetch_doi_from_title,
    _fetch_pmc_s3_pdf_url,
    _fetch_unpaywall_pdf_url,
    _promote_pmid_to_pmcid,
    _read_download_log,
    _recover_missing_doi,
    _update_download_log,
    download_paper_for_pdb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENRICHED_DATA: dict[str, Any] = {
    "data": {
        "entry": {
            "rcsb_id": "7W55",
            "rcsb_primary_citation": {
                "pdbx_database_id_DOI": "10.1038/s41586-022-04958-8",
            },
            "rcsb_entry_container_identifiers": {"pubmed_id": 12345},
            "pubmed": {"rcsb_pubmed_central_id": "PMC789"},
        }
    }
}


@pytest.fixture()
def papers_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a workspace for papers testing."""
    from gpcr_tools.config import reset_config

    workspace = tmp_path
    monkeypatch.setenv("GPCR_WORKSPACE", str(workspace))
    monkeypatch.setenv("GPCR_EMAIL_FOR_APIS", "test@example.com")
    reset_config()

    # Create enriched data
    enriched_dir = workspace / "enriched"
    enriched_dir.mkdir(parents=True)
    (enriched_dir / "7W55.json").write_text(json.dumps(_ENRICHED_DATA))

    # Create necessary dirs
    (workspace / "papers").mkdir()
    (workspace / "state").mkdir()
    (workspace / "cache").mkdir()

    yield workspace
    reset_config()


# ---------------------------------------------------------------------------
# Download log
# ---------------------------------------------------------------------------


class TestDownloadLog:
    def test_read_empty_log(self, papers_workspace: Path) -> None:
        log = _read_download_log()
        assert log == {}

    def test_write_and_read_log(self, papers_workspace: Path) -> None:
        _update_download_log("7W55", {"status": "success_pdf_downloaded"})
        log = _read_download_log()
        assert "7W55" in log
        assert log["7W55"]["status"] == "success_pdf_downloaded"

    def test_atomic_update_preserves_existing(self, papers_workspace: Path) -> None:
        _update_download_log("7W55", {"status": "success_pdf_downloaded"})
        _update_download_log("8ABC", {"status": "fallback_paywalled"})
        log = _read_download_log()
        assert "7W55" in log
        assert "8ABC" in log


# ---------------------------------------------------------------------------
# Tier API functions
# ---------------------------------------------------------------------------


class TestCrossRefMetadata:
    def test_extracts_pmid(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"PMID": "12345", "link": []}}
        mock_session.get.return_value = mock_resp

        result = _fetch_crossref_metadata("10.1038/test", mock_session)
        assert result["pmid"] == "12345"

    def test_handles_api_failure(self) -> None:
        mock_session = MagicMock()
        import requests

        mock_session.get.side_effect = requests.exceptions.ConnectionError("timeout")

        result = _fetch_crossref_metadata("10.1038/test", mock_session)
        assert result["pmid"] is None

    def test_extracts_direct_pdf_link(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "PMID": "999",
                "link": [
                    {
                        "URL": "https://www.nature.com/articles/x.pdf",
                        "content-type": "application/pdf",
                    }
                ],
            }
        }
        mock_session.get.return_value = mock_resp
        result = _fetch_crossref_metadata("10.1038/x", mock_session)
        assert result["crossref_pdf_url"] == "https://www.nature.com/articles/x.pdf"


class TestPmcS3PdfUrl:
    def test_returns_url_when_metadata_has_pdf(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"pdf_url": "PMC789.1/PMC789.1.pdf"}
        mock_session.get.return_value = mock_resp
        result = _fetch_pmc_s3_pdf_url("PMC789", mock_session)
        assert result == "https://pmc-oa-opendata.s3.amazonaws.com/PMC789.1/PMC789.1.pdf"

    def test_none_when_metadata_has_no_pdf(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"pdf_url": None}
        mock_session.get.return_value = mock_resp
        assert _fetch_pmc_s3_pdf_url("789", mock_session) is None

    def test_none_on_404(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_session.get.return_value = mock_resp
        assert _fetch_pmc_s3_pdf_url("PMC404", mock_session) is None


class TestPromotePmidToPmcid:
    def test_returns_pmcid(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": [{"pmid": "12345", "pmcid": "PMC987"}]}
        mock_session.get.return_value = mock_resp
        with patch("gpcr_tools.papers.downloader.time.sleep"):
            assert _promote_pmid_to_pmcid("12345", mock_session) == "PMC987"

    def test_none_when_no_pmcid(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": [{"pmid": "12345"}]}
        mock_session.get.return_value = mock_resp
        with patch("gpcr_tools.papers.downloader.time.sleep"):
            assert _promote_pmid_to_pmcid("12345", mock_session) is None


class TestFetchDoiFromTitle:
    def _session(self, title: str) -> MagicMock:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"items": [{"DOI": "10.1/found", "title": [title]}]}
        }
        mock_session.get.return_value = mock_resp
        return mock_session

    def test_strong_match_returns_doi(self) -> None:
        title = "Structure of the human serotonin 5-HT2A receptor complex"
        assert _fetch_doi_from_title(title, self._session(title)) == "10.1/found"

    def test_weak_match_returns_none(self) -> None:
        query = "Structure of the human serotonin 5-HT2A receptor complex"
        hit = "Crystal packing of an unrelated bacterial transporter protein"
        assert _fetch_doi_from_title(query, self._session(hit)) is None

    def test_short_title_returns_none(self) -> None:
        # Too few tokens to disambiguate safely -- never queries.
        mock_session = MagicMock()
        assert _fetch_doi_from_title("GPCR study", mock_session) is None
        mock_session.get.assert_not_called()

    def test_companion_paper_rejected(self) -> None:
        # A near-duplicate companion/series paper must NOT be accepted: the query
        # tokens are a subset of the hit's, so symmetric containment fails.
        # Attaching the wrong paper is worse than attaching none.
        query = "Structure of the dopamine D2 receptor"
        hit = "Structure of the dopamine D2 receptor bound to risperidone and a nanobody"
        assert _fetch_doi_from_title(query, self._session(hit)) is None


class TestRecoverMissingDoi:
    def test_recovers_doi_from_citation_table(self) -> None:
        entry = {"citation": [{"id": "primary", "pdbx_database_id_DOI": "10.2/cit"}]}
        doi, _pmid = _recover_missing_doi(entry, None, MagicMock())
        assert doi == "10.2/cit"

    def test_falls_back_to_title_search(self) -> None:
        entry = {"citation": [{"id": "primary", "title": "A distinctive receptor structure paper"}]}
        with patch(
            "gpcr_tools.papers.downloader._fetch_doi_from_title", return_value="10.3/title"
        ) as m:
            doi, _pmid = _recover_missing_doi(entry, None, MagicMock())
        assert doi == "10.3/title"
        m.assert_called_once()

    def test_no_citation_returns_none(self) -> None:
        doi, pmid = _recover_missing_doi({"citation": []}, 42, MagicMock())
        assert doi is None and pmid == 42


class TestUnpaywallPdfUrl:
    def test_returns_pdf_url(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}
        }
        mock_session.get.return_value = mock_resp

        result = _fetch_unpaywall_pdf_url("10.1038/test", mock_session)
        assert result == "https://example.com/paper.pdf"

    def test_returns_none_when_no_oa(self) -> None:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"best_oa_location": None}
        mock_session.get.return_value = mock_resp

        result = _fetch_unpaywall_pdf_url("10.1038/test", mock_session)
        assert result is None


# ---------------------------------------------------------------------------
# download_paper_for_pdb
# ---------------------------------------------------------------------------


class TestDownloadPaperForPdb:
    def test_skips_if_enriched_missing(self, papers_workspace: Path) -> None:
        result = download_paper_for_pdb("XXXX", email="test@example.com")
        assert result["status"] == "skipped_no_enriched_data"

    def test_skips_if_pdf_exists(self, papers_workspace: Path) -> None:
        (papers_workspace / "papers" / "7W55.pdf").write_text("fake pdf")
        result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "skipped_already_downloaded"

    @patch(
        "gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url",
        return_value="https://example.com/7W55.pdf",
    )
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": "12345", "pmcid": "PMC789"},
    )
    def test_success_downloads_pdf(
        self,
        _mock_cr: MagicMock,
        _mock_up: MagicMock,
        papers_workspace: Path,
    ) -> None:
        # Mock _download_file to actually create the temp file
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.4 fake content")
            return True

        with patch(
            "gpcr_tools.papers.downloader._download_file",
            side_effect=fake_download,
        ):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "unpaywall_pdf"

    @patch(
        "gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url",
        return_value="https://example.com/unpaywall.pdf",
    )
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={
            "pmid": None,
            "pmcid": None,
            "crossref_pdf_url": "https://publisher.example/challenge.pdf",
        },
    )
    def test_chain_continues_when_resolved_url_is_not_a_pdf(
        self, _mock_cr: MagicMock, _mock_up: MagicMock, papers_workspace: Path
    ) -> None:
        # The first candidate (crossref_pdf) resolves to a URL but yields a
        # non-PDF (HTML bot challenge); the chain must NOT stop there -- it falls
        # through to Unpaywall and succeeds. This is the load-bearing behaviour
        # the rewrite exists for.
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            if "challenge" in url:
                return False
            output_path.write_bytes(b"%PDF-1.5 real content")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "unpaywall_pdf"

    @patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None)
    @patch("gpcr_tools.papers.downloader._fetch_pmc_s3_pdf_url", return_value=None)
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": None, "pmcid": None, "crossref_pdf_url": None},
    )
    def test_fallback_paywalled(
        self,
        _mock_cr: MagicMock,
        _mock_pmc: MagicMock,
        _mock_up: MagicMock,
        papers_workspace: Path,
    ) -> None:
        result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "fallback_paywalled"

    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={
            "pmid": None,
            "pmcid": None,
            "crossref_pdf_url": "https://www.nature.com/articles/x.pdf",
        },
    )
    def test_crossref_pdf_link_is_first_choice(
        self, _mock_cr: MagicMock, papers_workspace: Path
    ) -> None:
        # The direct CrossRef PDF link is tried before Unpaywall/PMC and, when it
        # yields a real PDF, wins -- this is the largest DOI-only recovery path.
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.5 nature")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "crossref_pdf"

    @patch(
        "gpcr_tools.papers.downloader._fetch_pmc_s3_pdf_url",
        return_value="https://pmc-oa-opendata.s3.amazonaws.com/PMC789.1/PMC789.1.pdf",
    )
    @patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None)
    @patch(
        "gpcr_tools.papers.downloader._fetch_crossref_metadata",
        return_value={"pmid": "12345", "pmcid": "PMC789", "crossref_pdf_url": None},
    )
    def test_falls_through_to_pmc_s3(
        self,
        _mock_cr: MagicMock,
        _mock_up: MagicMock,
        _mock_s3: MagicMock,
        papers_workspace: Path,
    ) -> None:
        # crossref-pdf absent, Unpaywall empty -> the chain reaches the PMC S3 tier.
        def fake_download(url: str, output_path: Path, session: object) -> bool:
            output_path.write_bytes(b"%PDF-1.7 pmc")
            return True

        with patch("gpcr_tools.papers.downloader._download_file", side_effect=fake_download):
            result = download_paper_for_pdb("7W55", email="test@example.com")
        assert result["status"] == "success_pdf_downloaded"
        assert result["source"] == "pmc_s3_pdf"

    def test_no_doi_recovered_from_citation_table(self, papers_workspace: Path) -> None:
        # No DOI on the primary citation, but the citation[] table carries one:
        # it must be recovered rather than giving up as failed_no_doi.
        enriched = {
            "data": {
                "entry": {
                    "rcsb_id": "9NOD",
                    "rcsb_primary_citation": {},
                    "citation": [{"id": "primary", "pdbx_database_id_DOI": "10.1038/recovered"}],
                }
            }
        }
        (papers_workspace / "enriched" / "9NOD.json").write_text(json.dumps(enriched))
        with (
            patch(
                "gpcr_tools.papers.downloader._fetch_crossref_metadata",
                return_value={"pmid": None, "pmcid": None, "crossref_pdf_url": None},
            ),
            patch("gpcr_tools.papers.downloader._fetch_unpaywall_pdf_url", return_value=None),
            patch("gpcr_tools.papers.downloader._fetch_pmc_s3_pdf_url", return_value=None),
        ):
            result = download_paper_for_pdb("9NOD", email="test@example.com")
        # DOI was recovered, so it proceeds past failed_no_doi to paywalled.
        assert result["status"] == "fallback_paywalled"
        assert result["doi"] == "10.1038/recovered"

    def test_genuinely_no_doi_stays_failed_no_doi(self, papers_workspace: Path) -> None:
        enriched = {
            "data": {"entry": {"rcsb_id": "9XUP", "rcsb_primary_citation": {}, "citation": []}}
        }
        (papers_workspace / "enriched" / "9XUP.json").write_text(json.dumps(enriched))
        result = download_paper_for_pdb("9XUP", email="test@example.com")
        assert result["status"] == "failed_no_doi"


class TestRunFetchPapersIsolation:
    def test_per_pdb_error_does_not_abort_batch(self, papers_workspace: Path) -> None:
        # One PDB raising an unexpected error must not abort the batch nor escape
        # the log -- it is recorded as failed_no_data and the run continues.
        from gpcr_tools.papers.runner import run_fetch_papers

        with patch(
            "gpcr_tools.papers.runner.download_paper_for_pdb",
            side_effect=RuntimeError("boom"),
        ):
            run_fetch_papers(pdb_id="7W55", auto_only=True)  # must not raise
        log = _read_download_log()
        assert log["7W55"]["status"] == "failed_no_data"
