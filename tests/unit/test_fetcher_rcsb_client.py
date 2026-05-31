"""Tests for fetcher/rcsb_client.py — raw RCSB download with mocked HTTP."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


def _mock_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
@patch("gpcr_tools.fetcher.rcsb_client.requests.post")
def test_nonexistent_pdb_is_not_written(mock_post, _sleep, configure_paths) -> None:
    """RCSB answers an unknown PDB with HTTP 200 and a null entry. That must be
    treated as 'not found' — no hollow file, no success — rather than a record
    that masquerades as a real structure downstream."""
    from gpcr_tools.config import get_config
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    mock_post.return_value = _mock_response({"data": {"entry": None}})

    result = fetch_single_pdb("ZZZZ")

    assert result is None
    raw_path = get_config().raw_pdb_json_dir / "ZZZZ.json"
    assert not raw_path.exists()


@patch("gpcr_tools.fetcher.rcsb_client.time.sleep")
@patch("gpcr_tools.fetcher.rcsb_client.requests.post")
def test_valid_pdb_is_written(mock_post, _sleep, configure_paths) -> None:
    """A real entry is written verbatim and returned (guard must not reject it)."""
    from gpcr_tools.config import get_config
    from gpcr_tools.fetcher.rcsb_client import fetch_single_pdb

    payload = {"data": {"entry": {"rcsb_id": "7W55"}}}
    mock_post.return_value = _mock_response(payload)

    result = fetch_single_pdb("7W55")

    assert result == payload
    raw_path = get_config().raw_pdb_json_dir / "7W55.json"
    assert raw_path.exists()
