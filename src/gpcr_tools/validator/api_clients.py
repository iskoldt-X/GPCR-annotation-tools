"""API client wrappers for UniProt, PubChem, and RCSB GraphQL.

Network error handling: returns ``None`` (NOT ``True``) on
timeout/connection failure.  Callers translate ``None`` into an
``[API_UNAVAILABLE]`` warning.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from gpcr_tools.config import (
    API_MAX_RETRIES,
    PUBCHEM_REST_URL,
    RCSB_GRAPHQL_URL,
    SLEEP_VALIDATION_RETRY,
    TIMEOUT_PUBCHEM_VALIDATION,
    TIMEOUT_RCSB_GRAPHQL_VALIDATION,
    TIMEOUT_UNIPROT_VALIDATION,
    UNIPROT_REST_URL,
)
from gpcr_tools.validator.cache import ValidationCache

logger = logging.getLogger(__name__)


def check_uniprot_existence(
    entry_name: str,
    cache: ValidationCache,
) -> bool | None:
    """Validate whether a UniProt entry name exists.

    Returns ``True``/``False`` on success, ``None`` on network error.
    """
    clean_name = entry_name.split(".")[0].upper()
    key = f"uniprot:{clean_name.lower()}"

    cached = cache.get(key)
    if cached is not None:
        return cached

    url = f"{UNIPROT_REST_URL}/{clean_name}.txt"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.head(url, timeout=TIMEOUT_UNIPROT_VALIDATION, allow_redirects=True)
            is_valid = bool(resp.status_code == 200)
            cache.set(key, is_valid)
            return is_valid
        except (requests.RequestException, OSError) as exc:
            if attempt == API_MAX_RETRIES - 1:
                logger.warning("UniProt API error for '%s': %s", entry_name, exc)
                return None
            time.sleep(SLEEP_VALIDATION_RETRY)
    return None


def check_pubchem_existence(
    cid: str,
    cache: ValidationCache,
) -> bool | None:
    """Validate whether a PubChem CID exists.

    Returns ``True``/``False`` on success, ``None`` on network error.
    """
    clean_cid = "".join(filter(str.isdigit, str(cid)))
    if not clean_cid:
        return False  # Format error (non-numeric)

    key = f"pubchem:{clean_cid}"

    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        url = f"{PUBCHEM_REST_URL}/cid/{clean_cid}/description/JSON"
        resp = requests.get(url, timeout=TIMEOUT_PUBCHEM_VALIDATION)
        is_valid = bool(resp.status_code == 200)
        cache.set(key, is_valid)
        return is_valid
    except (requests.RequestException, OSError) as exc:
        logger.warning("PubChem API error for '%s': %s", cid, exc)
        return None


# ---------------------------------------------------------------------------
# RCSB GraphQL
# ---------------------------------------------------------------------------

_GRAPHQL_URL = RCSB_GRAPHQL_URL

GRAPHQL_POLYMER_FEATURES_QUERY: str = """\
query structure($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        uniprot_ids
      }
      rcsb_polymer_entity_align {
        reference_database_name
        reference_database_accession
        aligned_regions {
          entity_beg_seq_id
          ref_beg_seq_id
          length
        }
      }
      uniprots {
        rcsb_id
        rcsb_uniprot_feature {
          type
          name
          description
          feature_positions {
            beg_seq_id
            end_seq_id
          }
        }
      }
      rcsb_polymer_entity_feature {
        type
        name
        reference_scheme
        feature_positions {
          beg_seq_id
          end_seq_id
        }
      }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers {
          auth_asym_id
        }
        rcsb_polymer_instance_feature {
          type
          name
          feature_positions {
            beg_seq_id
            end_seq_id
          }
        }
      }
    }
  }
}
"""


def fetch_polymer_features(pdb_id: str) -> dict[str, Any] | None:
    """Fetch polymer entity/instance data from RCSB GraphQL.

    Returns the ``entry`` dict, or ``None`` on error.

    None-safety:
        Uses ``(data.get("data") or {}).get("entry")`` to handle
        ``{"data": null}`` responses.
    """
    payload = {
        "query": GRAPHQL_POLYMER_FEATURES_QUERY,
        "variables": {"id": pdb_id.upper()},
    }
    try:
        resp = requests.post(_GRAPHQL_URL, json=payload, timeout=TIMEOUT_RCSB_GRAPHQL_VALIDATION)
        if resp.status_code != 200:
            logger.warning("[%s] GraphQL returned status %d", pdb_id, resp.status_code)
            return None
        data = resp.json()
        if data.get("errors"):
            logger.warning("[%s] GraphQL returned errors: %s", pdb_id, data["errors"])
            return None
        # None-safe: (data.get("data") or {}).get("entry")
        return (data.get("data") or {}).get("entry")
    except (requests.RequestException, OSError, ValueError) as exc:
        logger.warning("[%s] GraphQL fetch error: %s", pdb_id, exc)
        return None


GRAPHQL_POLYMER_ALIGNMENT_QUERY: str = """\
query structure($id: String!) {
  entry(entry_id: $id) {
    polymer_entities {
      rcsb_polymer_entity_align {
        reference_database_name
        reference_database_accession
        aligned_regions { entity_beg_seq_id ref_beg_seq_id length }
      }
      polymer_entity_instances {
        rcsb_polymer_entity_instance_container_identifiers { auth_asym_id }
      }
    }
  }
}
"""


def fetch_polymer_alignment(
    pdb_id: str,
) -> dict[str, dict[str, list[tuple[int, int, int]]]] | None:
    """Fetch the RCSB SIFTS-derived entity->UniProt alignment, keyed by author chain.

    Returns ``{auth_chain: {uniprot_accession: [(entity_beg, ref_beg, length), ...]}}``
    -- multi-region per accession, so a fusion chain maps each segment to its own
    reference. ``None`` on network / parse failure.
    """
    payload = {"query": GRAPHQL_POLYMER_ALIGNMENT_QUERY, "variables": {"id": pdb_id.upper()}}
    try:
        resp = requests.post(
            RCSB_GRAPHQL_URL, json=payload, timeout=TIMEOUT_RCSB_GRAPHQL_VALIDATION
        )
        if resp.status_code != 200:
            logger.warning("[%s] alignment GraphQL status %d", pdb_id, resp.status_code)
            return None
        data = resp.json()
        if data.get("errors"):
            logger.warning("[%s] alignment GraphQL errors: %s", pdb_id, data["errors"])
            return None
    except (requests.RequestException, OSError, ValueError) as exc:
        logger.warning("[%s] alignment GraphQL fetch error: %s", pdb_id, exc)
        return None

    entry = (data.get("data") or {}).get("entry") or {}
    chains: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for entity in entry.get("polymer_entities") or []:
        if not isinstance(entity, dict):
            continue
        regions: dict[str, list[tuple[int, int, int]]] = {}
        for align in entity.get("rcsb_polymer_entity_align") or []:
            if not isinstance(align, dict) or align.get("reference_database_name") != "UniProt":
                continue
            acc = align.get("reference_database_accession")
            if not acc:
                continue
            for region in align.get("aligned_regions") or []:
                beg = region.get("entity_beg_seq_id")
                ref = region.get("ref_beg_seq_id")
                length = region.get("length")
                if beg is not None and ref is not None and length is not None:
                    regions.setdefault(acc, []).append((int(beg), int(ref), int(length)))
        if not regions:
            continue
        for inst in entity.get("polymer_entity_instances") or []:
            if not isinstance(inst, dict):
                continue
            auth = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
                "auth_asym_id"
            )
            if auth:
                chains[auth] = regions
    return chains
