"""Live calibration test for the site_ref detector against real data.

Fetches 9IIX (TAS2R14 with the agonist A1AEI modelled at two distinct sites)
and confirms the production pipeline -- coordinates + RCSB alignment + the shipped
generic-numbering table -- resolves the two sites (orthosteric + a second site).
Gated on the network so the unit suite stays offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpcr_tools.detector.signals import SIGNAL_SITE_REF
from gpcr_tools.detector.site_ref import detect_site_refs

pytestmark = pytest.mark.skipif(
    not os.environ.get("GPCR_RUN_LIVE_TESTS"),
    reason="Live network tests disabled; set GPCR_RUN_LIVE_TESTS=1 to enable",
)

# Minimal enriched fields the detector reads.
_ENTRY_9IIX = {
    "polymer_entities": [
        {
            "uniprots": [{"gpcrdb_entry_name_slug": "t2r14_human", "rcsb_id": "Q9NYV8"}],
            "polymer_entity_instances": [
                {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": "R"}}
            ],
        }
    ],
    "nonpolymer_entities": [
        {
            "rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": "A1AEI"},
            "nonpolymer_entity_instances": [
                {"rcsb_nonpolymer_entity_instance_container_identifiers": {"asym_id": "B"}}
            ],
        }
    ],
}


def test_9iix_site_ref_resolves_two_sites(tmp_path: Path) -> None:
    signals = detect_site_refs("9IIX", _ENTRY_9IIX, tmp_path)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.kind == SIGNAL_SITE_REF
    assert signal.payload["comp_id"] == "A1AEI"
    sites = signal.payload["sites"]
    # A1AEI binds the deep orthosteric pocket and the upper extracellular vestibule.
    assert "orthosteric" in sites
    assert "extracellular_vestibule" in sites
