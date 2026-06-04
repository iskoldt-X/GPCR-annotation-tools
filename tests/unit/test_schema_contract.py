"""Contract tests binding the Gemini tool schema to the config constants.

The schema's enums are hand-written literals; if one drifts from the config
constant the rest of the pipeline validates against, the model and the
validators silently disagree. These tests fail loudly on any such drift.
"""

from __future__ import annotations

from gpcr_tools.annotator.schema import ANNOTATION_TOOL
from gpcr_tools.config import SITE_REF_VALUES


def _ligand_item_properties() -> dict:
    """The per-ligand item schema's properties, from the live tool object."""
    params = ANNOTATION_TOOL.function_declarations[0].parameters
    return params.properties["ligands"].items.properties


def test_site_ref_enum_matches_config() -> None:
    # The schema's site_ref enum is the model-facing list; SITE_REF_VALUES is
    # what every downstream validator / grouping step uses. They must agree, or
    # the model can emit a value the pipeline does not recognise (and vice
    # versa). Compared as sets: an added or removed value is the real drift.
    schema_enum = _ligand_item_properties()["site_ref"].enum
    assert schema_enum is not None
    assert len(schema_enum) == len(set(schema_enum))  # no duplicate members
    assert set(schema_enum) == set(SITE_REF_VALUES)
