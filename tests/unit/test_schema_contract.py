"""Contract tests binding the Gemini tool schema to the config constants.

The schema's enums are hand-written literals; if one drifts from the config
constant the rest of the pipeline validates against, the model and the
validators silently disagree. These tests fail loudly on any such drift.
"""

from __future__ import annotations

from google.genai import types

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

    # The structure-state enum (incl. its 'unknown' escape) is pinned to the
    # downstream-accepted CSV tokens by test_state_vocab_contract.


def test_site_ref_justification_is_optional_string() -> None:
    # The justification is a curator-facing free-text field on each ligand: it must
    # exist with type string and must NOT be required (the model may omit it).
    props = _ligand_item_properties()
    assert "site_ref_justification" in props
    assert props["site_ref_justification"].type == types.Type.STRING
    required = (
        ANNOTATION_TOOL.function_declarations[0].parameters.properties["ligands"].items.required
    )
    assert "site_ref_justification" not in (required or [])


def test_role_site_rule_sets_match_schema_role_enum() -> None:
    # The ligand_validator role/site safety net keys on exact role.value strings;
    # a misspelling would silently never fire. Bind its role sets to the schema enum.
    from gpcr_tools.validator.ligand_validator import (
        _ALLOSTERIC_ROLES,
        _FUNCTIONAL_POCKET_ROLES,
    )

    role_enum = set(_ligand_item_properties()["role"].properties["value"].enum)
    assert _ALLOSTERIC_ROLES.issubset(role_enum)
    assert _FUNCTIONAL_POCKET_ROLES.issubset(role_enum)
    assert "Cofactor" in role_enum
