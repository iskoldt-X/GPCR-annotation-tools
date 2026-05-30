"""Contract test: every ligand role the annotator can emit must resolve to a
role the GPCRdb already uses.

The downstream build normalizes each ligand role with Django's ``slugify()`` and
``get_or_create()``s a role by that slug.  So a role string that slugifies to a
slug the database does not already use would silently create a brand-new, unused
role -- invisible vocabulary drift.  This test pins the role vocabulary to the
set of strings verified (against the live role table) to slugify onto used
roles, so any future edit to the enum that breaks the mapping fails loudly.

The enum naming is deliberate: "Agonist (partial)" slugifies to "agonist-partial"
(a used role), NOT "partial-agonist" (unused); "PAM" / "NAM" stay abbreviated
rather than spelling out to unused slugs.
"""

from __future__ import annotations

import re
from typing import Any

from gpcr_tools.annotator.schema import ANNOTATION_TOOL

# Each emittable ligand role -> the role slug it must resolve to. Every slug
# here was confirmed to be a role the database actually uses.
ROLE_TO_USED_SLUG = {
    "Agonist": "agonist",
    "Antagonist": "antagonist",
    "Apo (no ligand)": "apo-no-ligand",
    "PAM": "pam",
    "NAM": "nam",
    "Ago-PAM": "ago-pam",
    "Allosteric agonist": "allosteric-agonist",
    "Allosteric antagonist": "allosteric-antagonist",
    "Inverse agonist": "inverse-agonist",
    "Agonist (partial)": "agonist-partial",
    "Cofactor": "cofactor",
    "unknown": "unknown",
}

# Spelled-out / reordered forms that look natural but are NOT used downstream;
# the enum must never produce one of these.
UNUSED_LOOKALIKE_SLUGS = {
    "partial-agonist",
    "full-agonist",
    "positive-allosteric-modulator",
    "negative-allosteric-modulator",
}


def _slugify(value: str) -> str:
    """ASCII-faithful reimplementation of ``django.utils.text.slugify``.

    The downstream build slugifies the role string with Django; the roles we
    emit are plain ASCII (letters, spaces, parentheses, hyphens), for which this
    matches Django exactly: drop everything but word chars, whitespace and
    hyphens, then collapse runs of whitespace/hyphens to a single hyphen and
    trim.
    """
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def _extract_role_enum() -> list[str]:
    """Pull the ligand-role enum out of the annotation tool schema.

    Located by content (the enum that lists both Agonist and Antagonist) so the
    test survives restructuring of the schema object.
    """
    found: list[list[str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            if all(isinstance(x, str) for x in node) and {"Agonist", "Antagonist"} <= set(node):
                found.append(node)
            for value in node:
                walk(value)

    walk(ANNOTATION_TOOL.model_dump())
    assert len(found) == 1, f"expected exactly one ligand-role enum, found {len(found)}"
    return found[0]


class TestLigandRoleVocabularyContract:
    def test_enum_matches_pinned_vocabulary(self) -> None:
        # If this fails the role enum drifted: re-verify every new value against
        # the live role table before updating ROLE_TO_USED_SLUG.
        assert set(_extract_role_enum()) == set(ROLE_TO_USED_SLUG)

    def test_every_role_slugifies_to_a_used_slug(self) -> None:
        for role, expected_slug in ROLE_TO_USED_SLUG.items():
            assert _slugify(role) == expected_slug

    def test_no_role_slugifies_to_an_unused_lookalike(self) -> None:
        produced = {_slugify(role) for role in _extract_role_enum()}
        assert produced.isdisjoint(UNUSED_LOOKALIKE_SLUGS)
