"""Tests for G-protein alpha5 identity analysis.

Covers: is_g_alpha_description, calculate_match_score, and get_chimera_analysis
(resolved subtype, inseparable-set family-only, cross-family-only, low
confidence, sliding rescue of a displaced alpha5, and the no-G-protein /
too-short / no-comparison paths).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from gpcr_tools.config import (
    CHIMERA_A5_WINDOW,
    CHIMERA_STATUS_NO_G_PROTEIN,
    CHIMERA_STATUS_NO_VALID_COMPARISONS,
    CHIMERA_STATUS_SUCCESS,
    CHIMERA_STATUS_TOO_SHORT,
    CHIMERA_SUBTYPE_FAMILY_ONLY,
    CHIMERA_SUBTYPE_INSEPARABLE_SET,
    CHIMERA_SUBTYPE_LOW_CONFIDENCE,
    CHIMERA_SUBTYPE_RESOLVED,
    FULL_G_ALPHA_CANDIDATES,
)
from gpcr_tools.validator.cache import SequenceCache
from gpcr_tools.validator.chimera import (
    calculate_match_score,
    get_chimera_analysis,
    is_g_alpha_description,
)

# ===================================================================
# is_g_alpha_description
# ===================================================================


class TestIsGAlphaDescription:
    def test_standard_galpha(self) -> None:
        assert is_g_alpha_description("G alpha subunit") is True

    def test_g_dash_alpha(self) -> None:
        assert is_g_alpha_description("G-alpha protein") is True

    def test_g_protein_alpha(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein alpha subunit") is True

    def test_gq_family(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein Gq") is True

    def test_gs_family(self) -> None:
        assert is_g_alpha_description("G protein Gs subunit") is True

    def test_minig(self) -> None:
        assert is_g_alpha_description("miniGsq fusion") is True

    def test_engineered_g13(self) -> None:
        assert is_g_alpha_description("Engineered G13") is True

    def test_guanine_terminal_pattern(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein G(q)") is True

    def test_fusion_catch(self) -> None:
        assert is_g_alpha_description("Guanine nucleotide-binding protein subunit alpha") is True

    # Negative cases
    def test_receptor_excluded(self) -> None:
        assert is_g_alpha_description("Dopamine receptor D2") is False

    def test_antibody_excluded(self) -> None:
        assert is_g_alpha_description("Nanobody Nb35") is False

    def test_beta_subunit_excluded(self) -> None:
        assert is_g_alpha_description("G protein beta subunit") is False

    def test_gamma_subunit_excluded(self) -> None:
        assert is_g_alpha_description("G protein gamma subunit") is False

    def test_random_protein(self) -> None:
        assert is_g_alpha_description("Ubiquitin ligase") is False

    def test_empty_string(self) -> None:
        assert is_g_alpha_description("") is False

    def test_alpha_overrides_exclude(self) -> None:
        """If 'alpha' is present, exclude keywords are bypassed."""
        assert is_g_alpha_description("G-alpha receptor fusion") is True


# ===================================================================
# calculate_match_score
# ===================================================================


class TestCalculateMatchScore:
    def test_exact_match(self) -> None:
        assert calculate_match_score("ACDE", "ACDE") == 4

    def test_partial_match(self) -> None:
        assert calculate_match_score("ACDE", "ACDF") == 3

    def test_no_match(self) -> None:
        assert calculate_match_score("AAAA", "BBBB") == 0

    def test_empty_first(self) -> None:
        assert calculate_match_score("", "ACDE") == 0

    def test_empty_second(self) -> None:
        assert calculate_match_score("ACDE", "") == 0

    def test_different_lengths(self) -> None:
        assert calculate_match_score("AC", "ACDE") == 0

    def test_both_empty(self) -> None:
        assert calculate_match_score("", "") == 0

    def test_single_char_match(self) -> None:
        assert calculate_match_score("A", "A") == 1

    def test_single_char_mismatch(self) -> None:
        assert calculate_match_score("A", "B") == 0


# ===================================================================
# get_chimera_analysis
# ===================================================================

# The real transducin / Gi alpha5 tip. Its last residues (CGLF) are shared
# across Gi and the transducins, which is why a short window cannot tell them
# apart; the full 11-mer separates the transducin subgroup from Gi but still
# cannot pick one transducin over another.
TRANSDUCIN_A5 = "IKENLKDCGLF"
DISTINCT_A5 = "WWWWWWWWWWW"  # matches no real alpha5; used as the default ref tail

assert len(TRANSDUCIN_A5) == CHIMERA_A5_WINDOW
assert len(DISTINCT_A5) == CHIMERA_A5_WINDOW


def _make_enriched(
    *,
    desc: str = "G alpha subunit",
    sequence: str = "MMMMMMMMMM" + TRANSDUCIN_A5,
    uniprots: list[dict[str, Any]] | None = None,
    has_galpha: bool = True,
) -> dict[str, Any]:
    """Build an enriched_entry with a single polymer entity."""
    if not has_galpha:
        return {"polymer_entities": []}

    entity: dict[str, Any] = {
        "rcsb_polymer_entity": {"pdbx_description": desc},
        "entity_poly": {"pdbx_seq_one_letter_code_can": sequence},
    }
    if uniprots is not None:
        entity["uniprots"] = uniprots
    return {"polymer_entities": [entity]}


def _mock_refs(tail_by_slug: dict[str, str], default_tail: str = DISTINCT_A5) -> Any:
    """Return a get_sequence_from_uniprot stand-in.

    Each accession maps (via FULL_G_ALPHA_CANDIDATES) to a slug; the returned
    sequence ends with that slug's assigned alpha5 window, padded so it exceeds
    the window length.
    """

    def _fetch(accession: str, cache: Any) -> str | None:
        slug = FULL_G_ALPHA_CANDIDATES.get(accession)
        tail = tail_by_slug.get(slug, default_tail) if slug else default_tail
        return "GGGGG" + tail

    return _fetch


class TestGetChimeraAnalysis:
    def test_resolved_subtype(self, tmp_path: Path) -> None:
        """A unique best match resolves to that subtype and its family."""
        cache = SequenceCache(tmp_path / "seq.json")
        target = "ACDEFGHIKLM"
        enriched = _make_enriched(sequence="MMMMMMMMMM" + target)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnas2_human": target}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["score"] == CHIMERA_A5_WINDOW
        assert result["subtype"] == "gnas2_human"
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_RESOLVED
        assert result["family"] == "Gs"
        assert result["family_confident"] is True
        assert result["candidate_set"] == ["gnas2_human"]
        assert result["a5_tail"] == target

    def test_transducin_alpha5_resolves_family_not_subtype(self, tmp_path: Path) -> None:
        """The transducin alpha5 is shared by gnat1/2/3.

        The family must come back as Gi/o, the subtype must NOT be forced to one
        member, and the indistinguishable set is reported for human review.
        """
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        tails = {
            "gnat1_human": TRANSDUCIN_A5,
            "gnat2_human": TRANSDUCIN_A5,
            "gnat3_human": TRANSDUCIN_A5,
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("9IIX", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["family"] == "Gi/o"
        assert result["family_confident"] is True
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_INSEPARABLE_SET
        assert result["candidate_set"] == ["gnat1_human", "gnat2_human", "gnat3_human"]

    def test_cross_member_tie_is_family_only(self, tmp_path: Path) -> None:
        """A tie across same-family members that are not a defined inseparable
        set stops at the family with a family_only resolution."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        tails = {"gnai1_human": TRANSDUCIN_A5, "gnai3_human": TRANSDUCIN_A5}
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs(tails),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["family"] == "Gi/o"
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_FAMILY_ONLY
        assert result["candidate_set"] == ["gnai1_human", "gnai3_human"]

    def test_low_confidence_when_nothing_matches(self, tmp_path: Path) -> None:
        """A weak best score is reported as low confidence, not a forced call."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="A" * 25)  # matches no real alpha5
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({}),  # every ref gets DISTINCT_A5 (no 'A')
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["subtype"] is None
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_LOW_CONFIDENCE

    def test_sliding_rescues_displaced_alpha5(self, tmp_path: Path) -> None:
        """When the alpha5 is not at the C-terminus (fusion/extra residues), the
        sliding scan still locates it and resolves the subtype."""
        cache = SequenceCache(tmp_path / "seq.json")
        # alpha5 sits in the middle, followed by a non-matching C-terminal tag.
        enriched = _make_enriched(sequence="GGGGG" + TRANSDUCIN_A5 + "PPPPPPPPPP")
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            side_effect=_mock_refs({"gnao_human": TRANSDUCIN_A5}),
        ):
            result = get_chimera_analysis("TEST", enriched, cache)

        assert result["status"] == CHIMERA_STATUS_SUCCESS
        assert result["subtype"] == "gnao_human"
        assert result["subtype_resolution"] == CHIMERA_SUBTYPE_RESOLVED
        assert result["a5_tail"] == TRANSDUCIN_A5

    def test_no_g_protein(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(has_galpha=False)
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_sequence_too_short(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="AB")  # shorter than the window
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_TOO_SHORT

    def test_no_valid_comparisons(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        enriched = _make_enriched(sequence="MMMMMMMMMM" + TRANSDUCIN_A5)
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            return_value=None,
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS

    def test_null_pdbx_description(self, tmp_path: Path) -> None:
        """A null pdbx_description must not crash."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": None},
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MDEF"},
                }
            ]
        }
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_null_entity_poly(self, tmp_path: Path) -> None:
        """A null entity_poly must not crash."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "G alpha subunit"},
                    "entity_poly": None,
                }
            ]
        }
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_TOO_SHORT

    def test_null_rcsb_polymer_entity(self, tmp_path: Path) -> None:
        """A null rcsb_polymer_entity must not crash."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": None,
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "MDEF"},
                }
            ]
        }
        result = get_chimera_analysis("TEST", enriched, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_empty_enriched(self, tmp_path: Path) -> None:
        cache = SequenceCache(tmp_path / "seq.json")
        result = get_chimera_analysis("TEST", {}, cache)
        assert result["status"] == CHIMERA_STATUS_NO_G_PROTEIN

    def test_fallback_to_pdbx_seq(self, tmp_path: Path) -> None:
        """Falls back to pdbx_seq_one_letter_code when _can is missing."""
        cache = SequenceCache(tmp_path / "seq.json")
        enriched: dict[str, Any] = {
            "polymer_entities": [
                {
                    "rcsb_polymer_entity": {"pdbx_description": "G alpha subunit"},
                    "entity_poly": {"pdbx_seq_one_letter_code": "MDEFGHIJKLM"},
                }
            ]
        }
        with patch(
            "gpcr_tools.validator.chimera.get_sequence_from_uniprot",
            return_value=None,
        ):
            result = get_chimera_analysis("TEST", enriched, cache)
        # The sequence was found via the fallback field, but no references
        # could be fetched, so no comparison was possible.
        assert result["status"] == CHIMERA_STATUS_NO_VALID_COMPARISONS

    def test_result_keys(self, tmp_path: Path) -> None:
        """Every documented key is present even on the early-exit paths."""
        cache = SequenceCache(tmp_path / "seq.json")
        result = get_chimera_analysis("TEST", {}, cache)
        for key in (
            "status",
            "family",
            "family_confident",
            "subtype",
            "subtype_resolution",
            "candidate_set",
            "score",
            "a5_window",
            "a5_tail",
            "candidates_checked",
            "error",
        ):
            assert key in result
