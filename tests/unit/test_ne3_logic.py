"""Tests for New Era Epic 3: Scientific Transformation Layer (logic.py).

Covers all 4 pure functions: map_label_asym_id, collect_ligand_chains,
apply_db_truncation, build_structure_note.
"""

from gpcr_tools.config import ALERT_NON_RECEPTOR_PARTNER
from gpcr_tools.csv_generator.logic import (
    apply_db_truncation,
    build_structure_note,
    collect_ligand_chains,
    map_label_asym_id,
    resolve_partner_protomer,
)


def _oligo(*chains: tuple[str, str]) -> dict:
    """all_gpcr_chains from (chain_id, slug) pairs."""
    return {"all_gpcr_chains": [{"chain_id": c, "slug": s} for c, s in chains]}


def _oligo_tm(
    *chains: tuple[str, str, int],
    tm_data_available: bool = True,
) -> dict:
    """all_gpcr_chains from (chain_id, slug, total_tms) triples, with the TM flag.

    Mirrors the analysis dict resolve_partner_protomer reads: each chain carries a
    ``total_tms`` annotation (the count the partner-column gate checks), and the
    top-level ``tm_data_available`` flag governs whether the gate runs at all.
    """
    return {
        "all_gpcr_chains": [{"chain_id": c, "slug": s, "total_tms": tm} for c, s, tm in chains],
        "tm_data_available": tm_data_available,
    }


class TestResolvePartnerProtomer:
    def test_heterodimer_records_partner_gene(self) -> None:
        # GABA-B: primary GABBR2 (B); partner GABBR1 (A) must be recorded, not lost.
        oligo = _oligo(("A", "gabr1_human"), ("B", "gabr2_human"))
        assert resolve_partner_protomer(oligo, "B") == ("gabr1_human", "A")

    def test_homodimer_records_other_chain(self) -> None:
        oligo = _oligo(("A", "grm2_human"), ("B", "grm2_human"))
        assert resolve_partner_protomer(oligo, "A") == ("grm2_human", "B")

    def test_monomer_has_no_partner(self) -> None:
        oligo = _oligo(("A", "casr_human"))
        assert resolve_partner_protomer(oligo, "A") == ("", "")

    def test_single_chain_primary_still_surfaces_partner(self) -> None:
        # AI reported only the (wrong) GABBR1 chain A; the partner GABBR2 (B) is
        # still recorded even though the primary stays A (single-chain override deferred).
        oligo = _oligo(("A", "gabr1_human"), ("B", "gabr2_human"))
        assert resolve_partner_protomer(oligo, "A") == ("gabr2_human", "B")

    def test_higher_order_joins_extra_chains(self) -> None:
        oligo = _oligo(
            ("A", "gp156_human"), ("B", "gp156_human"), ("C", "gp156_human"), ("D", "gp156_human")
        )
        partner_uniprot, partner_chains = resolve_partner_protomer(oligo, "A")
        assert partner_uniprot == "gp156_human"
        assert partner_chains == "B, C, D"

    def test_empty_all_gpcr_chains(self) -> None:
        assert resolve_partner_protomer({}, "A") == ("", "")

    def test_empty_primary_records_no_partner(self) -> None:
        # No known primary chain (malformed receptor_info) -> no partner attribution,
        # rather than treating every chain as a partner.
        oligo = _oligo(("A", "gabr1_human"), ("B", "gabr2_human"))
        assert resolve_partner_protomer(oligo, "") == ("", "")

    def test_genuine_second_receptor_kept(self) -> None:
        # A real GABA-B heterodimer (GBR2 primary B, GBR1 partner A): both chains
        # are full 7TM receptors, so the partner is retained and no alert fires.
        oligo = _oligo_tm(("A", "gabr1_human", 7), ("B", "gabr2_human", 7))
        assert resolve_partner_protomer(oligo, "B") == ("gabr1_human", "A")
        assert not oligo.get("alerts")

    def test_non_receptor_partner_evicted_with_alert(self) -> None:
        # A 7TM receptor (chain A primary) plus a short peptide ligand (chain B, 0
        # TMs) that carries a receptor-ish slug NOT on the roster denylist -- i.e. an
        # un-catalogued peptide that slipped past the roster build. This TM-count
        # gate is the independent safety net for exactly that case: the peptide must
        # NOT land in the additional-receptor column, and the eviction is surfaced as
        # a curator alert.
        oligo = _oligo_tm(("A", "ednrb_human", 7), ("B", "pep1_human", 0))
        partner_uniprot, partner_chains = resolve_partner_protomer(oligo, "A")
        assert partner_uniprot == ""
        assert partner_chains == ""
        alerts = oligo.get("alerts") or []
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_NON_RECEPTOR_PARTNER
        assert "chain B" in alerts[0]["message"]

    def test_single_pass_coreceptor_evicted(self) -> None:
        # A 7TM receptor (chain A) plus a single-pass (1 TM) co-receptor and a 0-TM
        # soluble agonist, both mis-mapped to a GPCR slug that is NOT on the roster
        # denylist. Neither is a 7TM protomer, so the partner column stays empty and
        # one alert covers both evicted chains.
        oligo = _oligo_tm(
            ("A", "lgr4_human", 7),
            ("C", "znrf3_human", 1),
            ("D", "pep2_human", 0),
        )
        partner_uniprot, partner_chains = resolve_partner_protomer(oligo, "A")
        assert partner_uniprot == ""
        assert partner_chains == ""
        alerts = oligo.get("alerts") or []
        assert len(alerts) == 1
        assert alerts[0]["type"] == ALERT_NON_RECEPTOR_PARTNER
        assert "chain C, D" in alerts[0]["message"]

    def test_fetch_failure_skips_gate_keeps_real_partner(self) -> None:
        # When the TM-feature fetch failed for the whole structure the counts are
        # unverified (all total_tms fall back to 0) but tm_data_available is False:
        # the gate is skipped entirely, so a real GABA-B-style heterodimer keeps its
        # partner rather than being wrongly evicted on zeroed-out counts.
        oligo = _oligo_tm(
            ("A", "gabr1_human", 0),
            ("B", "gabr2_human", 0),
            tm_data_available=False,
        )
        assert resolve_partner_protomer(oligo, "B") == ("gabr1_human", "A")
        assert not oligo.get("alerts")

    def test_chain_missing_total_tms_passes(self) -> None:
        # Legacy / pre-TM analysis dicts have all_gpcr_chains without a total_tms
        # key. Even with the gate active, a chain MISSING the key must pass
        # (backward-compat) so older recorded data is unaffected.
        oligo = {
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "gabr2_human"},
                {"chain_id": "B", "slug": "gabr1_human"},
            ],
            "tm_data_available": True,
        }
        assert resolve_partner_protomer(oligo, "A") == ("gabr1_human", "B")
        assert not oligo.get("alerts")


# ── map_label_asym_id ────────────────────────────────────────────────


class TestMapLabelAsymId:
    def test_identity(self):
        assert map_label_asym_id("A", {"A": "A"}) == "A"

    def test_remap(self):
        assert map_label_asym_id("R", {"R": "E"}) == "E"

    def test_comma_separated(self):
        assert map_label_asym_id("A, B", {"A": "A", "B": "C"}) == "A, C"

    def test_empty_string(self):
        assert map_label_asym_id("", {"A": "B"}) == ""

    def test_missing_key_fallback(self):
        """Keys not in the map fall through unchanged."""
        assert map_label_asym_id("X", {"A": "A"}) == "X"

    def test_empty_map(self):
        assert map_label_asym_id("A", {}) == "A"

    def test_multiple_comma_remap(self):
        label_map = {"A": "X", "B": "Y", "C": "Z"}
        assert map_label_asym_id("A, B, C", label_map) == "X, Y, Z"


# ── collect_ligand_chains ────────────────────────────────────────────


class TestCollectLigandChains:
    def test_basic(self):
        ligands = [{"chain_id": "A"}, {"chain_id": "B"}]
        assert collect_ligand_chains(ligands) == {"A", "B"}

    def test_skip_null_sentinels(self):
        ligands = [
            {"chain_id": "A"},
            {"chain_id": "None"},
            {"chain_id": "null"},
            {"chain_id": "n/a"},
        ]
        assert collect_ligand_chains(ligands) == {"A"}

    def test_comma_separated(self):
        ligands = [{"chain_id": "A, B"}]
        assert collect_ligand_chains(ligands) == {"A", "B"}

    def test_empty_chain_id(self):
        ligands = [{"chain_id": ""}]
        assert collect_ligand_chains(ligands) == set()

    def test_missing_chain_id(self):
        ligands = [{"name": "ligand without chain"}]
        assert collect_ligand_chains(ligands) == set()

    def test_deduplication(self):
        ligands = [{"chain_id": "A"}, {"chain_id": "A"}]
        assert collect_ligand_chains(ligands) == {"A"}

    def test_empty_list(self):
        assert collect_ligand_chains([]) == set()


# ── apply_db_truncation ─────────────────────────────────────────────


class TestApplyDbTruncation:
    def test_single_chain_no_truncation(self):
        chain, uniprot, note = apply_db_truncation("A", "aa2ar_human", {}, set())
        assert chain == "A"
        assert uniprot == "aa2ar_human"
        assert note == ""

    def test_multi_chain_with_suggestion(self):
        oligo = {
            "primary_protomer_suggestion": {
                "chain_id": "A",
                "reason": "G protein bound",
            },
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "aa2ar_human"},
                {"chain_id": "B", "slug": "aa2ar_human"},
            ],
        }
        chain, uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, set())
        assert chain == "A"
        assert uniprot == "aa2ar_human"
        assert "[DB TRUNCATION:" in note
        assert "primary chain A" in note

    def test_preserves_uniprot_from_chain_info(self):
        oligo = {
            "primary_protomer_suggestion": {"chain_id": "B", "reason": "test"},
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "drd2_human"},
                {"chain_id": "B", "slug": "oprm_human"},
            ],
        }
        chain, uniprot, _note = apply_db_truncation("A, B", "drd2_human", oligo, set())
        assert chain == "B"
        assert uniprot == "oprm_human"

    def test_orphaned_ligand_warning(self):
        oligo = {
            "primary_protomer_suggestion": {"chain_id": "A", "reason": "test"},
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "aa2ar_human"},
                {"chain_id": "B", "slug": "aa2ar_human"},
            ],
        }
        chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, {"A", "B"})
        assert chain == "A"
        assert "[WARNING: Ligands are bound to truncated chains B!]" in note

    def test_no_orphan_when_ligand_on_primary(self):
        oligo = {
            "primary_protomer_suggestion": {"chain_id": "A", "reason": "test"},
            "all_gpcr_chains": [
                {"chain_id": "A", "slug": "aa2ar_human"},
                {"chain_id": "B", "slug": "aa2ar_human"},
            ],
        }
        _chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, {"A"})
        assert "WARNING" not in note
        assert "[DB TRUNCATION:" in note

    def test_no_suggestion_returns_original(self):
        """Multi-chain but no primary_protomer_suggestion → no truncation."""
        chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", {}, set())
        assert chain == "A, B"
        assert note == ""

    def test_suggestion_missing_chain_id(self):
        oligo = {"primary_protomer_suggestion": {"reason": "test"}}
        chain, _uniprot, note = apply_db_truncation("A, B", "aa2ar_human", oligo, set())
        assert chain == "A, B"
        assert note == ""


# ── build_structure_note ─────────────────────────────────────────────


class TestBuildStructureNote:
    def test_empty_oligo(self):
        result = build_structure_note({"note": "Base note"}, {})
        assert result == "Base note"

    def test_no_note_no_oligo(self):
        result = build_structure_note({}, {})
        assert result == ""

    def test_chain_corrected(self):
        oligo = {
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "G",
                "corrected_chain_id": "R",
                "trigger": "HALLUCINATION",
            },
            "alerts": [],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[CHAIN CORRECTED: G -> R" in result

    def test_homomer_classification(self):
        oligo = {
            "classification": "HOMOMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [
                {"chain_id": "A"},
                {"chain_id": "B"},
            ],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[HOMOMER: chains A, B]" in result

    def test_heteromer_classification(self):
        oligo = {
            "classification": "HETEROMER",
            "chain_id_override": {"applied": False},
            "alerts": [],
            "all_gpcr_chains": [
                {"chain_id": "R"},
                {"chain_id": "S"},
            ],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[HETEROMER: chains R, S]" in result

    def test_missed_protomer_alert(self):
        # Real validator messages already carry a "[TYPE] at '...'" prefix.
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [
                {
                    "type": "MISSED_PROTOMER",
                    "message": "[MISSED_PROTOMER] at 'oligomer_analysis': Missed B",
                }
            ],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[MISSED_PROTOMER]" in result
        assert "Missed B" in result
        # The type prefix must appear exactly once in the persisted note.
        assert result.count("[MISSED_PROTOMER]") == 1
        assert "[MISSED_PROTOMER: [MISSED_PROTOMER]" not in result

    def test_hallucination_alert(self):
        oligo = {
            "chain_id_override": {"applied": False},
            "alerts": [
                {
                    "type": "HALLUCINATION",
                    "message": "[HALLUCINATION] at 'oligomer_analysis': Chain G fake",
                }
            ],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "[HALLUCINATION]" in result
        assert "Chain G fake" in result
        assert result.count("[HALLUCINATION]") == 1
        assert "[HALLUCINATION: [HALLUCINATION]" not in result

    def test_confirmed_oligomer_not_included(self):
        """CONFIRMED_OLIGOMER alerts should NOT appear in the note."""
        oligo = {
            "classification": "MONOMER",
            "chain_id_override": {"applied": False},
            "alerts": [{"type": "CONFIRMED_OLIGOMER", "message": "All good"}],
            "all_gpcr_chains": [],
        }
        result = build_structure_note({"note": ""}, oligo)
        assert "CONFIRMED_OLIGOMER" not in result

    def test_with_truncation_note(self):
        result = build_structure_note(
            {"note": "Base"},
            {},
            truncation_note="[DB TRUNCATION: test]",
        )
        assert result == "Base [DB TRUNCATION: test]"

    def test_combined(self):
        """Base note + override + classification + truncation → all present."""
        oligo = {
            "classification": "HOMOMER",
            "chain_id_override": {
                "applied": True,
                "original_chain_id": "X",
                "corrected_chain_id": "A",
                "trigger": "7TM_UPGRADE",
            },
            "alerts": [],
            "all_gpcr_chains": [{"chain_id": "A"}, {"chain_id": "B"}],
        }
        result = build_structure_note(
            {"note": "Cryo-EM complex"},
            oligo,
            truncation_note="[DB TRUNCATION: test]",
        )
        assert "Cryo-EM complex" in result
        assert "[CHAIN CORRECTED:" in result
        assert "[HOMOMER:" in result
        assert "[DB TRUNCATION:" in result

    def test_non_string_note(self):
        """Non-string note values should be handled gracefully."""
        result = build_structure_note({"note": 42}, {})
        assert result == "42"
