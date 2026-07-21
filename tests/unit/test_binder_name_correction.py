"""Tests for antigen-mislabelled binder-name correction.

The model sometimes names an auxiliary binder (Fab / nanobody / scFv / DARPin)
after the ANTIGEN it binds -- an "anti-BRIL Fab" annotated simply as "BRIL". The
correct name lives in the chain's RCSB description (``pdbx_description``), read at
aggregation time. A binder is renamed only when it is a binder type, carries no
GPCRdb slug, and its description matches an antigen-agnostic anti-X / X-binding
pattern; the correction is advisory (non-blocking). Descriptions here are the real
RCSB ``pdbx_description`` strings for 9D3G / 9IMA / 7EPB / 8TB7.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.aggregator.runner import _build_validation_report
from gpcr_tools.config import CHIMERA_STATUS_SKIPPED
from gpcr_tools.validator.oligomer import correct_binder_names


def _entity(auth: str, slug: str | None, description: str) -> dict[str, Any]:
    return {
        "uniprots": [{"gpcrdb_entry_name_slug": slug}] if slug else [],
        "rcsb_polymer_entity": {"pdbx_description": description},
        "polymer_entity_instances": [
            {"rcsb_polymer_entity_instance_container_identifiers": {"auth_asym_id": auth}}
        ],
    }


def _enriched(*entities: Any) -> dict[str, Any]:
    return {"polymer_entities": list(entities)}


def _aux(name: str, type_value: str, chain_id: str) -> dict[str, Any]:
    return {"name": name, "type": {"value": type_value}, "chain_id": chain_id}


class TestCorrectBinderNames:
    def test_9d3g_anti_bril_fab_and_nanobody(self) -> None:
        # 9D3G: an anti-BRIL Fab (H/L) + an anti-Fab nanobody, all named "BRIL"
        # by the model. Each is renamed from its own RCSB chain description; the
        # Fab's two chains share antigen+type, differing only by Heavy/Light role.
        enriched = _enriched(
            _entity("A", None, "CCR6, Soluble cytochrome b562"),
            _entity("H", None, "anti-BRIL Fab Heavy chain"),
            _entity("L", None, "anti-BRIL Fab Light chain"),
            _entity("K", None, "anti-BRIL Fab Nanobody"),
        )
        heavy = _aux("BRIL", "Antibody fab fragment", "H")
        light = _aux("BRIL", "Antibody fab fragment", "L")
        nanobody = _aux("BRIL", "Nanobody", "K")
        notes = correct_binder_names(enriched, [heavy, light, nanobody])
        assert heavy["name"] == "anti-BRIL Fab Heavy chain"
        assert light["name"] == "anti-BRIL Fab Light chain"
        assert nanobody["name"] == "anti-BRIL Nanobody"
        # The Fab's two chains agree on antigen+type; only the role differs.
        assert heavy["name"].rsplit(" ", 2)[0] == light["name"].rsplit(" ", 2)[0] == "anti-BRIL Fab"
        assert len(notes) == 3
        assert all("BINDER NAME CORRECTED" in n and "auxiliary_proteins" in n for n in notes)

    def test_9ima_receptor_antigen_paren_stripped(self) -> None:
        # 9IMA: "Talquetamab Fab (anti-GPRC5D)" -> antigen GPRC5D, paren stripped.
        enriched = _enriched(
            _entity("A", "gpc5d_human", "G-protein coupled receptor family C group 5 member D"),
            _entity("C", None, "Talquetamab Fab (anti-GPRC5D) Heavy chain"),
            _entity("D", None, "Talquetamab Fab (anti-GPRC5D) Light chain"),
        )
        heavy = _aux("GPRC5D", "Antibody fab fragment", "C")
        light = _aux("GPRC5D", "Antibody fab fragment", "D")
        correct_binder_names(enriched, [heavy, light])
        assert heavy["name"] == "anti-GPRC5D Fab Heavy chain"
        assert light["name"] == "anti-GPRC5D Fab Light chain"

    def test_7epb_anti_ron_nanobody(self) -> None:
        # 7EPB: "Anti-RON nanobody" on two chains -> anti-RON Nanobody.
        enriched = _enriched(
            _entity("A", "grm2_human", "Metabotropic glutamate receptor 2"),
            _entity("C", None, "Anti-RON nanobody"),
            _entity("D", None, "Anti-RON nanobody"),
        )
        nb = _aux("RON", "Nanobody", "C, D")
        correct_binder_names(enriched, [nb])
        assert nb["name"] == "anti-RON Nanobody"

    def test_8tb7_hinge_binding_nanobody(self) -> None:
        # 8TB7: "Fab hinge-binding nanobody" -> X-binding pattern captures "hinge".
        # The model named it after the antigen it binds ("hinge"), so the guard
        # (raw token must appear in the current name) passes and it is corrected.
        enriched = _enriched(_entity("N", None, "Fab hinge-binding nanobody"))
        nb = _aux("hinge", "Nanobody", "N")
        correct_binder_names(enriched, [nb])
        assert nb["name"] == "anti-hinge Nanobody"

    def test_negative_clone_name_not_containing_antigen_not_renamed(self) -> None:
        # 7SRS: the model correctly named an anti-5-HT2B Fab by its CLONE ("P2C2
        # Fab"); the RCSB description is "Anti-5HT2BR Fab light chain" (antigen
        # token "5HT2BR"). All three original gates pass, but the model did NOT
        # name the binder after the antigen -- "5ht2br" is absent from "p2c2 fab"
        # -- so the guard must keep the more-specific clone name, not overwrite it.
        enriched = _enriched(
            _entity("R", "5ht2b_human", "5-hydroxytryptamine receptor 2B"),
            _entity("P", None, "Anti-5HT2BR Fab light chain"),
            _entity("Q", None, "Anti-5HT2BR Fab heavy chain"),
        )
        light = _aux("P2C2 Fab", "Antibody fab fragment", "P")
        heavy = _aux("P2C2 Fab", "Antibody fab fragment", "Q")
        notes = correct_binder_names(enriched, [light, heavy])
        assert light["name"] == "P2C2 Fab"
        assert heavy["name"] == "P2C2 Fab"
        assert notes == []

    def test_7srs_clone_tag_preserved_on_rename(self) -> None:
        # 7SRS: the model name is ALREADY informative -- "Anti-5HT2BR Fab (P2C2)"
        # carries the antigen (so the rename fires) AND the clone code "(P2C2)".
        # The rebuilt canonical name must PRESERVE the clone tag: "anti-5-HT2B Fab
        # (P2C2)", not drop it. (Role suffix omitted here: single-chain entry whose
        # description has no Heavy/Light word.)
        enriched = _enriched(
            _entity("R", "5ht2b_human", "5-hydroxytryptamine receptor 2B"),
            _entity("P", None, "Anti-5HT2BR Fab"),
        )
        fab = _aux("Anti-5HT2BR Fab (P2C2)", "Antibody fab fragment", "P")
        notes = correct_binder_names(enriched, [fab])
        assert fab["name"] == "anti-5-HT2B Fab (P2C2)"
        assert len(notes) == 1

    def test_9ima_antigen_parenthetical_not_duplicated(self) -> None:
        # 9IMA-style: the model name's trailing parenthetical only RESTATES the
        # antigen ("(anti-GPRC5D)"), so it must NOT be appended -- that would double
        # it into "anti-GPRC5D Fab (anti-GPRC5D)". The canonical name stays clean.
        enriched = _enriched(
            _entity("A", "gpc5d_human", "G-protein coupled receptor family C group 5 member D"),
            _entity("C", None, "anti-GPRC5D Fab"),
        )
        fab = _aux("Talquetamab Fab (anti-GPRC5D)", "Antibody fab fragment", "C")
        correct_binder_names(enriched, [fab])
        assert fab["name"] == "anti-GPRC5D Fab"

    def test_bril_no_parenthetical_unchanged_by_clone_rule(self) -> None:
        # A plain antigen name with no trailing parenthetical is renamed exactly as
        # before -- the clone-preservation rule is a no-op when there is no paren.
        enriched = _enriched(_entity("H", None, "anti-BRIL Fab Heavy chain"))
        fab = _aux("BRIL", "Antibody fab fragment", "H")
        notes = correct_binder_names(enriched, [fab])
        assert fab["name"] == "anti-BRIL Fab Heavy chain"
        assert len(notes) == 1

    def test_combined_two_chain_entry_gets_base_name_no_role_suffix(self) -> None:
        # 7SRS anti-5-HT2B Fab, but the model captured BOTH chains in ONE entry
        # (chain_id "P, Q") and named it after the antigen ("5HT2BR"). Appending a
        # single chain's Heavy/Light role would mislabel the other chain, so a
        # combined entry gets the base "anti-5-HT2B Fab" with NO role suffix.
        enriched = _enriched(
            _entity("R", "5ht2b_human", "5-hydroxytryptamine receptor 2B"),
            _entity("P", None, "Anti-5HT2BR Fab light chain"),
            _entity("Q", None, "Anti-5HT2BR Fab heavy chain"),
        )
        fab = _aux("5HT2BR", "Antibody fab fragment", "P, Q")
        notes = correct_binder_names(enriched, [fab])
        assert fab["name"] == "anti-5-HT2B Fab"
        assert "Heavy chain" not in fab["name"]
        assert "Light chain" not in fab["name"]
        assert len(notes) == 1

    def test_negative_clone_named_binder_not_renamed(self) -> None:
        # A clone-named binder (no anti-X / X-binding pattern) keeps its name.
        enriched = _enriched(
            _entity("L", None, "Fab24 BAK5 light chain"),
            _entity("S", None, "scFv16"),
        )
        fab = _aux("Fab24 BAK5", "Antibody fab fragment", "L")
        scfv = _aux("scFv16", "scFv", "S")
        notes = correct_binder_names(enriched, [fab, scfv])
        assert fab["name"] == "Fab24 BAK5"
        assert scfv["name"] == "scFv16"
        assert notes == []

    def test_negative_receptor_side_fusion_not_touched(self) -> None:
        # A receptor-side fusion HAS a GPCRdb slug, so the no-slug gate excludes it
        # -- even if it were (wrongly) typed as a binder.
        enriched = _enriched(
            _entity("R", "gpr61_human", "G-protein coupled receptor 61,Soluble cytochrome b562"),
        )
        entry = _aux("BRIL", "Antibody fab fragment", "R")
        notes = correct_binder_names(enriched, [entry])
        assert entry["name"] == "BRIL"
        assert notes == []

    def test_failsafe_no_pattern_keeps_model_name(self) -> None:
        # A binder whose description has no anti-X / X-binding pattern is not
        # renamed and is never blanked.
        enriched = _enriched(_entity("N", None, "Nanobody Nb35"))
        nb = _aux("Nb35", "Nanobody", "N")
        notes = correct_binder_names(enriched, [nb])
        assert nb["name"] == "Nb35"
        assert notes == []

    def test_non_binder_type_ignored(self) -> None:
        # A non-binder aux type (e.g. Fusion protein / RAMP) is never renamed even
        # if its description happens to match a pattern.
        enriched = _enriched(_entity("A", None, "anti-BRIL Fab Heavy chain"))
        fusion = _aux("BRIL", "Fusion protein", "A")
        assert correct_binder_names(enriched, [fusion]) == []
        assert fusion["name"] == "BRIL"

    def test_none_safe(self) -> None:
        assert correct_binder_names({}, None) == []
        assert correct_binder_names({}, []) == []
        assert correct_binder_names({"polymer_entities": []}, [_aux("X", "Nanobody", "A")]) == []


class TestRoutingInvariant:
    def test_binder_rename_note_is_advisory_not_blocking(self) -> None:
        # The correction note lands in detector_notes (non-blocking), never in
        # critical_warnings (which would gate accept-all).
        enriched = _enriched(_entity("H", None, "anti-BRIL Fab Heavy chain"))
        best = {"auxiliary_proteins": [_aux("BRIL", "Antibody fab fragment", "H")]}
        report = _build_validation_report(
            "TEST", best, enriched, [], {"status": CHIMERA_STATUS_SKIPPED, "score": 0}, None
        )
        assert any("BINDER NAME CORRECTED" in n for n in report["detector_notes"])
        assert not any("BINDER NAME CORRECTED" in w for w in report["critical_warnings"])
        # The rename is applied in place on the best-run aux entry.
        assert best["auxiliary_proteins"][0]["name"] == "anti-BRIL Fab Heavy chain"
