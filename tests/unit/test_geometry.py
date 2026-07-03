"""Tests for validator.geometry pure helpers (no coordinate I/O or network).

The burial metric is the load-bearing separator between a ligand enclosed in a
pocket and one lying on the membrane-facing surface, so it is exercised directly
on synthetic environments.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
import requests

from gpcr_tools.config import API_MAX_RETRIES
from gpcr_tools.validator import geometry as geom
from gpcr_tools.validator.geometry import (
    _SPHERE_DIRECTIONS,
    LigandCopyGeometry,
    _burial,
    analyze_ligand_copies,
    centroid,
    fetch_structure,
    fibonacci_directions,
    ligand_contact_residues,
    ligand_interaction_counts,
    load_structure,
)


class TestFibonacciDirections:
    def test_count_and_unit_length(self) -> None:
        dirs = fibonacci_directions(200)
        assert len(dirs) == 200
        for d in dirs:
            assert abs(d.length() - 1.0) < 1e-6


class TestBurial:
    def test_fully_enclosed_is_one(self) -> None:
        # An environment atom in every sampled direction shields the centroid fully.
        origin = gemmi.Position(0.0, 0.0, 0.0)
        env = [gemmi.Position(d.x * 3.0, d.y * 3.0, d.z * 3.0) for d in _SPHERE_DIRECTIONS]
        assert _burial(origin, env) == 1.0

    def test_one_sided_is_low(self) -> None:
        # Atoms only on one hemisphere (a surface-exposed copy) cover well under half.
        origin = gemmi.Position(0.0, 0.0, 0.0)
        env = [
            gemmi.Position(d.x * 3.0, d.y * 3.0, d.z * 3.0) for d in _SPHERE_DIRECTIONS if d.x > 0.5
        ]
        assert _burial(origin, env) < 0.8

    def test_no_environment_is_zero(self) -> None:
        assert _burial(gemmi.Position(0.0, 0.0, 0.0), []) == 0.0


class TestCentroid:
    def test_mean_position(self) -> None:
        atoms = []
        for x in (0.0, 2.0):
            atom = gemmi.Atom()
            atom.pos = gemmi.Position(x, 0.0, 0.0)
            atoms.append(atom)
        c = centroid(atoms)
        assert (c.x, c.y, c.z) == (1.0, 0.0, 0.0)


class TestLigandCopyGeometry:
    def test_primary_chain_and_residue_numbers(self) -> None:
        copy = LigandCopyGeometry(
            auth_chain="R",
            seq_id=601,
            burial=0.99,
            pocket_residues=frozenset({("R", 104), ("R", 107), ("A", 12)}),
            contacts_partner=True,
        )
        assert copy.n_pocket_residues == 3
        assert copy.primary_gpcr_chain() == "R"  # the chain with the most residues
        assert copy.residue_numbers_on("R") == frozenset({104, 107})
        assert copy.residue_numbers_on("A") == frozenset({12})

    def test_primary_chain_none_when_empty(self) -> None:
        copy = LigandCopyGeometry("R", 1, 0.0, frozenset(), False)
        assert copy.primary_gpcr_chain() is None


class _FakeResp:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        pass


class TestFetchStructure:
    def test_cache_hit_does_not_hit_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cached = tmp_path / "structure_files" / "9iix.cif.gz"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")

        def boom(*a: object, **k: object) -> None:
            raise AssertionError("network must not be used on a cache hit")

        monkeypatch.setattr(requests, "get", boom)
        assert fetch_structure("9IIX", tmp_path) == cached

    def test_downloads_and_caches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResp(b"coords"))
        path = fetch_structure("9IIX", tmp_path)
        assert path is not None and path.read_bytes() == b"coords"

    def test_download_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*a: object, **k: object) -> None:
            raise requests.RequestException("boom")

        monkeypatch.setattr(requests, "get", fail)
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        assert fetch_structure("9IIX", tmp_path) is None

    def test_transient_status_then_success_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A transient 503 on the first attempt must be retried; the next 200 wins.
        responses = [_FakeResp(b"", status_code=503), _FakeResp(b"coords", status_code=200)]
        monkeypatch.setattr(requests, "get", lambda *a, **k: responses.pop(0))
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        path = fetch_structure("9IIX", tmp_path)
        assert path is not None and path.read_bytes() == b"coords"
        assert responses == []  # both responses consumed (one retry happened)

    def test_404_returns_none_without_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A definitive 404 is not retried -- the request is made exactly once.
        calls = {"n": 0}

        def get(*a: object, **k: object) -> _FakeResp:
            calls["n"] += 1
            return _FakeResp(b"", status_code=404)

        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        assert fetch_structure("9IIX", tmp_path) is None
        assert calls["n"] == 1

    def test_all_attempts_transient_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def get(*a: object, **k: object) -> _FakeResp:
            calls["n"] += 1
            return _FakeResp(b"", status_code=503)

        monkeypatch.setattr(requests, "get", get)
        monkeypatch.setattr(geom.time, "sleep", lambda *_: None)
        assert fetch_structure("9IIX", tmp_path) is None
        assert calls["n"] == API_MAX_RETRIES


class TestLoadStructure:
    def test_missing_download_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(geom, "fetch_structure", lambda *a, **k: None)
        assert load_structure("9IIX", tmp_path) is None

    def test_unparseable_file_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bogus = tmp_path / "bogus.cif.gz"
        bogus.write_bytes(b"not a structure")
        monkeypatch.setattr(geom, "fetch_structure", lambda *a, **k: bogus)
        assert load_structure("9IIX", tmp_path) is None


def _atom(name: str, x: float, y: float, z: float, element: str = "C") -> gemmi.Atom:
    atom = gemmi.Atom()
    atom.name = name
    atom.pos = gemmi.Position(x, y, z)
    atom.element = gemmi.Element(element)
    return atom


def _het_residue(name: str, seq_id: int, x: float, elements: list[str]) -> gemmi.Residue:
    """A HETATM (non-polymer) residue -- a genuine free ligand or ion."""
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(seq_id, " ")
    res.het_flag = "H"
    for i, element in enumerate(elements):
        res.add_atom(_atom(f"{element}{i}", x + i * 1.4, 0.0, 0.0, element))
    return res


def _polymer_glu_chain(name: str) -> gemmi.Chain:
    """A short backbone stretch of glutamate residues (ATOM records).

    These share the ``GLU`` name of a free glutamate ligand but are part of the
    polymer, so they must never be picked up as ligand copies.
    """
    chain = gemmi.Chain(name)
    for i in range(3):
        res = gemmi.Residue()
        res.name = "GLU"
        res.seqid = gemmi.SeqId(i + 1, " ")
        res.het_flag = "A"
        base = i * 3.8
        res.add_atom(_atom("N", base, 0.0, 0.0, "N"))
        res.add_atom(_atom("CA", base + 1.0, 0.0, 0.0))
        res.add_atom(_atom("C", base + 2.0, 0.0, 0.0))
        res.add_atom(_atom("O", base + 2.5, 0.0, 0.0, "O"))
        chain.add_residue(res)
    return chain


def _modified_residue_chain(name: str) -> gemmi.Chain:
    """A polymer chain carrying one modified residue (selenomethionine, MSE).

    MSE is deposited as a HETATM but is part of the polymer, so it must be treated
    as protein -- never selected as a ligand copy. ``setup_entities`` assigns the
    polymer entity type that ``is_protein_atom`` reads.
    """
    chain = gemmi.Chain(name)
    for i, res_name in enumerate(("MET", "MSE", "MET")):
        res = gemmi.Residue()
        res.name = res_name
        res.seqid = gemmi.SeqId(i + 1, " ")
        res.het_flag = "H" if res_name == "MSE" else "A"  # MSE is HETATM in-polymer
        base = i * 3.8
        res.add_atom(_atom("N", base, 0.0, 0.0, "N"))
        res.add_atom(_atom("CA", base + 1.0, 0.0, 0.0))
        res.add_atom(_atom("C", base + 2.0, 0.0, 0.0))
        res.add_atom(_atom("O", base + 2.5, 0.0, 0.0, "O"))
        chain.add_residue(res)
    return chain


def _modified_residue_structure() -> gemmi.Structure:
    """A single polymer chain with an in-polymer modified residue (MSE)."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    model.add_chain(_modified_residue_chain("A"))
    st.add_model(model)
    st.setup_entities()
    return st


def _name_collision_structure() -> gemmi.Structure:
    """A structure where a free ``GLU`` ligand shares its name with backbone GLU.

    Chain A is three backbone glutamate residues; chain B carries one free
    glutamate ligand (comp_id GLU) plus a zinc ion, both close enough to the
    polymer to register contacts. Only the free ligand is a GLU copy.
    """
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    model.add_chain(_polymer_glu_chain("A"))
    ligands = gemmi.Chain("B")
    ligands.add_residue(_het_residue("GLU", 501, 2.0, ["N", "C", "C", "O"]))
    ligands.add_residue(_het_residue("ZN", 601, 3.0, ["ZN"]))
    model.add_chain(ligands)
    st.add_model(model)
    st.setup_entities()  # assigns entity types that is_protein_atom relies on
    return st


class TestLigandCopySelectionSkipsPolymer:
    """A ligand comp_id can collide with a standard amino-acid name (e.g. a free
    GLU ligand). Copy selection must count only genuine non-polymer copies, never
    the backbone residues that happen to share the name."""

    def test_analyze_ligand_copies_excludes_backbone(self) -> None:
        st = _name_collision_structure()
        copies = analyze_ligand_copies(st, "GLU", {"A"})
        # The three backbone GLU are excluded; only the free-ligand copy remains.
        assert len(copies) == 1

    def test_contact_residues_excludes_backbone(self) -> None:
        st = _name_collision_structure()
        assert len(ligand_contact_residues(st, "GLU", {"A"})) == 1

    def test_contact_residues_carry_copy_id(self) -> None:
        # Each copy leads with its own author chain + residue number (the copy identifier
        # auth_asym_id:auth_seq_id), read from the coordinate residue. The free GLU
        # ligand is chain B, residue 501.
        st = _name_collision_structure()
        (copy,) = ligand_contact_residues(st, "GLU", {"A"})
        auth_chain, auth_seq_id, burial, contacts = copy
        assert (auth_chain, auth_seq_id) == ("B", 501)
        assert isinstance(burial, float)
        assert isinstance(contacts, list)

    def test_interaction_counts_excludes_backbone(self) -> None:
        st = _name_collision_structure()
        assert len(ligand_interaction_counts(st, "GLU")) == 1

    def test_ion_copy_still_counted(self) -> None:
        # An ion is already non-polymer and must remain unaffected by the gate.
        st = _name_collision_structure()
        assert len(analyze_ligand_copies(st, "ZN", {"A"})) == 1
        assert len(ligand_contact_residues(st, "ZN", {"A"})) == 1
        assert len(ligand_interaction_counts(st, "ZN")) == 1

    def test_modified_residue_excluded(self) -> None:
        # A modified residue (MSE) is HETATM but in-polymer, so it is protein and
        # never a ligand copy. This exercises is_protein_atom's entity_type branch,
        # not het_flag alone.
        st = _modified_residue_structure()
        assert len(analyze_ligand_copies(st, "MSE", {"A"})) == 0
        assert len(ligand_contact_residues(st, "MSE", {"A"})) == 0
        assert len(ligand_interaction_counts(st, "MSE")) == 0
