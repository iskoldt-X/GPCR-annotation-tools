"""Tests for the membrane-frame geometry foundation (gemmi, synthetic structures).

No network, no fixtures: a synthetic "membrane belt" (a hydrophobic slab of Cα
perpendicular to z, with hydrophilic caps above/below) is built in-memory so the
ANVIL-style fit can be checked deterministically.
"""

import math

import gemmi

from gpcr_tools.validator.geometry import ligand_bfactor_ratios
from gpcr_tools.validator.membrane import (
    MembraneFrame,
    ligand_membrane_depth,
    membrane_frame,
)


def _atom(name: str, x: float, y: float, z: float, b_iso: float = 20.0) -> gemmi.Atom:
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.b_iso = b_iso
    a.element = gemmi.Element("C")
    return a


def _residue(name: str, seqid: int, atoms: list[gemmi.Atom], het: str = "A") -> gemmi.Residue:
    res = gemmi.Residue()
    res.name = name
    res.seqid = gemmi.SeqId(seqid, " ")
    res.het_flag = het
    for a in atoms:
        res.add_atom(a)
    return res


def _structure(residues: list[gemmi.Residue], chain_name: str = "A") -> gemmi.Structure:
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_name)
    for res in residues:
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def _membrane_belt_structure() -> gemmi.Structure:
    """Hydrophobic Cα slab in z in [-14, 14]; hydrophilic caps at z = +/-28."""
    residues: list[gemmi.Residue] = []
    for i in range(80):  # belt: hydrophobic (LEU), spread around a cylinder
        ang = i * 0.7
        z = -14.0 + 28.0 * (i / 79.0)
        residues.append(
            _residue("LEU", i + 1, [_atom("CA", 18 * math.cos(ang), 18 * math.sin(ang), z)])
        )
    for j in range(30):  # caps: hydrophilic (ARG)
        ang = j * 0.9
        z = 28.0 if j % 2 == 0 else -28.0
        residues.append(
            _residue("ARG", 200 + j, [_atom("CA", 6 * math.cos(ang), 6 * math.sin(ang), z)])
        )
    return _structure(residues)


class TestMembraneFrame:
    def test_recovers_belt_normal_and_band(self):
        frame = membrane_frame(_membrane_belt_structure())
        assert frame is not None
        # The membrane belt is perpendicular to z, so the normal aligns with z
        # (either sign), centred near the slab mid-plane (z=0), ~28 Å thick.
        assert abs(frame.normal[2]) > 0.8
        assert abs(frame.center) < 6.0
        # Half-thickness is approximate (objective plateaus once the slab spans the
        # belt); 20.0 is the theoretical max given MEMBRANE_THICKNESS_MAX=40.
        assert 10.0 < frame.half_thickness <= 20.0

    def test_too_few_residues_returns_none(self):
        residues = [_residue("LEU", i + 1, [_atom("CA", float(i), 0.0, 0.0)]) for i in range(10)]
        assert membrane_frame(_structure(residues)) is None

    def test_empty_structure_returns_none(self):
        assert membrane_frame(gemmi.Structure()) is None


class TestLigandMembraneDepth:
    _FRAME = MembraneFrame(normal=(0.0, 0.0, 1.0), center=0.0, half_thickness=14.0)

    def test_centre_in_band(self):
        depth, in_band = ligand_membrane_depth(self._FRAME, [_atom("C1", 0.0, 0.0, 0.0)])
        assert depth == 0.0
        assert in_band

    def test_far_out_of_band(self):
        depth, in_band = ligand_membrane_depth(self._FRAME, [_atom("C1", 0.0, 0.0, 30.0)])
        assert depth == 30.0
        assert not in_band

    def test_edge_within_margin(self):
        # 15 Å is outside the 14 Å half-slab but within the +/-3 Å band margin.
        _, in_band = ligand_membrane_depth(self._FRAME, [_atom("C1", 0.0, 0.0, 15.0)])
        assert in_band

    def test_empty_atoms_returns_none(self):
        assert ligand_membrane_depth(self._FRAME, []) is None


class TestLigandBFactorRatios:
    def _structure_with_ligand(self, lig_b: float, env_b: float) -> gemmi.Structure:
        # One protein residue (env) ~2 Å from a one-atom ligand "LIG".
        prot = _residue("LEU", 1, [_atom("CA", 0.0, 0.0, 0.0, env_b)])
        lig = _residue("LIG", 2, [_atom("C1", 2.0, 0.0, 0.0, lig_b)], het="H")
        return _structure([prot, lig])

    def test_ratio_one_when_matched(self):
        assert ligand_bfactor_ratios(self._structure_with_ligand(20.0, 20.0), "LIG") == [1.0]

    def test_ratio_high_when_disordered(self):
        assert ligand_bfactor_ratios(self._structure_with_ligand(60.0, 20.0), "LIG") == [3.0]

    def test_none_without_protein_environment(self):
        lig = _residue("LIG", 1, [_atom("C1", 0.0, 0.0, 0.0, 30.0)], het="H")
        assert ligand_bfactor_ratios(_structure([lig]), "LIG") == [None]

    def test_none_when_environment_b_unset(self):
        # All-zero protein B (e.g. a cryo-EM model) -> no usable ratio, not 0/inf.
        assert ligand_bfactor_ratios(self._structure_with_ligand(40.0, 0.0), "LIG") == [None]

    def test_none_when_ligand_b_unset(self):
        # Zero ligand B with a set environment must not read as a perfect ratio of 0.
        assert ligand_bfactor_ratios(self._structure_with_ligand(0.0, 20.0), "LIG") == [None]
