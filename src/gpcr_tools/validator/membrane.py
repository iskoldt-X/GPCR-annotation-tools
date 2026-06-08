"""Membrane spatial frame from coordinates alone (gemmi, no AI, no GPCRdb).

Fits an approximate lipid-bilayer frame for a membrane protein by an ANVIL-style
hydrophobic-slab search: over many candidate membrane normals (golden-spiral
sphere points) and slab thicknesses, find the slab that best encloses exposed
HYDROPHOBIC residues while excluding hydrophilic ones. From that frame we derive,
per ligand copy, its signed depth relative to the bilayer mid-plane and whether it
sits within the membrane band.

Pure geometry, offline, no GPCRdb dependency. The exposure proxy + objective are
deliberately simple and want calibration against OPM/PPM on a labelled GPCR set
before the absolute numbers are trusted; the *direction* (membrane normal) is the
robust output. The fitted half-thickness is only approximate -- the simplified
objective plateaus once the slab spans the hydrophobic belt -- so
``ligand_membrane_depth`` reports the signed depth as the primary value and treats
the membrane band as a soft margin. Note: the normal's sign is arbitrary (the slab
is symmetric), so this
frame gives depth + in/out-of-band, NOT which side is extracellular vs
intracellular — orienting the sides needs a separate reference and is left to the
caller. Failures (too few residues, degenerate geometry) return ``None`` so a
detector never breaks on an odd structure.

The membrane normal is computed from the whole protein; for a receptor-plus-
transducer complex the soluble partner's surface is mostly hydrophilic and does
not pull the slab off the bilayer belt (a calibration note, not a correctness
claim).
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass

import gemmi

from gpcr_tools.config import (
    MEMBRANE_BAND_MARGIN,
    MEMBRANE_EXPOSURE_CA_RADIUS,
    MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA,
    MEMBRANE_HYDROPHOBIC_RESIDUES,
    MEMBRANE_MIN_TM_CA,
    MEMBRANE_NORMAL_CANDIDATES,
    MEMBRANE_OFFSET_STEP,
    MEMBRANE_THICKNESS_MAX,
    MEMBRANE_THICKNESS_MIN,
    MEMBRANE_THICKNESS_STEP,
)

# Reused geometry helpers (golden-spiral directions; polymer-residue test).
from gpcr_tools.validator.geometry import fibonacci_directions, is_protein_atom

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MembraneFrame:
    """A fitted bilayer frame. ``normal`` is a unit vector (arbitrary sign);
    ``center`` is the bilayer mid-plane's projection along ``normal``;
    ``half_thickness`` is half the hydrophobic slab thickness (Å)."""

    normal: tuple[float, float, float]
    center: float
    half_thickness: float


def _exposed_residue_projections(
    model: gemmi.Model, cell: gemmi.UnitCell
) -> list[tuple[gemmi.Position, bool]]:
    """Surface-exposed protein Cα as (position, is_hydrophobic).

    Exposure proxy (no SASA dependency): a Cα with fewer than
    ``MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA`` other Cα within
    ``MEMBRANE_EXPOSURE_CA_RADIUS`` is treated as surface-facing.
    """
    cas: list[tuple[gemmi.Atom, bool]] = []
    for chain in model:
        for residue in chain:
            if not is_protein_atom(residue):
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is not None:
                cas.append((ca, residue.name in MEMBRANE_HYDROPHOBIC_RESIDUES))
    if len(cas) < MEMBRANE_MIN_TM_CA:
        return []
    ns = gemmi.NeighborSearch(model, cell, MEMBRANE_EXPOSURE_CA_RADIUS).populate()
    exposed: list[tuple[gemmi.Position, bool]] = []
    for ca, hydrophobic in cas:
        neighbors = 0
        for mark in ns.find_atoms(ca.pos, "\0", radius=MEMBRANE_EXPOSURE_CA_RADIUS):
            cra = mark.to_cra(model)
            # Protein-gate: a calcium ion is also named "CA"; only protein
            # Cα count toward residue density.
            if (
                is_protein_atom(cra.residue)
                and cra.atom.name == "CA"
                and ca.pos.dist(cra.atom.pos) <= MEMBRANE_EXPOSURE_CA_RADIUS
            ):
                neighbors += 1
        if neighbors - 1 < MEMBRANE_EXPOSURE_MAX_NEIGHBOR_CA:  # minus self
            exposed.append((ca.pos, hydrophobic))
    return exposed


def membrane_frame(structure: gemmi.Structure) -> MembraneFrame | None:
    """Fit a bilayer frame to *structure*, or ``None`` if it cannot be resolved."""
    if len(structure) == 0:
        return None
    model = structure[0]
    exposed = _exposed_residue_projections(model, structure.cell)
    if len(exposed) < MEMBRANE_MIN_TM_CA:
        return None

    best: tuple[float, tuple[float, float, float], float, float] | None = None
    for normal in fibonacci_directions(MEMBRANE_NORMAL_CANDIDATES):
        projected = sorted(
            (p.x * normal.x + p.y * normal.y + p.z * normal.z, hydrophobic)
            for p, hydrophobic in exposed
        )
        coords = [c for c, _ in projected]
        # Prefix sum of hydrophobic count over the sorted projections, so the
        # hydrophobic/hydrophilic counts inside any [a, b] window are O(log n).
        prefix_hydrophobic = [0]
        for _, hydrophobic in projected:
            prefix_hydrophobic.append(prefix_hydrophobic[-1] + (1 if hydrophobic else 0))
        lo, hi = coords[0], coords[-1]
        thickness = MEMBRANE_THICKNESS_MIN
        while thickness <= MEMBRANE_THICKNESS_MAX:
            half = thickness / 2.0
            center = lo + half
            while center <= hi - half + 1e-9:
                left = bisect.bisect_left(coords, center - half)
                right = bisect.bisect_right(coords, center + half)
                hydrophobic_in = prefix_hydrophobic[right] - prefix_hydrophobic[left]
                hydrophilic_in = (right - left) - hydrophobic_in
                # ANVIL principle: the membrane slab maximises enclosed exposed
                # hydrophobics minus enclosed hydrophilics (a thickness-bounded,
                # simplified form of the published Q score).
                score = float(hydrophobic_in - hydrophilic_in)
                if best is None or score > best[0]:
                    best = (score, (normal.x, normal.y, normal.z), center, half)
                center += MEMBRANE_OFFSET_STEP
            thickness += MEMBRANE_THICKNESS_STEP

    # A non-positive best score means the exposed set has no hydrophobic belt
    # to enclose (e.g. a soluble fragment): return None rather than a noise fit.
    if best is None or best[0] <= 0:
        return None
    return MembraneFrame(normal=best[1], center=best[2], half_thickness=best[3])


def ligand_membrane_depth(
    frame: MembraneFrame, ligand_atoms: list[gemmi.Atom]
) -> tuple[float, bool] | None:
    """Signed depth of a ligand copy's centroid vs the bilayer mid-plane.

    Returns ``(signed_depth, in_band)`` (depth in Å along the membrane normal,
    sign arbitrary; ``in_band`` true within the hydrophobic slab plus
    ``MEMBRANE_BAND_MARGIN``), or ``None`` for an empty atom list.
    """
    if not ligand_atoms:
        return None
    n = len(ligand_atoms)
    cx = sum(a.pos.x for a in ligand_atoms) / n
    cy = sum(a.pos.y for a in ligand_atoms) / n
    cz = sum(a.pos.z for a in ligand_atoms) / n
    nx, ny, nz = frame.normal
    signed_depth = (cx * nx + cy * ny + cz * nz) - frame.center
    in_band = abs(signed_depth) <= frame.half_thickness + MEMBRANE_BAND_MARGIN
    return round(signed_depth, 1), in_band
