"""Cross-field consistency advisories over a structure's best-run annotation.

These are coupling-aware sanity checks that compare independently annotated
fields and, where they sit uneasily together, ask a curator to confirm rather
than assert an error. The functions are pure (dict in, list-of-strings out) and
depend only on ``config`` so they can run inside the aggregator's validation
report and be replayed offline over recorded annotations.
"""

from __future__ import annotations

from typing import Any

from gpcr_tools.config import ALERT_PREFIX_STATE_CONFIRMATION

# Ligand roles that stabilise the inactive receptor conformation. Kept narrow on
# purpose: an antagonist / NAM merely blocks activation and routinely co-occurs
# with an active-state call, so including them would raise false advisories.
_INACTIVE_STABILISING_ROLES: frozenset[str] = frozenset({"Inverse agonist"})


def _role_value(ligand: dict[str, Any]) -> str | None:
    role = ligand.get("role")
    if isinstance(role, dict):
        return role.get("value")
    return role if isinstance(role, str) else None


def _has_g_protein_partner(best_run_data: dict[str, Any]) -> bool:
    """Whether a G protein transducer (with an alpha subunit) is modelled."""
    signaling = best_run_data.get("signaling_partners") or {}
    g_protein = signaling.get("g_protein") or {}
    return bool(g_protein.get("alpha_subunit"))


def state_ligand_consistency_warnings(best_run_data: dict[str, Any]) -> list[str]:
    """Advisories where the receptor state sits uneasily with the bound ligand.

    Fires only when all of the following hold, which together describe an
    active-state call that its own supporting evidence does not obviously back:

    - ``structure_info.state.value`` is ``"active"``;
    - at least one ligand carries an inactive-stabilising role (inverse agonist);
    - no G protein transducer is modelled (``signaling_partners.g_protein``
      absent, or present without an ``alpha_subunit``).

    A modelled G protein is deliberately treated as settling the state: a
    G-protein-coupled complex carrying an inverse agonist is real published
    biology, not something to flag. Antagonist / NAM ligands are excluded so a
    blocked-but-active structure does not raise a false advisory.

    Returns a single light-touch, curator-facing advisory string, or ``[]``.
    """
    structure_info = best_run_data.get("structure_info") or {}
    state = structure_info.get("state")
    state_value = state.get("value") if isinstance(state, dict) else state
    if state_value != "active":
        return []

    if _has_g_protein_partner(best_run_data):
        return []

    inactive_ligands = [
        ligand.get("name") or "unnamed ligand"
        for ligand in best_run_data.get("ligands") or []
        if _role_value(ligand) in _INACTIVE_STABILISING_ROLES
    ]
    if not inactive_ligands:
        return []

    ligand_list = ", ".join(inactive_ligands)
    return [
        f"{ALERT_PREFIX_STATE_CONFIRMATION} at 'structure_info.state': the state "
        f"is annotated 'active', but an inactive-stabilising ligand ({ligand_list}) "
        f"is bound and no G protein transducer is modelled. Please confirm the "
        f"receptor state."
    ]
