"""CSV transformation and writing logic.

Data transformation and CSV file appending — no UI, no user interaction.
Converts reviewed JSON data into tabular CSV rows and appends them to disk.
"""

import csv
import os
from typing import Any

from gpcr_tools.config import (
    AUX_PROTEIN_DISPATCH,
    CSV_SCHEMA,
    MECHANICAL_AUX,
    SITE_REF_UNKNOWN,
    get_config,
    gpcrdb_aux_type_for,
    is_empty_key,
    ligand_routed_to_aux,
    ligand_row_dropped,
    sanitize_value,
)


def _primary_chain(value: Any) -> str:
    """Collapse a multi-chain G protein subunit value to a single chain.

    A single subunit physically occupies one chain per complex; a multi-value
    field (e.g. ``"C, D"``) means the asymmetric unit holds more than one
    redundant copy of the same complex. The structures row already collapses the
    receptor to its primary protomer, so the subunit collapses likewise — to the
    chain of the primary complex, which depositions list first. (A geometry-based
    pairing to the primary receptor chain would be exact; this first-listed
    heuristic covers the common ordered case.)
    """
    text = sanitize_value(value)
    # Split on comma or semicolon, tolerating optional spaces, to match the chain
    # parsers used elsewhere in the codebase; take the first (primary complex).
    return text.replace(";", ",").split(",")[0].strip()


def _per_site_label_and_residue_columns(
    ligands: list[Any],
    nonpolymer_instances: dict[str, Any],
    ligand_copies: Any,
) -> list[tuple[str, str]]:
    """Compute each ligand row's (``label_asym_id``, ``Residue_seq_id``) pair, filtered by site.

    A compound modelled at several binding sites is emitted as one ligand row per
    site (same component id, different ``site_ref``). Historically every such row
    listed the SAME full set of copies (a plain component-id join), so the rows
    were indistinguishable. Here each site row instead lists only the copies whose
    per-copy vote placed them at that site.

    A copy identifier is ``"<auth_asym_id>:<auth_seq_id>"`` -- the same string used
    both as the residue token and as the per-copy ``copy_id`` -- so a residue token
    maps to its voted site directly, and it belongs to the component because it
    comes from that component's own instance list. The paired ``label_asym_id`` and
    residue tokens are produced together from ONE filtered token list per row, so
    the two columns can never diverge in cardinality or order: they stay 1:1
    copy-for-copy on every row (the CSV schema's ``label_asym_id`` <-> residue
    contract).

    Only rows that SURVIVE to output are partitioned (``ligand_row_dropped``).
    ``unknown`` is treated as an ordinary binding site: a copy the runs could not
    place is voted ``unknown`` and lands on the compound's ``site_ref = unknown``
    row (which the aggregator builds for exactly these copies), residue token clean
    -- the uncertainty lives in the Site column, never as a mark on the residue. A
    copy voted to a site whose row was DROPPED (a non-functional / ghost / apo
    filter) follows that row out of the output, so a structural lipid the majority
    judged non-functional takes its copies with it instead of flooding a surviving
    sibling row.

    Honest fallbacks (never silently wrong, never silently lost):
      * No per-copy votes for this record (pre-feature data, or votes dropped
        upstream) -> every row keeps the full component-id-join list, unchanged.
      * A single-copy compound, or a polymer/keyless ligand with no instances ->
        nothing to partition; listed exactly as before.
      * A copy the aggregator did not route to a surviving site (an absent vote, or
        legacy data with no ``unknown`` row) is never dropped: it joins the
        compound's ``unknown`` row when one exists, else its first surviving row --
        always with a clean residue token.

    Returns a list parallel to *ligands* (same length and order); each element is
    that row's ``(label_asym_id, Residue_seq_id)`` pair.
    """
    results: list[tuple[str, str]] = [("", "") for _ in ligands]

    # Per-copy voted site, keyed by copy identifier. An absent/empty list means the
    # record carries no per-copy votes at all -> the component-id-join fallback.
    votes: dict[str, Any] = {}
    have_votes = isinstance(ligand_copies, list) and bool(ligand_copies)
    if have_votes:
        for row in ligand_copies:
            if isinstance(row, dict):
                copy_id = sanitize_value(row.get("copy_id"))
                if copy_id:
                    votes[copy_id] = row.get("site_ref")

    def _tokens(comp_id: str) -> list[tuple[str, str]]:
        # (label_asym_id, "<auth_asym_id>:<auth_seq_id>") for each modelled copy
        # that carries a label. The residue token is also the per-copy copy_id, so
        # a copy's voted site is looked up by it. Label and residue are emitted
        # together so the two columns stay aligned copy-for-copy.
        tokens: list[tuple[str, str]] = []
        for inst in nonpolymer_instances.get(comp_id) or []:
            if isinstance(inst, dict) and inst.get("label_asym_id"):
                label = sanitize_value(inst.get("label_asym_id"))
                residue = (
                    f"{sanitize_value(inst.get('auth_asym_id'))}"
                    f":{sanitize_value(inst.get('auth_seq_id'))}"
                )
                tokens.append((label, residue))
        return tokens

    def _norm_site(value: Any) -> str:
        text = sanitize_value(value)
        return "" if is_empty_key(text) else text.lower()

    # Group the SURVIVING ligand rows by component id so a compound's per-copy votes
    # can be split across its site rows (``unknown`` among them, an ordinary site).
    # Dropped rows are excluded as placement targets, but their sites are recorded
    # per component so a copy voted to a dropped row's site can follow it out
    # (rather than flood a surviving sibling row) -- distinct from an un-sited copy,
    # which lands on the compound's ``unknown`` row (or, absent one, its first row).
    rows_by_comp: dict[str, list[int]] = {}
    row_site: dict[int, str] = {}
    dropped_sites_by_comp: dict[str, set[str]] = {}
    for idx, lig in enumerate(ligands):
        if not isinstance(lig, dict):
            continue
        comp_id = sanitize_value(lig.get("chem_comp_id"))
        if is_empty_key(comp_id):
            continue  # keyless (peptide/glycan): no per-copy residues, as before
        if ligand_row_dropped(lig):
            dropped_sites_by_comp.setdefault(comp_id, set()).add(_norm_site(lig.get("site_ref")))
            continue
        row_site[idx] = _norm_site(lig.get("site_ref"))
        rows_by_comp.setdefault(comp_id, []).append(idx)

    def _join(pairs: list[tuple[str, str]]) -> tuple[str, str]:
        return (
            ", ".join(label for label, _residue in pairs),
            ", ".join(residue for _label, residue in pairs),
        )

    for comp_id, idxs in rows_by_comp.items():
        tokens = _tokens(comp_id)
        # Nothing to partition -> keep the full join on every surviving row of this
        # compound (byte-identical to the pre-feature behavior): no votes, a single
        # copy, or no instance data at all.
        if not have_votes or len(tokens) <= 1:
            joined = _join(tokens)
            for idx in idxs:
                results[idx] = joined
            continue

        existing_sites = {row_site[idx] for idx in idxs}
        dropped_sites = dropped_sites_by_comp.get(comp_id, ())
        by_site: dict[str, list[tuple[str, str]]] = {}
        homeless: list[tuple[str, str]] = []
        for label, residue in tokens:
            voted = _norm_site(votes[residue]) if residue in votes else ""
            if voted and voted in existing_sites:
                # A real site (``unknown`` included) with a surviving row: the copy
                # lands on it. An un-sited copy voted ``unknown`` lands on the
                # compound's ``unknown`` row the aggregator built for exactly these.
                by_site.setdefault(voted, []).append((label, residue))
            elif voted and voted in dropped_sites:
                # The copy voted to a site whose row was filtered out (a
                # non-functional / ghost / apo drop): it follows that row out
                # rather than flooding a surviving sibling row of the same compound.
                continue
            else:
                # A copy the aggregator did not route to a surviving site (an absent
                # vote, or legacy data with no ``unknown`` row): never dropped -- it
                # falls back to the compound's ``unknown`` row (or its first surviving
                # row when there is none), always with a clean residue token.
                homeless.append((label, residue))

        # Row that absorbs any residual un-sited copy: the compound's ``unknown`` row
        # when it has one, else its first surviving row.
        fallback_idx = next((idx for idx in idxs if row_site[idx] == SITE_REF_UNKNOWN), idxs[0])
        for idx in idxs:
            placed = list(by_site.get(row_site[idx], []))
            if idx == fallback_idx and homeless:
                placed.extend(homeless)
            results[idx] = _join(placed)

    return results


def _auxiliary_small_molecule_rows(
    pdb_id: str,
    ligands: list[Any],
    nonpolymer_instances: dict[str, Any],
) -> list[dict[str, str]]:
    """Rows for ``auxiliary_small_molecules.csv`` -- molecules that are NOT receptor
    ligands, located by the same columns as a ligands.csv row (ChainID + label_asym_id
    + Residue_seq_id copy tokens, the token being ``"<auth_asym_id>:<auth_seq_id>"``).
    ChainID here is the comma-joined set of every copy's author chain (a component can
    have copies on several chains); ligands.csv instead shows the single model-supplied
    chain. A molecule lives in exactly one file, so the two never disagree on one row.

    Two sources:
      * model-lane molecules the model judged auxiliary (``ligand_routed_to_aux``):
        a cofactor, or a detector-flagged candidate judged non-functional. Only a
        component with NO surviving ligands row is catalogued whole here; a
        component that still has a surviving functional row is left whole in
        ligands.csv (comp-level split), so its copies are never double-counted
        across the two files.
      * mechanical-lane molecules (ions / cofactors / glycans / detergents / matrix
        lipids) stripped before the model ever saw them, enumerated from the
        structure's nonpolymer roster.

    Each row lists every modelled copy of its component, comma-joined and 1:1 across
    the label_asym_id and Residue_seq_id columns.
    """

    def _columns(comp_id: str) -> tuple[str, str, str]:
        labels: list[str] = []
        residues: list[str] = []
        chains: list[str] = []
        for inst in nonpolymer_instances.get(comp_id) or []:
            if not (isinstance(inst, dict) and inst.get("label_asym_id")):
                continue
            auth_chain = sanitize_value(inst.get("auth_asym_id"))
            labels.append(sanitize_value(inst.get("label_asym_id")))
            residues.append(f"{auth_chain}:{sanitize_value(inst.get('auth_seq_id'))}")
            if auth_chain:
                chains.append(auth_chain)
        return ", ".join(labels), ", ".join(residues), ", ".join(sorted(set(chains)))

    surviving_comps = {
        sanitize_value(lig.get("chem_comp_id"))
        for lig in ligands
        if isinstance(lig, dict) and not ligand_row_dropped(lig)
    }

    rows: list[dict[str, str]] = []
    cataloged: set[str] = set()

    # (1) Model-lane auxiliary molecules (cofactor / non-functional candidate).
    for lig in ligands:
        if not isinstance(lig, dict) or not ligand_routed_to_aux(lig):
            continue
        comp_id = sanitize_value(lig.get("chem_comp_id"))
        if is_empty_key(comp_id) or comp_id in cataloged or comp_id in surviving_comps:
            continue
        cataloged.add(comp_id)
        # Unlike the mechanical lane (which skips a comp with no roster location), a
        # model-lane molecule is a real ligand the model annotated, so it is
        # catalogued even when the roster carries no per-copy location for it (ChainID
        # then falls back to the ligand's own chain); dropping it would lose an
        # annotated molecule.
        label_join, residue_join, chain_join = _columns(comp_id)
        # Function records WHY the molecule is auxiliary: a cofactor the model named as
        # such, versus an incidental structural molecule -- left blank, like the
        # mechanical lane, rather than mislabelled a cofactor.
        role_value = sanitize_value((lig.get("role") or {}).get("value"))
        rows.append(
            {
                "PDB": pdb_id,
                "ChainID": chain_join or sanitize_value(lig.get("chain_id")),
                "Name": comp_id,
                "Type": gpcrdb_aux_type_for(comp_id),
                "Function": "Cofactor" if role_value == "Cofactor" else "",
                "label_asym_id": label_join,
                "Residue_seq_id": residue_join,
            }
        )

    # (2) Mechanical-lane molecules stripped before the model, from the roster.
    for comp_id in nonpolymer_instances:
        if comp_id not in MECHANICAL_AUX or comp_id in cataloged or comp_id in surviving_comps:
            continue
        label_join, residue_join, chain_join = _columns(comp_id)
        if not label_join:
            continue  # no modelled copy carries a label: nothing to catalogue
        cataloged.add(comp_id)
        rows.append(
            {
                "PDB": pdb_id,
                "ChainID": chain_join,
                "Name": comp_id,
                "Type": gpcrdb_aux_type_for(comp_id),
                "Function": "",
                "label_asym_id": label_join,
                "Residue_seq_id": residue_join,
            }
        )

    return rows


def transform_for_csv(pdb_id: str, data: dict) -> dict[str, list[dict[str, str]]]:
    """Transform reviewed PDB data into CSV-ready row dictionaries.

    Applies scientific transformations via :mod:`logic`:
    * Multi-chain receptor truncation to primary protomer
    * Orphaned-ligand radar (warns when ligands sit on truncated chains)
    * ``label_asym_id`` mapping (auth_asym_id → PDB standard identifiers)
    * Structure note enrichment with oligomer annotations

    Returns a mapping of CSV filename → list of row dicts.
    """
    from gpcr_tools.csv_generator.logic import (
        apply_db_truncation,
        build_structure_note,
        collect_ligand_chains,
        map_label_asym_id,
        resolve_partner_protomer,
    )

    rows_map: dict[str, list[dict[str, str]]] = {fname: [] for fname in CSV_SCHEMA}

    s_info = data.get("structure_info") or {}
    r_info = data.get("receptor_info") or {}
    oligo = data.get("oligomer_analysis") or {}
    label_map = oligo.get("label_asym_id_map") or {}
    nonpolymer_instances = oligo.get("nonpolymer_instance_index") or {}

    receptor_chain = sanitize_value(r_info.get("chain_id"))
    receptor_uniprot = sanitize_value(r_info.get("uniprot_entry_name"))
    # NOTE: receptor_info.oligomeric_state is captured in the aggregated JSON but is
    # intentionally NOT yet exported to any CSV column or structure note. Adding it
    # requires the four-place CSV schema lockstep (CSV_SCHEMA in config.py,
    # transform_for_csv here, test_csv_writer.py, and the README output table); it is
    # deferred to a dedicated change so this stays a minimal addition.

    # ── Truncation + orphaned-ligand radar ─────────────────────────
    ligand_chains = collect_ligand_chains(data.get("ligands") or [])
    receptor_chain, receptor_uniprot, truncation_note = apply_db_truncation(
        receptor_chain,
        receptor_uniprot,
        oligo,
        ligand_chains,
    )

    # ── Structure note enrichment ──────────────────────────────────
    s_note = build_structure_note(s_info, oligo, truncation_note)

    # ── Dimer partner protomer (recorded, not dropped) ─────────────
    partner_uniprot, partner_chain = resolve_partner_protomer(oligo, receptor_chain)

    # ── structures.csv ─────────────────────────────────────────────
    rows_map["structures.csv"].append(
        {
            "PDB": pdb_id,
            "Receptor_UniProt": receptor_uniprot,
            "Method": sanitize_value(s_info.get("method")),
            "Resolution": sanitize_value(s_info.get("resolution")),
            "State": sanitize_value((s_info.get("state") or {}).get("value") or "").capitalize(),
            "ChainID": receptor_chain,
            "label_asym_id": map_label_asym_id(receptor_chain, label_map),
            "Partner_UniProt": partner_uniprot,
            "Partner_ChainID": partner_chain,
            "Note": s_note,
            "Date": sanitize_value(s_info.get("release_date")),
        }
    )

    # ── ligands.csv ────────────────────────────────────────────────
    # label_asym_id and Residue_seq_id are filtered per binding site from the
    # per-copy votes and built together, so a multi-site compound's rows each list
    # only their own copies and the two columns stay 1:1 copy-for-copy. Computed
    # once here over the SAME rows the loop below keeps, so a copy is never
    # attributed to a dropped row and lost (see _per_site_label_and_residue_columns).
    per_site_columns = _per_site_label_and_residue_columns(
        data.get("ligands") or [],
        nonpolymer_instances,
        data.get("ligand_copies"),
    )
    for idx, lig in enumerate(data.get("ligands") or []):
        # A ligand row filtered out before writing -- a non-dict entry, a GHOST
        # ligand the validator could not find (unless a curator kept it), an apo /
        # "no ligand" placeholder, or a dual-use molecule the model judged
        # non-functional -- is skipped. The SAME predicate drives the per-site
        # partition above, so a copy voted to a dropped row's site follows that
        # row OUT of the output (rather than flooding a surviving sibling row); an
        # un-sited copy instead lands cleanly on the compound's ``unknown`` row (or,
        # absent one, its first surviving row) -- never a residue mark.
        if ligand_row_dropped(lig):
            continue
        smiles = lig.get("SMILES_stereo") or lig.get("SMILES") or ""
        lig_chain = sanitize_value(lig.get("chain_id"))
        comp_id = sanitize_value(lig.get("chem_comp_id"))
        # A non-polymer ligand's label_asym_id is its OWN mmCIF instance label(s),
        # never routed through the polymer label_asym_id_map. Both label_asym_id and
        # Residue_seq_id come from the per-site partition: one modelled copy -> its
        # label; several -> only the copies voted to this row's site, comma-joined,
        # falling back to the full component-id join when there are no per-copy
        # votes. Each residue token is "<auth_asym_id>:<auth_seq_id>" (author chain :
        # author residue number), the same identifier used as the per-copy copy_id
        # and aligned copy-for-copy with label_asym_id.
        lig_label, lig_residue_seq = per_site_columns[idx]
        # The PDBe chemical-component id is the canonical ligand identity, so it
        # goes in Name; the descriptive free-text name goes in Title. When the
        # component id is empty/"None"/missing (e.g. a peptide or branched
        # glycan), fall back to the descriptive name rather than writing a blank
        # or the literal "None" — and if that descriptive name is itself empty/
        # "None", write an empty string instead of the sentinel.
        if is_empty_key(comp_id):
            lig_name = "" if is_empty_key(lig.get("name")) else sanitize_value(lig.get("name"))
        else:
            lig_name = comp_id
        rows_map["ligands.csv"].append(
            {
                "PDB": pdb_id,
                "ChainID": lig_chain,
                "label_asym_id": lig_label,
                "Name": lig_name,
                # Prefer the authoritative, structure-derived CID (matched from
                # the PDBe nonpolymer entity) over the model's self-reported id;
                # fall back to the model's id only when the authoritative one is
                # absent. The schema tells the model to emit the string "None"
                # when there is no PubChem id; normalize that sentinel to empty
                # rather than writing a literal "None" into the numeric column.
                "PubChemID": sanitize_value(lig.get("api_pubchem_cid"))
                if not is_empty_key(lig.get("api_pubchem_cid"))
                else (
                    ""
                    if is_empty_key(lig.get("pubchem_id"))
                    else sanitize_value(lig.get("pubchem_id"))
                ),
                "Role": sanitize_value((lig.get("role") or {}).get("value")),
                # A dual-role ligand the model split per site carries a site_ref
                # (e.g. orthosteric / allosteric); blank for an ordinary ligand.
                # This keeps the per-site rows distinct rather than duplicate.
                "Site": sanitize_value(lig.get("site_ref")),
                # Title is the descriptive free-text name, now distinct from Name
                # (which carries the canonical chemical-component id).
                "Title": sanitize_value(lig.get("name")),
                "Type": sanitize_value(lig.get("type")),
                "Date": sanitize_value(s_info.get("release_date")),
                "In structure": "",
                "SMILES": sanitize_value(smiles),
                "InChIKey": sanitize_value(lig.get("InChIKey")),
                "Sequence": sanitize_value(lig.get("Sequence")),
                "is_endogenous": sanitize_value(lig.get("is_endogenous")),
                "Residue_seq_id": lig_residue_seq,
            }
        )

    # ── auxiliary_small_molecules.csv ──────────────────────────────
    rows_map["auxiliary_small_molecules.csv"].extend(
        _auxiliary_small_molecule_rows(pdb_id, data.get("ligands") or [], nonpolymer_instances)
    )

    # ── g_proteins.csv ─────────────────────────────────────────────
    partners = data.get("signaling_partners") or {}
    if partners.get("g_protein"):
        gp = partners["g_protein"]
        alpha = gp.get("alpha_subunit") or {}
        alpha_chain = _primary_chain(alpha.get("chain_id"))
        beta_chain = _primary_chain((gp.get("beta_subunit") or {}).get("chain_id"))
        gamma_chain = _primary_chain((gp.get("gamma_subunit") or {}).get("chain_id"))
        rows_map["g_proteins.csv"].append(
            {
                "PDB": pdb_id,
                # Alpha_identity = the deposited/voted alpha slug, unchanged. The
                # alpha5 helix identity (Alpha_alpha5_identity) and the modelled
                # backbone scaffold (Alpha_backbone) are the distinct trailing columns below.
                "Alpha_identity": sanitize_value(alpha.get("uniprot_entry_name")),
                "Alpha_ChainID": alpha_chain,
                "Alpha_label_asym_id": map_label_asym_id(alpha_chain, label_map),
                "Beta_UniProt": sanitize_value(
                    (gp.get("beta_subunit") or {}).get("uniprot_entry_name")
                ),
                "Beta_ChainID": beta_chain,
                "Beta_label_asym_id": map_label_asym_id(beta_chain, label_map),
                "Gamma_UniProt": sanitize_value(
                    (gp.get("gamma_subunit") or {}).get("uniprot_entry_name")
                ),
                "Gamma_ChainID": gamma_chain,
                "Gamma_label_asym_id": map_label_asym_id(gamma_chain, label_map),
                "Note": sanitize_value(gp.get("note")),
                "Alpha_alpha5_identity": sanitize_value(alpha.get("functional_coupling")),
                "Alpha_backbone": sanitize_value(alpha.get("backbone")),
            }
        )

    # ── arrestins.csv ──────────────────────────────────────────────
    if partners.get("arrestin"):
        ar = partners["arrestin"]
        ar_chain = sanitize_value(ar.get("chain_id"))
        rows_map["arrestins.csv"].append(
            {
                "PDB": pdb_id,
                "UniProt": sanitize_value(ar.get("uniprot_entry_name")),
                "ChainID": ar_chain,
                "label_asym_id": map_label_asym_id(ar_chain, label_map),
                "Note": sanitize_value(ar.get("note")),
            }
        )

    # ── auxiliary protein CSVs ─────────────────────────────────────
    for aux in data.get("auxiliary_proteins") or []:
        target = AUX_PROTEIN_DISPATCH.get(
            (aux.get("type") or {}).get("value") or "Other",
            "other_aux_proteins.csv",
        )
        rows_map[target].append({"PDB": pdb_id, "Name": sanitize_value(aux.get("name"))})

    return rows_map


def append_to_csvs(csv_data_map: dict[str, list[dict[str, str]]]) -> None:
    """Append rows to the appropriate CSV files, creating them with headers if needed.

    A file with no rows for this batch is still created header-only, so the
    downstream build never hits a missing file (e.g. grk/ramp when a batch has no
    such entities). Files use LF line endings to match the consumed data.

    Performs a header migration check: if an existing file has outdated headers
    (e.g. missing ``label_asym_id`` columns), a CsvSchemaMismatchError is raised
    to prevent silent column misalignment.
    """
    from gpcr_tools.csv_generator.exceptions import CsvSchemaMismatchError

    cfg = get_config()
    csv_dir = cfg.csv_output_dir
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: validate the schema of every existing target file before
    # writing anything, to avoid partial writes (e.g. structures.csv written but
    # ligands.csv rejected). Checked even for empty inputs so a stale header is
    # caught rather than silently left behind.
    for filename in csv_data_map:
        filepath = csv_dir / filename
        if not filepath.exists():
            continue
        expected_fields = CSV_SCHEMA[filename]
        with open(filepath, encoding="utf-8") as f:
            existing_header = f.readline().strip().split("\t")
        if existing_header != list(expected_fields):
            raise CsvSchemaMismatchError(
                filename=filename,
                expected_fields=expected_fields,
                found_fields=existing_header,
            )

    for filename, rows in csv_data_map.items():
        filepath = csv_dir / filename
        expected_fields = CSV_SCHEMA[filename]

        # No rows for this file: still ensure a header-only file exists so the
        # downstream build never sees a missing file. Existing files are left
        # untouched (don't clobber data written for another PDB in this run).
        if not rows:
            if not filepath.exists():
                tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f, fieldnames=expected_fields, delimiter="\t", lineterminator="\n"
                    ).writeheader()
                os.replace(tmp_path, filepath)
            continue

        pdb_col = expected_fields[0]  # first column is the PDB id in every schema
        incoming_pdbs = {r.get(pdb_col) for r in rows}

        # Upsert, not blind append: drop any existing rows for the same PDB(s)
        # before writing, so re-curating a PDB replaces its rows instead of
        # accumulating duplicate, conflicting entries. Rewrite atomically.
        kept: list[dict[str, str]] = []
        if filepath.exists():
            with open(filepath, newline="", encoding="utf-8") as f:
                kept = [
                    row
                    for row in csv.DictReader(f, delimiter="\t")
                    if row.get(pdb_col) not in incoming_pdbs
                ]

        tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=expected_fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(kept)
            writer.writerows(rows)
        os.replace(tmp_path, filepath)
