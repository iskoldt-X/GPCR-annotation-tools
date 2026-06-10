"""Gemini tool schema definition for GPCR structure annotation."""

from __future__ import annotations

from google.genai import types

ANNOTATION_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            **{  # type: ignore[arg-type]
                "name": "annotate_gpcr_db_structure",
                "description": "Extracts and structures all key information for a GPCR structure from a scientific paper and PDB metadata, preparing it for direct import into the GPCRdb. For any field requiring inference, confidence and evidence must be provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_info": {
                            "type": "object",
                            "description": "Core details about the PDB structure itself, primarily from PDB metadata.",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "description": "The experimental method used. Must be one of the specified values.",
                                    "enum": [
                                        "Electron crystallography",
                                        "Electron microscopy",
                                        "Refined CEM",
                                        "Refined X-ray",
                                        "X-ray diffraction",
                                    ],
                                },
                                "resolution": {
                                    "type": "number",
                                    "description": "Resolution in Angstroms.",
                                },
                                "release_date": {
                                    "type": "string",
                                    "description": "Initial release date (YYYY-MM-DD).",
                                },
                                "state": {
                                    "type": "object",
                                    "description": "Inferred functional state of the receptor, with confidence and evidence.",
                                    "properties": {
                                        "value": {
                                            "type": "string",
                                            "description": "The state value. Must be one of the specified enum values. Use 'unknown' if neither the paper nor the metadata establishes the functional state — do not guess.",
                                            "enum": [
                                                "inactive",
                                                "active",
                                                "other",
                                                "intermediate",
                                                "unknown",
                                            ],
                                        },
                                        "confidence": {
                                            "type": "string",
                                            "description": "Confidence level of this inference.",
                                            "enum": ["High", "Medium", "Low"],
                                        },
                                        "evidence": {
                                            "type": "object",
                                            "description": "The justification for the state assignment.",
                                            "properties": {
                                                "source": {
                                                    "type": "string",
                                                    "description": "The source of the evidence.",
                                                    "enum": [
                                                        "Paper",
                                                        "PDB Metadata",
                                                        "Both Paper and PDB Metadata",
                                                    ],
                                                },
                                                "quote_or_path": {
                                                    "type": "string",
                                                    "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                },
                                                "reasoning": {
                                                    "type": "string",
                                                    "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                },
                                            },
                                            "required": [
                                                "source",
                                                "quote_or_path",
                                                "reasoning",
                                            ],
                                        },
                                    },
                                    "required": ["value", "confidence", "evidence"],
                                },
                                "note": {
                                    "type": "string",
                                    "description": "Any specific notes, e.g., 'Fragment', 'Engineered'.",
                                },
                            },
                            "required": ["method", "resolution", "release_date", "state"],
                        },
                        "receptor_info": {
                            "type": "object",
                            "description": "Information about the main GPCR receptor protein.",
                            "properties": {
                                "uniprot_entry_name": {
                                    "type": "string",
                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                },
                                "chain_id": {
                                    "type": "string",
                                    "description": "The chain ID of the GPCR.",
                                },
                            },
                            "required": ["uniprot_entry_name", "chain_id"],
                        },
                        "ligands": {
                            "type": "array",
                            "description": "A list of ALL ligands. Must include an 'Apo' entry if no ligand is present. A G-protein-derived or transducer-mimetic peptide is a signaling partner, not a receptor ligand — do not record it here.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Common name of the ligand. Use 'Apo' for empty structures.",
                                    },
                                    "chain_id": {
                                        "type": "string",
                                        "description": "Chain ID, if applicable. Can be 'None'.",
                                    },
                                    "role": {
                                        "type": "object",
                                        "description": "Inferred pharmacological role of the ligand, with confidence and evidence.",
                                        "properties": {
                                            "value": {
                                                "type": "string",
                                                "description": "The role value. Must be one of the specified enum values.",
                                                "enum": [
                                                    "Antagonist",
                                                    "PAM",
                                                    "NAM",
                                                    "Ago-PAM",
                                                    "Allosteric antagonist",
                                                    "Apo (no ligand)",
                                                    "Cofactor",
                                                    "unknown",
                                                    "Inverse agonist",
                                                    "Agonist",
                                                    "Allosteric agonist",
                                                    "Co-agonist",
                                                    "Agonist (partial)",
                                                ],
                                            },
                                            "confidence": {
                                                "type": "string",
                                                "description": "Confidence level of this inference.",
                                                "enum": ["High", "Medium", "Low"],
                                            },
                                            "evidence": {
                                                "type": "object",
                                                "description": "The justification for the role assignment.",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "The source of the evidence.",
                                                        "enum": [
                                                            "Paper",
                                                            "PDB Metadata",
                                                            "Both Paper and PDB Metadata",
                                                        ],
                                                    },
                                                    "quote_or_path": {
                                                        "type": "string",
                                                        "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                    },
                                                    "reasoning": {
                                                        "type": "string",
                                                        "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                    },
                                                },
                                                "required": [
                                                    "source",
                                                    "quote_or_path",
                                                    "reasoning",
                                                ],
                                            },
                                        },
                                        "required": ["value", "confidence", "evidence"],
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Molecular type. prioritize info from 'gpcrdb_determined_type', but must be one of the specified values.",
                                        "enum": [
                                            "lipid",
                                            "na",
                                            "none",
                                            "peptide",
                                            "protein",
                                            "small-molecule",
                                        ],
                                    },
                                    "pubchem_id": {
                                        "type": "string",
                                        "description": "'gpcrdb_pubchem_cid' if available, otherwise try to find the correct PubChem ID in the paper. Use 'None' if not applicable.",
                                    },
                                    "chem_comp_id": {
                                        "type": "string",
                                        "description": "The chemical component ID (e.g., 'CAU', 'CLR') from the PDB metadata. Use 'None' for Apo entries.",
                                    },
                                    "synonyms": {
                                        "type": "array",
                                        "description": "List of synonyms from the PDB metadata. Can be an empty list.",
                                        "items": {"type": "string"},
                                    },
                                    "site_ref": {
                                        "type": "string",
                                        "description": (
                                            "The binding site of this ligand on the receptor. Infer it from the geometric facts in the "
                                            "DETECTOR EVIDENCE block plus the paper; use 'unknown' if neither settles it — do not guess. "
                                            "If one ligand is modelled at more than one distinct site, emit a separate entry per site. "
                                            "Boundaries of each value: "
                                            "'orthosteric' = the classic agonist core pocket where the endogenous/primary ligand binds. "
                                            "'extracellular_vestibule' = the vestibule just above the orthosteric pocket toward the extracellular end of the 7TM bundle, often adjacent to or overlapping the orthosteric site. "
                                            "'allosteric_7tm' = a non-orthosteric pocket embedded WITHIN the 7TM helical bundle / mid-bilayer (between helices, mid-membrane), NOT at either membrane end; keep this tight — a mid-bilayer positive allosteric modulator (e.g. an inter-helical TM3/4/5 pocket) is allosteric_7tm, not intracellular. "
                                            "'intracellular' = on the receptor cytoplasmic face / the receptor-transducer (G protein / arrestin) interface / around the DRY-Arg (3.50), helix 8, and the intracellular ends of TM3/5/6/7. EVEN IF the ligand is pharmacologically allosteric, if it sits on the cytoplasmic side it is intracellular, NOT allosteric_7tm. "
                                            "'extracellular_domain' = the extracellular domain (ECD), Venus-flytrap (VFT), or N-terminal domain, outside the 7TM bundle. "
                                            "'lipidic' = a structural lipid or detergent sitting against the membrane-facing (lipid-exposed) surface of the bundle."
                                        ),
                                        "enum": [
                                            "orthosteric",
                                            "allosteric_7tm",
                                            "extracellular_vestibule",
                                            "intracellular",
                                            "extracellular_domain",
                                            "lipidic",
                                            "unknown",
                                        ],
                                    },
                                    "site_ref_justification": {
                                        "type": "string",
                                        "description": "Brief free-text justification for the site_ref call: which geometric facts and/or paper statements support it. Optional; omit if the site is obvious or unknown.",
                                    },
                                },
                                "required": [
                                    "name",
                                    "chem_comp_id",
                                    "chain_id",
                                    "role",
                                    "type",
                                    "pubchem_id",
                                    "synonyms",
                                    "site_ref",
                                ],
                            },
                        },
                        "signaling_partners": {
                            "type": "object",
                            "description": "Container for all signaling partners. Can be empty.",
                            "properties": {
                                "g_protein": {
                                    "type": "object",
                                    "description": "G-protein heterotrimer details. Omit if not present.",
                                    "properties": {
                                        "alpha_subunit": {
                                            "type": "object",
                                            "properties": {
                                                "uniprot_entry_name": {
                                                    "type": "string",
                                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                                },
                                                "chain_id": {"type": "string"},
                                            },
                                            "required": ["uniprot_entry_name", "chain_id"],
                                        },
                                        "beta_subunit": {
                                            "type": "object",
                                            "properties": {
                                                "uniprot_entry_name": {
                                                    "type": "string",
                                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                                },
                                                "chain_id": {"type": "string"},
                                            },
                                            "required": [
                                                "uniprot_entry_name",
                                                "chain_id",
                                            ],
                                        },
                                        "gamma_subunit": {
                                            "type": "object",
                                            "properties": {
                                                "uniprot_entry_name": {
                                                    "type": "string",
                                                    "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                                },
                                                "chain_id": {"type": "string"},
                                            },
                                            "required": [
                                                "uniprot_entry_name",
                                                "chain_id",
                                            ],
                                        },
                                        "is_chimeric": {
                                            "type": "boolean",
                                            "description": "Set to true if the PDB metadata or paper indicates this is an engineered chimeric G protein.",
                                        },
                                        "note": {
                                            "type": "string",
                                            "description": "Any notes, e.g., 'Engineered G protein', 'Gs/Gi chimera'. If is_chimeric is true, briefly explain the chimera's composition here. SOURCING REQUIREMENT: the specific composition details (parent isoforms, species, chimera breakpoints) MUST be stated in the paper or PDB metadata. If the source does not state them, do NOT write them -- stay at the family level (e.g. 'Gs-family G-alpha; specific subtype/species not stated in paper').",
                                        },
                                    },
                                    "required": ["alpha_subunit"],
                                },
                                "arrestin": {
                                    "type": "object",
                                    "description": "Arrestin details. Omit if not present.",
                                    "properties": {
                                        "uniprot_entry_name": {
                                            "type": "string",
                                            "description": "UniProt entry name (e.g., 'opsd_bovin'). MUST be in all lowercase letters.",
                                        },
                                        "chain_id": {"type": "string"},
                                        "note": {"type": "string"},
                                    },
                                    "required": ["uniprot_entry_name", "chain_id"],
                                },
                            },
                        },
                        "auxiliary_proteins": {
                            "type": "array",
                            "description": "List of other non-GPCR, non-signaling proteins. Can be an empty list.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The specific, standardized name of the protein (e.g., 'T4-Lysozyme', 'Nanobody-35').",
                                    },
                                    "type": {
                                        "type": "object",
                                        "description": "The functional category of the protein, with confidence and evidence.",
                                        "properties": {
                                            "value": {
                                                "type": "string",
                                                "description": "The category value. Must be one of the specified enum values.",
                                                "enum": [
                                                    "Fusion protein",
                                                    "Nanobody",
                                                    "Antibody",
                                                    "scFv",
                                                    "Antibody fab fragment",
                                                    "GRK",
                                                    "RAMP",
                                                    "MRAP",
                                                    "DARPin",
                                                    "Other",
                                                ],
                                            },
                                            "confidence": {
                                                "type": "string",
                                                "description": "Confidence level of this classification.",
                                                "enum": ["High", "Medium", "Low"],
                                            },
                                            "evidence": {
                                                "type": "object",
                                                "description": "The justification for the type assignment.",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "The source of the evidence.",
                                                        "enum": [
                                                            "Paper",
                                                            "PDB Metadata",
                                                            "Both Paper and PDB Metadata",
                                                        ],
                                                    },
                                                    "quote_or_path": {
                                                        "type": "string",
                                                        "description": "A direct quote from the paper or a JSON path from the metadata.",
                                                    },
                                                    "reasoning": {
                                                        "type": "string",
                                                        "description": "A brief defense explaining why the evidence supports the conclusion.",
                                                    },
                                                },
                                                "required": [
                                                    "source",
                                                    "quote_or_path",
                                                    "reasoning",
                                                ],
                                            },
                                        },
                                        "required": ["value", "confidence", "evidence"],
                                    },
                                    "chain_id": {
                                        "type": "string",
                                        "description": "Chain ID(s). Can be comma-separated if multiple chains.",
                                    },
                                },
                                "required": ["name", "type", "chain_id"],
                            },
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of 2-3 novel and specific scientific insights from the paper's structural analysis.",
                        },
                    },
                    "required": [
                        "structure_info",
                        "receptor_info",
                        "ligands",
                        "key_findings",
                    ],
                },
            }
        )
    ]
)

# Optional per-ligand field injected by detect_orchestrator.build_tool_for_signals
# ONLY when an incidental-candidate signal is present (e.g. cholesterol, palmitate).
# It is never part of the base tool, so an ordinary structure's schema is unchanged.
PHARMACOLOGICAL_ROLE_CHECK_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    description=(
        "Only for a detector-flagged incidental-candidate molecule (e.g. cholesterol, "
        "palmitate): your judgment of whether it is a functional ligand or an "
        "incidental / structural component, with the evidence."
    ),
    properties={
        "is_functional_ligand": types.Schema(
            type=types.Type.BOOLEAN,
            description="True if it acts as a functional ligand; False if incidental / structural.",
        ),
        "confidence": types.Schema(type=types.Type.STRING, enum=["High", "Medium", "Low"]),
        "evidence": types.Schema(
            type=types.Type.STRING,
            description="A direct paper quote or brief reasoning supporting the judgment.",
        ),
    },
)

TOOL_CONFIG = types.GenerateContentConfig(
    tools=[ANNOTATION_TOOL],
    temperature=0.0,
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="ANY")  # type: ignore[arg-type]
    ),
)
