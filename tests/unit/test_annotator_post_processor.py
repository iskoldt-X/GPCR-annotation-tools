from gpcr_tools.aggregator.voting import get_majority_votes
from gpcr_tools.annotator.post_processor import (
    _is_signaling_partners_empty,
    _standardize_auxiliary_name,
    _unwrap_composite,
    is_meaningfully_empty,
    post_process_annotation,
)


def test_unwrap_composite():
    class DummyCompositeMap:
        def __init__(self, data):
            self.data = data

        def items(self):
            return self.data.items()

        def __iter__(self):
            return iter(self.data)

        def __getitem__(self, k):
            return self.data[k]

    # Python 3 map duck-typing mapping. Mapping check covers dicts.
    from collections.abc import Mapping, Sequence

    class DummyMapping(Mapping):
        def __init__(self, d):
            self.d = d

        def __getitem__(self, key):
            return self.d[key]

        def __iter__(self):
            return iter(self.d)

        def __len__(self):
            return len(self.d)

    class DummySequence(Sequence):
        def __init__(self, s):
            self.s = s

        def __getitem__(self, i):
            return self.s[i]

        def __len__(self):
            return len(self.s)

    # Test that a custom mapping converts to a python dict
    composite = DummyMapping({"key": DummySequence(["val"])})
    result = _unwrap_composite(composite)
    assert isinstance(result, dict)
    assert result == {"key": ["val"]}

    # Ensure native structures are preserved appropriately
    assert _unwrap_composite({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}


def test_is_meaningfully_empty():
    assert is_meaningfully_empty(None) is True
    assert is_meaningfully_empty("") is True
    assert is_meaningfully_empty([]) is True
    assert is_meaningfully_empty({}) is True
    assert is_meaningfully_empty("N/A") is True
    assert is_meaningfully_empty(" none ") is True
    assert is_meaningfully_empty("-") is True
    assert is_meaningfully_empty("missing") is True
    assert is_meaningfully_empty("not present") is True

    assert is_meaningfully_empty("Valid string") is False
    assert is_meaningfully_empty(["content"]) is False
    assert is_meaningfully_empty(0) is False  # 0 is not a string but falsey
    assert is_meaningfully_empty(False) is False


def test_is_signaling_partners_empty():
    assert _is_signaling_partners_empty({}) is True
    assert _is_signaling_partners_empty({"note": "They are missing"}) is True
    assert _is_signaling_partners_empty({"g_protein": {}}) is True
    assert _is_signaling_partners_empty({"g_protein": {"alpha": "gnas"}}) is False


def test_standardize_auxiliary_name():
    # BRIL collapse is gated on a fusion type.
    assert _standardize_auxiliary_name("BRIL", "Fusion protein") == "BRIL"
    assert _standardize_auxiliary_name("bRiL", "Fusion protein") == "BRIL"
    assert _standardize_auxiliary_name("T4-Lysozyme-BRIL fusion", "Fusion protein") == "BRIL"
    assert _standardize_auxiliary_name("cytochrome b562 RIL", "Fusion protein") == "BRIL"

    # Short RCSB spellings of the cytochrome-b562 fusion fold to the house name.
    assert _standardize_auxiliary_name("Soluble cytochrome b562", "Fusion protein") == "BRIL"
    assert _standardize_auxiliary_name("Cytochrome b562", "Fusion protein") == "BRIL"
    # No-space "Cytochrome b562RIL" (seen in older-model runs) folds via the
    # cytochrome substring — locked so the substring path keeps catching it.
    assert _standardize_auxiliary_name("Cytochrome b562RIL", "Fusion protein") == "BRIL"

    # Other crystallization fusions keep their own names — the BRIL collapse must
    # NOT reuse the broad fusion-keyword tuple (which also matches these).
    assert _standardize_auxiliary_name("T4-Lysozyme", "Fusion protein") == "T4-Lysozyme"
    assert (
        _standardize_auxiliary_name("Green fluorescent protein", "Fusion protein")
        == "Green fluorescent protein"
    )

    # Pfam term "Cytochrome c/b562" is a distinct family, not the BRIL fusion.
    assert _standardize_auxiliary_name("Cytochrome c/b562", "Fusion protein") == "Cytochrome c/b562"

    # Fail closed: a non-fusion binder named for its target is never flattened.
    assert _standardize_auxiliary_name("anti-BRIL Fab", "Antibody fab fragment") == "anti-BRIL Fab"
    # Missing / None type does not trigger the collapse — the name passes through.
    assert _standardize_auxiliary_name("cytochrome b562 RIL", None) == "cytochrome b562 RIL"
    assert _standardize_auxiliary_name("Soluble cytochrome b562") == "Soluble cytochrome b562"

    # Nanobody standardisation (independent of type gating)
    assert _standardize_auxiliary_name("Nb35", "Nanobody") == "Nanobody-35"
    assert _standardize_auxiliary_name("nb-12", "Nanobody") == "Nanobody-12"
    assert _standardize_auxiliary_name("Nanobody", "Nanobody") == "Nanobody"
    assert _standardize_auxiliary_name(None) is None


def test_post_process_annotation():
    # The real ``type`` shape is a nested dict ({"value": ...}), not a flat
    # string. The BRIL collapse reads ``type.value``; using the real shape is
    # what lets the cytochrome-b562 fusion fold to the house name.
    raw_response = {
        "receptor_info": {
            "uniprot_entry_name": "OPSD_BOVIN",
        },
        "signaling_partners": {"note": "None found", "g_protein": {}},
        "auxiliary_proteins": [
            {"name": "Nb6", "type": {"value": "Nanobody"}},
            {"name": "cytochrome b562 ril", "type": {"value": "Fusion protein"}},
        ],
    }

    result = post_process_annotation(raw_response)

    # 1. Lowercase receptor_info uniprot
    assert result["receptor_info"]["uniprot_entry_name"] == "opsd_bovin"

    # 2. empty signaling partners deleted
    assert "signaling_partners" not in result

    # 3. auxiliary proteins parsed
    assert result["auxiliary_proteins"][0]["name"] == "Nanobody-6"
    assert result["auxiliary_proteins"][1]["name"] == "BRIL"


def test_post_process_flat_type_string_does_not_trigger_bril_collapse():
    """A flat ``type`` string is the wrong shape and must not collapse to BRIL.

    The collapse reads ``type.value`` on a nested dict. A flat ``{"type":
    "Fusion"}`` carries no readable value, so it fails closed and the name is
    left untouched — guarding against the historical test-shape trap.
    """
    raw_response = {
        "auxiliary_proteins": [
            {"name": "cytochrome b562 ril", "type": "Fusion"},
        ],
    }
    result = post_process_annotation(raw_response)
    assert result["auxiliary_proteins"][0]["name"] == "cytochrome b562 ril"


def test_post_process_signaling_partners_lower_uniprot():
    raw_response = {
        "signaling_partners": {
            "g_protein": {"alpha_subunit": {"uniprot_entry_name": "GNAS_HUMAN"}},
            "arrestin": {"uniprot_entry_name": "ARRB1_HUMAN"},
        }
    }
    result = post_process_annotation(raw_response)
    assert (
        result["signaling_partners"]["g_protein"]["alpha_subunit"]["uniprot_entry_name"]
        == "gnas_human"
    )
    assert result["signaling_partners"]["arrestin"]["uniprot_entry_name"] == "arrb1_human"


def test_bril_name_collapse_merges_vote_group_across_runs():
    """Normalising cytochrome-b562 spellings collapses the fusion vote group.

    Grounded on the real per-run auxiliary entries seen for 9D3G (a chimeric
    construct with a BRIL fusion plus an anti-BRIL Fab / anti-Fab nanobody
    fiducial complex): across runs the model spells the fusion partner either
    "BRIL" or "Soluble cytochrome b562". Auxiliary entries are vote-grouped by
    ``name``, so before normalisation those spellings split into two groups.
    After post-processing folds the cytochrome-b562 fusion to the house name
    "BRIL", the fusion forms a single vote group whose elected ``type`` is
    unchanged ("Fusion protein"), while the binder groups (anti-BRIL Fab and the
    nanobody) stay separate and are never merged into BRIL.
    """
    # Per-run auxiliary rosters, mirroring the real 9D3G run shapes: the fusion
    # is named "BRIL" in some runs and "Soluble cytochrome b562" in others; the
    # Fab is a target-named binder; the nanobody binds the Fab.
    per_run_aux = [
        [
            {"name": "BRIL", "type": {"value": "Fusion protein"}},
            {"name": "anti-BRIL Fab", "type": {"value": "Antibody fab fragment"}},
            {"name": "anti-Fab Nanobody", "type": {"value": "Nanobody"}},
        ],
        [
            {"name": "Soluble cytochrome b562", "type": {"value": "Fusion protein"}},
            {"name": "anti-BRIL Fab", "type": {"value": "Antibody fab fragment"}},
            {"name": "anti-Fab Nanobody", "type": {"value": "Nanobody"}},
        ],
        [
            {"name": "Cytochrome b562", "type": {"value": "Fusion protein"}},
            {"name": "anti-BRIL Fab", "type": {"value": "Antibody fab fragment"}},
            {"name": "anti-Fab Nanobody", "type": {"value": "Nanobody"}},
        ],
    ]

    # Post-process each run exactly as the pipeline does, then vote.
    normalised_runs = [
        post_process_annotation({"auxiliary_proteins": run})["auxiliary_proteins"]
        for run in per_run_aux
    ]

    # Sanity: the three cytochrome-b562 spellings all became the house name.
    fusion_names = {run[0]["name"] for run in normalised_runs}
    assert fusion_names == {"BRIL"}
    # The binders are untouched (fail-closed gate).
    assert {run[1]["name"] for run in normalised_runs} == {"anti-BRIL Fab"}

    majority, _ = get_majority_votes(normalised_runs, path="auxiliary_proteins")

    by_name = {entry["name"]: entry for entry in majority}
    # Exactly one fusion group, one Fab group, one nanobody group — no split,
    # and the binders are NOT merged into BRIL.
    assert set(by_name) == {"BRIL", "anti-BRIL Fab", "anti-Fab Nanobody"}
    assert len(majority) == 3
    # The elected type for the collapsed fusion group is unchanged.
    assert by_name["BRIL"]["type"]["value"] == "Fusion protein"
    assert by_name["anti-BRIL Fab"]["type"]["value"] == "Antibody fab fragment"
    assert by_name["anti-Fab Nanobody"]["type"]["value"] == "Nanobody"
