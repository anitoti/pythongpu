"""
Desikan-Killiany region nomenclature for the 83-node structural connectome
(``data/DTI_A.mat``).

The empirical adjacency shipped in ``DTI_A.mat`` is an 83 x 83 symmetric binary
matrix. Eighty-three regions is the canonical cardinality of the Lausanne-2008
"scale 33" parcellation (Cammoun et al., 2012; Hagmann et al., 2008), which is
constructed directly from the FreeSurfer Desikan-Killiany cortical atlas
(``aparc``) augmented with the subcortical segmentation (``aseg``):

    34 cortical regions per hemisphere  x 2  = 68
     7 subcortical regions per hemisphere x 2 = 14
     1 brainstem                              =  1
                                             ----
                                               83

IMPORTANT — ORDERING ASSUMPTION
-------------------------------
``DTI_A.mat`` carries no region lookup table (the .mat exposes only the raw
variable ``A``), so the index -> label association below is an *assumed*
convention, not a fact read from the file. We adopt the standard FreeSurfer
``aparc+aseg`` enumeration in the layout most commonly distributed with the
83-node Lausanne connectome:

    indices  0..33  : left-hemisphere cortex   (FreeSurfer aparc order)
    indices 34..40  : left-hemisphere subcortex (FreeSurfer aseg order)
    index      41   : brainstem
    indices 42..48  : right-hemisphere subcortex (FreeSurfer aseg order)
    indices 49..82  : right-hemisphere cortex   (FreeSurfer aparc order)

If your ``DTI_A.mat`` was exported under a different convention (e.g. right
hemisphere first, or subcortex appended after both cortices), supply the true
lookup table via :func:`labels_from_file` and the numeric node indices remain
authoritative regardless. Downstream reporting always prints the integer index
alongside the label so the adjacency structure is never ambiguous.
"""

from __future__ import annotations

from pathlib import Path

# ── FreeSurfer Desikan-Killiany cortical parcels, in aparc annotation order ──
# (the order in which the 34 cortical labels appear in FreeSurferColorLUT /
#  the aparc.annot colour table; index 0 = "unknown" is excluded).
DK_CORTICAL: tuple[str, ...] = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
)

# ── FreeSurfer aseg subcortical structures, in canonical aseg order ──
DK_SUBCORTICAL: tuple[str, ...] = (
    "thalamus-proper",
    "caudate",
    "putamen",
    "pallidum",
    "hippocampus",
    "amygdala",
    "accumbens-area",
)


def _hemisphere_block(prefix: str) -> list[str]:
    cortex = [f"ctx-{prefix}-{name}" for name in DK_CORTICAL]
    subcortex = [f"{prefix}-{name}" for name in DK_SUBCORTICAL]
    return cortex, subcortex


def build_labels_83() -> list[str]:
    """
    Return the assumed 83-element Desikan-Killiany label vector, index-aligned
    to the rows/columns of ``DTI_A.mat`` under the convention documented in the
    module header.
    """
    lh_ctx, lh_sub = _hemisphere_block("lh")
    rh_ctx, rh_sub = _hemisphere_block("rh")
    labels = lh_ctx + lh_sub + ["brainstem"] + rh_sub + rh_ctx
    if len(labels) != 83:
        raise AssertionError(
            f"expected 83 Desikan-Killiany labels, assembled {len(labels)}")
    return labels


def labels_from_file(path: str | Path) -> list[str]:
    """
    Load an explicit index -> label lookup table (one label per line, ordered by
    node index). Use this to override the assumed ordering when the true region
    nomenclature for a given ``DTI_A.mat`` is known.
    """
    path = Path(path)
    labels = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return labels


def labels_for(n_nodes: int, path: str | Path | None = None) -> list[str]:
    """
    Best-effort label vector for an ``n_nodes``-region connectome.

    Returns the Desikan-Killiany assumption for the canonical 83-node
    parcellation (or an explicit table loaded from ``path``); falls back to
    bare ``node_{i}`` placeholders when the node count is not the 83-region
    scale-33 cardinality and no override table is supplied.
    """
    if path is not None:
        labels = labels_from_file(path)
        if len(labels) != n_nodes:
            raise ValueError(
                f"label file has {len(labels)} entries, adjacency has {n_nodes} nodes")
        return labels
    if n_nodes == 83:
        return build_labels_83()
    return [f"node_{i}" for i in range(n_nodes)]
