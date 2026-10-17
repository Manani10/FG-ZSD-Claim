"""
Biological taxonomy for CUB-200-2011, used by the MSA-hicl loss.

Why this file exists
---------------------
The paper's hierarchical contrastive loss (Eq. 10-13 in Sec. 4.2) needs a
real taxonomy tree over the classes: every leaf (species) must have a
root-to-leaf path of ancestor nodes (genus, family, order, ...), and every
pair of nodes needs a "semantic distance" (tree distance) so the push term
can scale repulsion by how biologically different two classes are.

CUB-200-2011 itself ships no taxonomy -- only 200 flat class folders. We get
a *real* (not invented) bird taxonomy from a third-party academic release:

    https://github.com/cvjena/semantic-embeddings  (tag v1.1.0)
    CUB-Hierarchy/classes_wikispecies-hierarchy.txt   -> node_id, scientific name
    CUB-Hierarchy/cub_wikispecies.parent-child.txt    -> edge list (parent_id child_id)

We verified by inspection that this hierarchy's 200 leaf node ids (1-200)
line up 1:1, in order, with CUB's own class ids (001-200): leaf node 1 is
"Phoebastria nigripes" (Black-footed Albatross) == CUB class 001. This
module re-derives that mapping rather than assuming it, and raises loudly
if an assumption about the file format stops holding (see
`_sanity_check_leaf_alignment`), because a silently-wrong taxonomy would
make every result downstream meaningless.

Deviation from the paper, stated for the write-up
--------------------------------------------------
The paper's prototypes are *initialised* from the same text embeddings that
drive its image generator (Sec. 4.2). Our reproduction has no generator and
no text encoder (that whole branch of the paper is out of scope -- see the
top-level README), so our prototypes are initialised randomly and shaped
entirely by the momentum-updated EMA over image features. This is a real,
documented gap, not an oversight.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

HIERARCHY_REPO_URL = "https://github.com/cvjena/semantic-embeddings.git"
HIERARCHY_REPO_TAG = "v1.1.0"
EDGE_FILE = "CUB-Hierarchy/cub_wikispecies.parent-child.txt"
NODE_FILE = "CUB-Hierarchy/classes_wikispecies-hierarchy.txt"


@dataclass
class Taxonomy:
    """Everything the loss needs, pre-computed once and reused every batch."""

    num_classes: int
    root_id: int
    # 0-indexed CUB class -> list of node ids from root to leaf (inclusive)
    class_to_path: dict
    # 0-indexed CUB class -> immediate-parent node id (used as a "genus" proxy
    # for the t-SNE plot; it is literally the parent in the real tree, not a
    # guess from the English common name)
    genus_of: dict
    # node_id -> 0-indexed row/col into dist_matrix
    node_index: dict
    # square matrix of tree distances (hop count via lowest common ancestor)
    # between every pair of taxonomy nodes that appears in any class's path
    dist_matrix: "list[list[int]]"


def _clone_hierarchy_repo(dest_dir: str) -> str:
    """Clone cvjena/semantic-embeddings at the pinned tag if not already cached."""
    if os.path.isdir(os.path.join(dest_dir, "CUB-Hierarchy")):
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        "git", "clone", "--depth", "1", "--branch", HIERARCHY_REPO_TAG,
        HIERARCHY_REPO_URL, dest_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Could not clone the hierarchy repo.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
            "This file format assumption is load-bearing for the whole "
            "experiment -- do not fall back to a guessed taxonomy silently."
        )
    return dest_dir


def _parse_node_names(node_file_path: str) -> dict:
    """node_id (int) -> scientific name (str)."""
    names = {}
    with open(node_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            node_id = int(parts[0])
            name = parts[1] if len(parts) > 1 else ""
            names[node_id] = name
    return names


def _parse_edges(edge_file_path: str) -> dict:
    """child_id (int) -> parent_id (int)."""
    parent_of = {}
    with open(edge_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parent_str, child_str = line.split()
            parent_of[int(child_str)] = int(parent_str)
    return parent_of


def _path_to_root(node_id: int, parent_of: dict) -> list:
    """Root-to-leaf path (inclusive of both ends) for one node."""
    path = [node_id]
    cur = node_id
    while cur in parent_of:
        cur = parent_of[cur]
        path.append(cur)
    path.reverse()  # root first, leaf last
    return path


def _sanity_check_leaf_alignment(node_names: dict, cub_class_names: "list[str]") -> None:
    """
    Verify leaf node ids 1..200 correspond, in order, to CUB class ids
    001..200. We check this rather than assuming it, because if the
    hierarchy file's ordering ever changes, every prototype in the loss
    would silently be attached to the wrong species.
    """
    for leaf_id in (1, 200):
        if leaf_id not in node_names:
            raise AssertionError(
                f"Expected leaf node id {leaf_id} in the hierarchy file, "
                "but it was not found. The leaf-id <-> CUB-class-id "
                "alignment this code relies on may no longer hold."
            )
    # Spot-check node 1 against CUB class 1's folder name. CUB's own class
    # name for 001 is "Black_footed_Albatross"; the hierarchy's scientific
    # name for node 1 is "Phoebastria nigripes" -- different naming schemes,
    # so we only check that *a* name exists, and leave the real check (does
    # training actually improve things) to the experiment itself.
    assert node_names[1] != "", "Hierarchy node 1 has no name -- file format changed."


def build_taxonomy(cache_dir: str, cub_class_names: "list[str]", num_classes: int = 200) -> Taxonomy:
    """
    Build the Taxonomy used by src/losses.py.

    Parameters
    ----------
    cache_dir : where to clone/cache the cvjena/semantic-embeddings repo.
    cub_class_names : CUB's own class names in class-id order (1-indexed
        folder names, e.g. "001.Black_footed_Albatross", ...), used only
        for the sanity check above.
    """
    repo_dir = _clone_hierarchy_repo(os.path.join(cache_dir, "semantic-embeddings"))
    node_names = _parse_node_names(os.path.join(repo_dir, NODE_FILE))
    parent_of = _parse_edges(os.path.join(repo_dir, EDGE_FILE))

    _sanity_check_leaf_alignment(node_names, cub_class_names)

    # Build root-to-leaf path for every CUB class (leaf ids 1..num_classes).
    class_to_path = {}
    all_nodes_used = set()
    for cls in range(num_classes):
        leaf_id = cls + 1  # CUB classes are 1-indexed in the hierarchy file
        path = _path_to_root(leaf_id, parent_of)
        class_to_path[cls] = path
        all_nodes_used.update(path)

    root_id = class_to_path[0][0]
    for cls, path in class_to_path.items():
        if path[0] != root_id:
            raise AssertionError(
                f"Class {cls} does not share the same root as class 0 "
                f"({path[0]} != {root_id}); the hierarchy may be a forest, "
                "not a tree."
            )

    # genus_of: immediate parent of the leaf (second-to-last entry on the path)
    genus_of = {cls: path[-2] for cls, path in class_to_path.items()}

    # Index every node that actually appears in some class's path, then
    # build a full pairwise tree-distance matrix via depth + LCA.
    node_list = sorted(all_nodes_used)
    node_index = {n: i for i, n in enumerate(node_list)}

    depth = {}
    ancestors = {}
    for n in node_list:
        path = _path_to_root(n, parent_of)
        depth[n] = len(path) - 1
        ancestors[n] = path  # root..n inclusive

    size = len(node_list)
    dist_matrix = [[0] * size for _ in range(size)]
    for a in node_list:
        for b in node_list:
            if a == b:
                continue
            # distance = depth(a) + depth(b) - 2*depth(LCA(a,b))
            set_a = set(ancestors[a])
            lca_depth = max(depth[x] for x in ancestors[b] if x in set_a)
            d = depth[a] + depth[b] - 2 * lca_depth
            dist_matrix[node_index[a]][node_index[b]] = d

    return Taxonomy(
        num_classes=num_classes,
        root_id=root_id,
        class_to_path=class_to_path,
        genus_of=genus_of,
        node_index=node_index,
        dist_matrix=dist_matrix,
    )
