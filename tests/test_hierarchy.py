"""
Tests for the taxonomy-building logic that do NOT need the real
cvjena/semantic-embeddings download, torch, or a GPU.

Run with: python -m pytest tests/test_hierarchy.py -v
"""

from src.hierarchy import _path_to_root


def test_path_to_root_simple_chain():
    parent_of = {1: 2, 2: 3}
    assert _path_to_root(1, parent_of) == [3, 2, 1]


def test_path_to_root_root_only():
    parent_of = {}
    assert _path_to_root(99, parent_of) == [99]


def test_path_to_root_branching_tree():
    parent_of = {2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
    assert _path_to_root(4, parent_of) == [1, 2, 4]
    assert _path_to_root(5, parent_of) == [1, 2, 5]
    assert _path_to_root(6, parent_of) == [1, 3, 6]

    def depth_and_ancestors(node):
        path = _path_to_root(node, parent_of)
        return len(path) - 1, set(path)

    def tree_distance(a, b):
        da, anc_a = depth_and_ancestors(a)
        db, anc_b = depth_and_ancestors(b)
        lca_depth = max(
            len(_path_to_root(x, parent_of)) - 1
            for x in anc_b if x in anc_a
        )
        return da + db - 2 * lca_depth

    assert tree_distance(4, 5) == 2
    assert tree_distance(4, 6) == 4
