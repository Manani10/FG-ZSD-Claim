"""
Smoke test for msa_hicl_loss: checks the loss is finite on a tiny
synthetic taxonomy and a small random batch.

Run with: python -m pytest tests/test_losses.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from dataclasses import dataclass

from src.losses import MomentumCache, msa_hicl_loss


@dataclass
class _FakeTaxonomy:
    class_to_path: dict
    node_index: dict
    dist_matrix: list


def _build_fake_taxonomy():
    class_to_path = {0: [0, 1, 3], 1: [0, 1, 4], 2: [0, 2, 5]}
    node_ids = [0, 1, 2, 3, 4, 5]
    node_index = {n: i for i, n in enumerate(node_ids)}
    depth = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}

    def ancestors(n):
        path, cur = [n], n
        while cur in parent:
            cur = parent[cur]
            path.append(cur)
        return set(path)

    size = len(node_ids)
    dist = [[0] * size for _ in range(size)]
    for a in node_ids:
        for b in node_ids:
            if a == b:
                continue
            anc_a, anc_b = ancestors(a), ancestors(b)
            lca_depth = max(depth[x] for x in anc_a if x in anc_b)
            dist[node_index[a]][node_index[b]] = depth[a] + depth[b] - 2 * lca_depth

    return _FakeTaxonomy(class_to_path=class_to_path, node_index=node_index, dist_matrix=dist)


def test_msa_hicl_loss_is_finite():
    taxonomy = _build_fake_taxonomy()
    cache = MomentumCache(taxonomy.node_index, feature_dim=8, momentum=0.99)
    features = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    loss, leaf_node_ids = msa_hicl_loss(features, labels, taxonomy, cache)
    assert torch.isfinite(loss)
    assert len(leaf_node_ids) == 4
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
