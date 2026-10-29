
import torch
import torch.nn.functional as F


class MomentumCache:
    """One prototype per taxonomy node, EMA-updated towards whatever
    features currently map to that node."""

    def __init__(self, node_index: dict, feature_dim: int, momentum: float = 0.99, device: str = "cpu"):
        self.node_index = node_index
        self.momentum = momentum
        num_nodes = len(node_index)
        protos = torch.randn(num_nodes, feature_dim, device=device)
        self.prototypes = F.normalize(protos, dim=1)

    @torch.no_grad()
    def update(self, node_ids: "list[int]", features: torch.Tensor) -> None:
        """features: (N, D) detached image features belonging to each node_id."""
        features = F.normalize(features, dim=1)
        for node_id, feat in zip(node_ids, features):
            idx = self.node_index[node_id]
            self.prototypes[idx] = (
                self.momentum * self.prototypes[idx] + (1 - self.momentum) * feat
            )
            self.prototypes[idx] = F.normalize(self.prototypes[idx], dim=0)

    def to(self, device):
        self.prototypes = self.prototypes.to(device)
        return self


def msa_hicl_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    taxonomy,
    cache: MomentumCache,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    features: (B, D) backbone features (BEFORE the classifier head).
    labels:   (B,) 0-indexed CUB class ids.
    taxonomy: a hierarchy.Taxonomy (class_to_path, dist_matrix, node_index).
    cache:    a MomentumCache sharing the same node_index.

    Returns a scalar loss, averaged over the batch and over each sample's
    root-to-leaf levels.
    """
    device = features.device
    feats = F.normalize(features, dim=1)
    protos = cache.prototypes.to(device)  # (num_nodes, D)
    dist = torch.tensor(taxonomy.dist_matrix, device=device, dtype=torch.float32)

    batch_losses = []
    all_node_ids = []  # for the momentum-cache update after this function returns

    for i in range(feats.size(0)):
        label = int(labels[i].item())
        path = taxonomy.class_to_path[label]  # root ... leaf, node ids
        num_levels = len(path)

        level_losses = []
        for level, node_id in enumerate(path):
            pos_idx = taxonomy.node_index[node_id]
            pos_sim = torch.dot(feats[i], protos[pos_idx]) / temperature

            # push: every OTHER node in the tree, weighted by how far away
            # (biologically) that node is from this one -- distant relatives
            # get pushed harder.
            node_ids_all = list(taxonomy.node_index.keys())
            other_idx = [taxonomy.node_index[n] for n in node_ids_all if n != node_id]
            other_sims = (feats[i] @ protos[torch.tensor(other_idx, device=device)].T) / temperature
            d = dist[taxonomy.node_index[node_id]][torch.tensor(other_idx, device=device)]
            push_weight = torch.exp(-d)  # closer relatives contribute less repulsion

            # log-sum-exp over [positive, weighted negatives] for stability
            logits = torch.cat([pos_sim.unsqueeze(0), other_sims + torch.log(push_weight + 1e-8)])
            log_denom = torch.logsumexp(logits, dim=0)
            level_loss = -(pos_sim - log_denom)

            # phi: linear-in-depth weighting (paper's ablation winner, Sec. 5.3)
            phi = (level + 1) / num_levels
            level_losses.append(phi * level_loss)

        batch_losses.append(torch.stack(level_losses).sum() / num_levels)
        all_node_ids.append(path[-1])  # update the leaf prototype for this sample

    loss = torch.stack(batch_losses).mean()
    return loss, all_node_ids
