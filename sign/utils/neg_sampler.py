import random

import torch
from dgl.dataloading.negative_sampler import _BaseNegativeSampler

__all__ = ['RatioNegativeSampler']


class RatioNegativeSampler(_BaseNegativeSampler):

    def __init__(self, neg_sample_ratio=1.0):
        """Negative sampler that samples negative edges based on a certain ratio of positive edges

        :param neg_sample_ratio: float, optional The (approximate) ratio of the number of negative edges to positive edges, default is 1
        """
        self.neg_sample_ratio = neg_sample_ratio

    def _generate(self, g, eids, canonical_etype):
        stype, _, dtype = canonical_etype
        num_src_nodes, num_dst_nodes = g.num_nodes(stype), g.num_nodes(dtype)
        total = num_src_nodes * num_dst_nodes

        # Handle the case where |Vs×Vd-E|<r|E|
        num_neg_samples = min(int(self.neg_sample_ratio * len(eids)), total - len(eids))
        # Sample more to ensure uniqueness
        alpha = abs(1 / (1 - 1.1 * len(eids) / total))

        # Convert edges to indices ranging from 0 to |Vs||Vd|-1
        src, dst = g.find_edges(eids, etype=canonical_etype)
        idx = set((src * num_dst_nodes + dst).tolist())

        neg_idx = set(random.sample(range(total), min(int(alpha * num_neg_samples), total))) - idx
        neg_idx = torch.tensor(list(neg_idx))[:num_neg_samples]
        return neg_idx // num_dst_nodes, neg_idx % num_dst_nodes
