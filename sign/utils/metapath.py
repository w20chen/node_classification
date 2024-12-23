from collections import defaultdict

import dgl
import torch

__all__ = ['metapath_based_graph']


def metapath_based_graph(g, metapath, nids=None, edge_feat_name='inst'):
    """Returns a graph composed of neighbors based on a given metapath in a heterogeneous graph. 
    Metapath instances are used as edge features. If the metapath is symmetric, a homogeneous graph is returned; otherwise, a bipartite graph is returned.

    Difference from dgl.metapath_reachable_graph(): If there are multiple metapath instances between two vertices, multiple edges are present in the returned graph.

    :param g: DGLGraph Heterogeneous graph
    :param metapath: List[str or (str, str, str)] Metapath, list of edge types
    :param nids: tensor(N), optional Starting node ids, if None, select all nodes of that type
    :param edge_feat_name: str Name of the edge feature to store metapath instances
    :return: DGLGraph Graph based on the metapath
    """
    instances = metapath_instances(g, metapath, nids)
    src_nodes, dst_nodes = instances[:, 0], instances[:, -1]
    src_type, dst_type = g.to_canonical_etype(metapath[0])[0], g.to_canonical_etype(metapath[-1])[2]
    if src_type == dst_type:
        mg = dgl.graph((src_nodes, dst_nodes), num_nodes=g.num_nodes(src_type))
        mg.edata[edge_feat_name] = instances
    else:
        mg = dgl.heterograph(
            {(src_type, '_E', dst_type): (src_nodes, dst_nodes)},
            {src_type: g.num_nodes(src_type), dst_type: g.num_nodes(dst_type)}
        )
        mg.edges['_E'].data[edge_feat_name] = instances
    return mg


def metapath_instances(g, metapath, nids=None):
    """Returns all instances of a given metapath in a heterogeneous graph.

    :param g: DGLGraph Heterogeneous graph
    :param metapath: List[str or (str, str, str)] Metapath, list of edge types
    :param nids: tensor(N), optional Starting node ids, if None, select all nodes of that type
    :return: tensor(E, L) E is the number of metapath instances, L is the length of the metapath
    """
    src_type = g.to_canonical_etype(metapath[0])[0]
    if nids is None:
        nids = g.nodes(src_type)
    paths = nids.unsqueeze(1).tolist()
    for etype in metapath:
        new_paths = []
        neighbors = etype_neighbors(g, etype)
        for path in paths:
            for neighbor in neighbors[path[-1]]:
                new_paths.append(path + [neighbor])
        paths = new_paths
    return torch.tensor(paths, dtype=torch.long)


def etype_neighbors(g, etype):
    """Returns neighbors based on a given edge type in a heterogeneous graph.

    :param g: DGLGraph Heterogeneous graph
    :param etype: (str, str, str) Canonical edge type
    :return: Dict[int, List[int]] Neighbors of each source vertex based on this type of edge
    """
    adj = g.adj(scipy_fmt='coo', etype=etype)
    neighbors = defaultdict(list)
    for u, v in zip(adj.row, adj.col):
        neighbors[u].append(v)
    return neighbors


def to_ntype_list(g, metapath):
    """Converts a metapath represented by a list of edge types to a list of node types.

    Example: ['ap', 'pc', 'cp', 'pa'] -> ['a', 'p', 'c', 'p', 'a']

    :param g: DGLGraph Heterogeneous graph
    :param metapath: List[str or (str, str, str)] Metapath, list of edge types
    :return: List[str] Node type list representation of the metapath
    """
    metapath = [g.to_canonical_etype(etype) for etype in metapath]
    return [metapath[0][0]] + [etype[2] for etype in metapath]


def metapath_instance_feat(metapath, node_feats, instances):
    """Returns features of metapath instances, composed of features of intermediate nodes.

    :param metapath: List[str] Metapath, list of node types
    :param node_feats: Dict[str, tensor(N_i, d)] Mapping from node type to node features, features of all types of nodes should have the same dimension d
    :param instances: tensor(E, L) Metapath instances, E is the number of metapath instances, L is the length of the metapath
    :return: tensor(E, L, d) Features of metapath instances
    """
    feat_dim = node_feats[metapath[0]].shape[1]
    inst_feat = torch.zeros(instances.shape + (feat_dim,))
    for i, ntype in enumerate(metapath):
        inst_feat[:, i] = node_feats[ntype][instances[:, i]]
    return inst_feat


def metapath_adj(g, metapath):
    """Returns the adjacency matrix of the starting and ending node types connected by a given metapath.

    :param g: DGLGraph
    :param metapath: List[str or (str, str, str)] Metapath, list of edge types
    :return: scipy.sparse.csr_matrix
    """
    adj = 1
    for etype in metapath:
        adj *= g.adj(etype=etype, scipy_fmt='csr')
    return adj
