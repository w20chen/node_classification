import torch
from dgl.sampling import random_walk
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = ['metapath_random_walk']


def metapath_random_walk(g, metapaths, num_walks, walk_length, output_file):
    """Random walk based on metapath

    :param g: DGLGraph Heterogeneous graph
    :param metapaths: Dict[str, List[str]] Mapping from node type to metapath
    :param num_walks: int Number of walks per node
    :param walk_length: int Number of times the metapath is repeated
    :param output_file: str Output file name
    :return:
    """
    f = open(output_file, 'w')
    for ntype, metapath in metapaths.items():
        print(ntype)
        loader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=200)
        for b in tqdm(loader, ncols=80):
            nodes = torch.repeat_interleave(b, num_walks)
            traces, types = random_walk(g, nodes, metapath=metapath * walk_length)
            f.writelines([trace2name(g, trace, types) + '\n' for trace in traces])
    f.close()


def trace2name(g, trace, types):
    return ' '.join(g.ntypes[t] + '_' + str(int(n)) for n, t in zip(trace, types) if int(n) >= 0)
