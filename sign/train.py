import argparse

import dgl.function as fn
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CoraGraphDataset, RedditDataset, CiteseerGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.nodeproppred import Evaluator

from model import SIGN
from utils import set_random_seed, accuracy


def load_data(dataset, ogb_root):
    if dataset in ('cora', 'reddit', 'citeseer'):
        data = None
        if dataset == 'cora':
            data = CoraGraphDataset()
        elif dataset == 'reddit':
            data = RedditDataset(self_loop=True)
        else:
            data = CiteseerGraphDataset()
        g = data[0]
        train_idx = g.ndata['train_mask'].nonzero(as_tuple=True)[0]
        val_idx = g.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_idx = g.ndata['test_mask'].nonzero(as_tuple=True)[0]
        return g, g.ndata['label'], data.num_classes, train_idx, val_idx, test_idx
    else:
        data = DglNodePropPredDataset('ogbn-products', ogb_root)
        g, labels = data[0]
        split_idx = data.get_idx_split()
        return g, labels.squeeze(dim=-1), data.num_classes, \
            split_idx['train'], split_idx['valid'], split_idx['test']


def calc_weight(g):
    """Calculate row-normalized D^(-1/2)AD(-1/2)"""
    with g.local_scope():
        g.ndata['in_degree'] = g.in_degrees().float().pow(-0.5)
        g.ndata['out_degree'] = g.out_degrees().float().pow(-0.5)
        g.apply_edges(fn.u_mul_v('out_degree', 'in_degree', 'weight'))
        g.update_all(fn.copy_e('weight', 'msg'), fn.sum('msg', 'norm'))
        g.apply_edges(fn.e_div_v('weight', 'norm', 'weight'))
        return g.edata['weight']


def preprocess(g, features, num_hops):
    """Precompute r-hop neighbor aggregated features"""
    with torch.no_grad():
        g.edata['weight'] = calc_weight(g)
        g.ndata['feat_0'] = features
        for h in range(1, num_hops + 1):
            g.update_all(fn.u_mul_e(f'feat_{h - 1}', 'weight', 'msg'), fn.sum('msg', f'feat_{h}'))
        return [g.ndata.pop(f'feat_{h}') for h in range(num_hops + 1)]


def train(args):
    set_random_seed(args.seed)
    g, labels, num_classes, train_idx, val_idx, test_idx = load_data(args.dataset, args.ogb_root)
    print('Precomputing neighbor aggregated features...')
    features = preprocess(g, g.ndata['feat'], args.num_hops)  # List[tensor(N, d_in)], length is r+1
    train_feats = [feat[train_idx] for feat in features]
    val_feats = [feat[val_idx] for feat in features]
    test_feats = [feat[test_idx] for feat in features]

    model = SIGN(
        g.ndata['feat'].shape[1], args.num_hidden, num_classes, args.num_hops,
        args.num_layers, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        logits = model(train_feats)
        loss = F.cross_entropy(logits, labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, labels[train_idx])
        val_acc = evaluate(model, val_feats, labels[val_idx])
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss, train_acc, val_acc
        ))

    # test
    test_acc = evaluate(model, test_feats, labels[test_idx])
    print('Test Acc {:.4f}'.format(test_acc))

    # official evaluation
    evaluator = Evaluator(name='ogbn-products')

    print(evaluator.eval({
        'y_true': labels[test_idx].reshape(-1, 1),
        'y_pred': model(test_feats).argmax(dim=-1).reshape(-1, 1)
    }))


def evaluate(model, feats, labels):
    model.eval()
    with torch.no_grad():
        logits = model(feats)
    return accuracy(logits, labels)


def main():
    parser = argparse.ArgumentParser(description='SIGN')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--dataset', choices=['cora', 'reddit', 'citeseer', 'ogbn-products'], default='cora', help='dataset'
    )
    parser.add_argument('--ogb-root', default='./ogb', help='root directory to OGB datasets')
    parser.add_argument('--num-hidden', type=int, default=256, help='number of hidden units')
    parser.add_argument('--num_hops', type=int, default=3, help='number of hops')
    parser.add_argument('--num-layers', type=int, default=2, help='number of feed-forward layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0., help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()