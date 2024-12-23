import random

import numpy as np
import torch
import scipy.sparse as sp


class ogbData:
    def __init__(self, path_of_ogb):
        self.ogb_path = path_of_ogb

        self.feature_of_pg = []  # List to store node features
        self.label_of_pg = []    # List to store node labels
        self.edge_of_pg = []     # List to store edges
        self.label_set = set()   # Set to store unique labels

        self.__dataset_loader()  # Load the dataset
        self.num_nodes = len(self.feature_of_pg)  # Number of nodes
        self.num_edges = len(self.edge_of_pg)     # Number of edges (twice the number of directed edges)
        self.num_of_class = len(self.label_set)   # Number of unique classes
        print(f'# of label: {len(self.label_set)}')
        self.feature_dim = len(self.feature_of_pg[0])  # Dimension of node features
        print(f'feature dim: {self.feature_dim}')

    # Load ogb data
    def __dataset_loader(self):
        path_cites = self.ogb_path + "/edge.csv"
        path_contents = self.ogb_path + "/node-feat.csv"
        path_label = self.ogb_path + "/node-label.csv"

        with open(path_label, 'r', encoding='utf-8') as file_label:
            for line in file_label.readlines():
                label = int(line)
                self.label_of_pg.append(label)
                if label not in self.label_set:
                    self.label_set.add(label)

        print('Node label loaded')

        with open(path_contents, 'r', encoding='utf-8') as file_content:
            for node in file_content.readlines():
                node_cont = node.split(',')
                self.feature_of_pg.append([float(i) for i in node_cont])

        print('Node feature loaded')

        with open(path_cites, 'r', encoding='utf-8') as file_cite:
            for edge in file_cite.readlines():
                cited, citing = edge.split(',')
                edge_1 = [int(citing), int(cited)]
                edge_2 = [int(cited), int(citing)]
                self.edge_of_pg.append(edge_1)
                self.edge_of_pg.append(edge_2)

        print('Dataset loaded')

    # Create a sparse adjacency matrix
    def get_adjacent(self):
        graph_w = np.ones(self.num_edges)  # Weights for edges
        np_edge = np.array(self.edge_of_pg)
        # Create a sparse adjacency matrix in COO format
        adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # Get a random adjacency matrix with some edges dropped
    def random_adjacent_sampler(self, drop_edge=0.1):
        new_edge_of_pg = []
        half_edge_num = int(len(self.edge_of_pg)/2)
        sampler = np.random.rand(half_edge_num)
        for i in range(half_edge_num):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(self.edge_of_pg[2 * i])
                new_edge_of_pg.append(self.edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # Normalize adjacency matrix A with D^{-1/2}AD^{-1/2}, self-loop optional
    @staticmethod
    def normalization(adj, self_link=True):
        adj = sp.coo_matrix(adj)
        if self_link:
            adj += sp.eye(adj.shape[0])  # Add self-loops
        row_sum = np.array(adj.sum(1))   # Sum of each row, i.e., degree of each node
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_hat = sp.diags(d_inv_sqrt)
        return d_hat.dot(adj).dot(d_hat).tocoo()   # Return in COO format

    def putout_data(self):
        print(f"the shape of feature is {len(self.feature_of_pg),len(self.feature_of_pg[0])}")
        print(f"the shape of label is {len(self.label_of_pg)}")
        print(f"the shape of edge is {len(self.edge_of_pg),len(self.edge_of_pg[0])}")

    # Partition data into training, validation, and test sets
    @staticmethod
    def data_partition_node(data_size=2449029):
        mask = torch.randperm(data_size)
        train_mask = mask[:1000]
        val_mask = mask[196615:197615]
        test_mask = mask[235938:2449029]
        return train_mask, val_mask, test_mask


def get_adjacent(edge_of_pg, num_graph_node, symmetric_of_edge=False):
    if not symmetric_of_edge:
        new_edge_of_pg = convert_symmetric(edge_of_pg)
    else:
        new_edge_of_pg = np.copy(edge_of_pg)
    num_edges = len(new_edge_of_pg)
    graph_w = np.ones(num_edges)
    np_edge = np.array(new_edge_of_pg)
    adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                        shape=[num_graph_node, num_graph_node])

    return adj


def convert_symmetric(edge_of_pg):
    new_edge_of_pg = []
    for edge_index in edge_of_pg:
        symmetric_edge_index = [edge_index[1], edge_index[0]]
        if symmetric_edge_index not in edge_of_pg:
            new_edge_of_pg.append(symmetric_edge_index)

    new_edge_of_pg.extend(edge_of_pg)
    return np.array(new_edge_of_pg)


def random_adjacent_sampler(edge_of_pg, num_graph_node, drop_edge=0.1, symmetric_of_edge=False):
    if not symmetric_of_edge:
        new_edge_of_pg = []
        edge_num = int(len(edge_of_pg))
        sampler = np.random.rand(edge_num)
        for i in range(int(edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[i])
        new_edge_of_pg = np.array(new_edge_of_pg)
        new_edge_of_pg = convert_symmetric(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    else:
        new_edge_of_pg = []
        half_edge_num = int(len(edge_of_pg) / 2)
        sampler = np.random.rand(half_edge_num)
        for i in range(int(half_edge_num)):
            if sampler[i] >= drop_edge:
                new_edge_of_pg.append(edge_of_pg[2 * i])
                new_edge_of_pg.append(edge_of_pg[2 * i + 1])
        new_edge_of_pg = np.array(new_edge_of_pg)
        graph_w = np.ones(len(new_edge_of_pg))
        adj = sp.coo_matrix((graph_w, (new_edge_of_pg[:, 0], new_edge_of_pg[:, 1])),
                            shape=[num_graph_node, num_graph_node])
    return adj


def normalization(adj, self_link=True):
    adj = sp.coo_matrix(adj)
    if self_link:
        adj += sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_hat = sp.diags(d_inv_sqrt)
    return d_hat.dot(adj).dot(d_hat).tocoo()
