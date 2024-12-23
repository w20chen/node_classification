import random

import numpy as np
import torch
import scipy.sparse as sp


class CoraData:
    def __init__(self, path_of_cora):
        self.cora_path = path_of_cora

        # Re-index papers starting from 0
        self.index_of_pg = dict()
        # Represent paper labels in numerical form
        self.index_of_pg_label = dict()

        self.feature_of_pg = []
        self.label_of_pg = []
        self.edge_of_pg = []

        self.__dataset_loader()
        self.num_nodes = len(self.feature_of_pg)
        # num_edges is 2 times # of directed edges
        self.num_edges = len(self.edge_of_pg)
        self.num_of_class = len(self.index_of_pg_label)
        self.feature_dim = len(self.feature_of_pg[0])

    def __dataset_loader(self):
        path_cites = self.cora_path + "/cora.cites"
        path_contents = self.cora_path + "/cora.content"

        with open(path_contents, 'r', encoding='utf-8') as file_content:
            for node in file_content.readlines():
                node_cont = node.split()
                # Rearrange papers according to the order they enter the index_of_pg dict
                self.index_of_pg[node_cont[0]] = len(self.index_of_pg)
                self.feature_of_pg.append([int(i) for i in node_cont[1:-1]])

                label = node_cont[-1]
                if label not in self.index_of_pg_label.keys():
                    # Rearrange labels according to the order they enter the index_of_pg_label dict
                    self.index_of_pg_label[label] = len(self.index_of_pg_label)
                self.label_of_pg.append(self.index_of_pg_label[label])

        with open(path_cites, 'r', encoding='utf-8') as file_cite:
            for edge in file_cite.readlines():
                cited, citing = edge.split()
                # Cora is a directed graph, here we set it as undirected
                edge_1 = [self.index_of_pg[citing], self.index_of_pg[cited]]
                edge_2 = [self.index_of_pg[cited], self.index_of_pg[citing]]
                if edge_1 not in self.edge_of_pg:
                    self.edge_of_pg.append(edge_1)
                    self.edge_of_pg.append(edge_2)

    # Get a sparse adjacency matrix
    def get_adjacent(self):
        graph_w = np.ones(self.num_edges)
        np_edge = np.array(self.edge_of_pg)
        # Create a sparse adjacency matrix in COO format
        adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # Get a randomly sampled adjacency matrix with hidden edges
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

    # Perform D^{-1/2}AD^{-1/2} operation on adjacency matrix A, self-loops are optional
    @staticmethod
    def normalization(adj, self_link=True):
        adj = sp.coo_matrix(adj)
        if self_link:
            adj += sp.eye(adj.shape[0])  # Add self-loops
        row_sum = np.array(adj.sum(1))   # Sum over columns to get the degree of each row
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_hat = sp.diags(d_inv_sqrt)
        return d_hat.dot(adj).dot(d_hat).tocoo()   # Return in COO matrix format

    def output_data(self):
        print(f"the shape of feature is {len(self.feature_of_pg),len(self.feature_of_pg[0])}")
        print(f"the shape of label is {len(self.label_of_pg)}")
        print(f"the shape of edge is {len(self.edge_of_pg),len(self.edge_of_pg[0])}")

    # Partition data into training, validation, and test sets
    @staticmethod
    def data_partition_node(data_size=2708):
        mask = torch.randperm(data_size)    # Permutation of 0~2707 in random order
        train_mask = mask[:140]
        val_mask = mask[140:640]
        test_mask = mask[1708:2708]
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
        for i in range(edge_num):
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
        for i in range(half_edge_num):
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
