from CoraData import *


class CiteseerData:
    def __init__(self, path_of_citeseer):
        self.path_of_citeseer = path_of_citeseer

        # Re-index papers starting from 0
        self.index_of_pg = dict()
        # Represent paper labels in numerical form
        self.index_of_pg_label = dict()

        self.feature_of_pg = []
        self.label_of_pg = []
        self.edge_of_pg = []

        self.__dataset_loader()
        self.num_nodes = len(self.feature_of_pg)
        self.num_edges = len(self.edge_of_pg)
        self.num_of_class = len(self.index_of_pg_label)
        self.feature_dim = len(self.feature_of_pg[0])

    def __dataset_loader(self):
        path_cites = self.path_of_citeseer + "/citeseer.cites"
        path_contents = self.path_of_citeseer + "/citeseer.content"

        with open(path_contents, 'r', encoding='utf-8') as file_content:
            for node in file_content.readlines():
                node_cont = node.split()
                # Rearrange articles according to the order they enter the dict
                self.index_of_pg[node_cont[0]] = len(self.index_of_pg)
                self.feature_of_pg.append([int(i) for i in node_cont[1:-1]])

                label = node_cont[-1]
                if label not in self.index_of_pg_label.keys():
                    # Rearrange labels according to the order they enter the dict
                    self.index_of_pg_label[label] = len(self.index_of_pg_label)
                self.label_of_pg.append(self.index_of_pg_label[label])

        with open(path_cites, 'r', encoding='utf-8') as file_cite:
            for edge in file_cite.readlines():
                cited, citing = edge.split()

                # Note that Citeseer is a faulty dataset, here we handle it specially 
                # by deleting edges corresponding to nodes not present in .content
                if (cited not in self.index_of_pg.keys()) or (citing not in self.index_of_pg.keys()):
                    continue

                # Consider the directed graph as undirected
                edge_1 = [self.index_of_pg[citing], self.index_of_pg[cited]]
                edge_2 = [self.index_of_pg[cited], self.index_of_pg[citing]]
                if edge_1 not in self.edge_of_pg:
                    self.edge_of_pg.append(edge_1)
                    self.edge_of_pg.append(edge_2)

    # Get a sparse adjacency matrix
    def get_adjacent(self):
        graph_w = np.ones(self.num_edges)
        np_edge = np.array(self.edge_of_pg)
        adj = sp.coo_matrix((graph_w, (np_edge[:, 0], np_edge[:, 1])),
                            shape=[self.num_nodes, self.num_nodes])
        return adj

    # Get an adjacency matrix with randomly hidden edges
    def random_adjacent_sampler(self, drop_edge=0.1):
        new_edge_of_pg = []
        half_edge_num = int(len(self.edge_of_pg) / 2)
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

    # Perform D^{-1/2}AD^{-1/2} operation on adjacency matrix A
    @staticmethod
    def normalization(adj, self_link=True):
        adj = sp.coo_matrix(adj)
        if self_link:
            adj += sp.eye(adj.shape[0])  # Add self-connections
        row_sum = np.array(adj.sum(1))   # Sum over columns to get the degree of each row
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_hat = sp.diags(d_inv_sqrt)
        return d_hat.dot(adj).dot(d_hat).tocoo()  # Return in coo_matrix format

    def output_data(self):
        print(f"the shape of feature is {len(self.feature_of_pg), len(self.feature_of_pg[0])}")
        print(f"the shape of label is {len(self.label_of_pg)}")
        print(f"the shape of edge is {len(self.edge_of_pg), len(self.edge_of_pg[0])}")

    # Partition data into training, validation, and test sets
    @staticmethod
    def data_partition_node(data_size=3312):
        mask = torch.randperm(data_size)
        train_mask = mask[:180]
        val_mask = mask[180:800]
        test_mask = mask[2312:3312]
        return train_mask, val_mask, test_mask
