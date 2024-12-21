import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch.nn import Sequential, Linear, ReLU

# 本文件中记录了我用PYG搭建的GCN、GAT、SAGEConv网络以验证我的卷积网络正确性


# 使用了pyg中GCNConv的类
class ClassificationGCNFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(ClassificationGCNFromPYG, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GCNConv(input_feature_dim, hidden_layer_dim))
        # 隐藏层
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GCNConv(hidden_layer_dim, hidden_layer_dim))
        # 输出层
        self.layers.append(GCNConv(hidden_layer_dim, num_of_class))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, adj):
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(output, adj)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        output = torch.sigmoid(output)
        return output


# 使用 GAT 替代 GCNConv
class ClassificationGATFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, heads=8, dropout=0.01, use_pair_norm=True):
        super(ClassificationGATFromPYG, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GATConv(input_feature_dim, hidden_layer_dim, heads=heads, concat=True))
        # 隐藏层
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GATConv(hidden_layer_dim * heads, hidden_layer_dim, heads=heads, concat=True))
        # 输出层
        self.layers.append(GATConv(hidden_layer_dim * heads, num_of_class, heads=1, concat=False))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, adj):
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(output, adj)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        output = torch.sigmoid(output)
        return output
    

# 使用 GraphSAGE 替代 GCNConv
class ClassificationSAGEFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(ClassificationSAGEFromPYG, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(SAGEConv(input_feature_dim, hidden_layer_dim))
        # 隐藏层
        for i in range(num_of_hidden_layer-2):
            self.layers.append(SAGEConv(hidden_layer_dim, hidden_layer_dim))
        # 输出层
        self.layers.append(SAGEConv(hidden_layer_dim, num_of_class))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, adj):
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(output, adj)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        output = torch.sigmoid(output)
        return output


class ClassificationGINFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(ClassificationGINFromPYG, self).__init__()
        self.layers = nn.ModuleList()

        # GIN 需要一个 MLP 作为每层的聚合器
        def create_mlp(input_dim, output_dim):
            return Sequential(
                Linear(input_dim, output_dim),
                ReLU(),
                Linear(output_dim, output_dim)
            )

        # 输入层
        self.layers.append(GINConv(create_mlp(input_feature_dim, hidden_layer_dim)))
        # 隐藏层
        for i in range(num_of_hidden_layer - 2):
            self.layers.append(GINConv(create_mlp(hidden_layer_dim, hidden_layer_dim)))
        # 输出层
        self.layers.append(GINConv(create_mlp(hidden_layer_dim, num_of_class)))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer

    @staticmethod
    def PairNorm(x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature

    def forward(self, x_feature, edge_index):
        """
        :param x_feature: 节点特征矩阵 (N, D)
        :param edge_index: 图的边列表 (2, E)
        """
        output = x_feature
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)

            output = layer(output, edge_index)
            if i != (self.num_of_hidden_layer - 1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        output = torch.sigmoid(output)
        return output
