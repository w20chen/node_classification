import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphConvolution(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, use_bias=True):
        # This part calculates the convolution D^-1/2 A D^-1/2 * X * W, where X is the feature and W is the parameter
        super(GraphConvolution, self).__init__()

        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.use_bias = use_bias

        # Define the shape of the W weight for the GCN layer
        self.weight = nn.Parameter(torch.Tensor(in_features_dim, out_features_dim))

        # Define the b weight matrix for the GCN layer
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # Initialize the W and b parameters in the nn.Module class
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        # init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adj, in_feature):
        # The input is a sparse matrix adj
        support = torch.mm(in_feature, self.weight)  # X*W
        output = torch.sparse.mm(adj, support)       # A*X*W
        if self.use_bias:
            output += self.bias                      # Add bias term
        return output

class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_num, dropout=0.01):
        super(MyMLP, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        # Hidden layers
        for i in range(hidden_layer_num - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.dropout = nn.Dropout(p=dropout)
        self.num_of_hidden_layer = hidden_layer_num

    def forward(self, x):
        output = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                output = self.dropout(output)
            output = layer(output)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
        return output


# Build a GCN network for node classification using my custom convolutional network
class MyClassificationGCN(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(MyClassificationGCN, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphConvolution(input_feature_dim, hidden_layer_dim))
        # Hidden layers
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GraphConvolution(hidden_layer_dim, hidden_layer_dim))
        # Output layer
        self.layers.append(GraphConvolution(hidden_layer_dim, num_of_class))
        self.dropout = nn.Dropout(p=dropout)
        self.use_pair_norm = use_pair_norm
        self.num_of_hidden_layer = num_of_hidden_layer
        self.mlp = MyMLP(input_size=hidden_layer_dim, hidden_size=hidden_layer_dim,
                         output_size=num_of_class, hidden_layer_num=5)

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

            output = layer(adj, output)
            if i != (self.num_of_hidden_layer-1):
                output = F.relu(output)
            if self.use_pair_norm:
                output = self.PairNorm(output)

        # output = self.mlp(output)
        output = torch.sigmoid(output)
        return output


# Use the GCNConv class from PYG
class ClassificationGCNFromPYG(nn.Module):
    def __init__(self, hidden_layer_dim, num_of_hidden_layer,
                 num_of_class=7, input_feature_dim=1433, dropout=0.01, use_pair_norm=True):
        super(ClassificationGCNFromPYG, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GCNConv(input_feature_dim, hidden_layer_dim))
        # Hidden layers
        for i in range(num_of_hidden_layer-2):
            self.layers.append(GCNConv(hidden_layer_dim, hidden_layer_dim))
        # Output layer
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
