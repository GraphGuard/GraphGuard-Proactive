import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import dgl
class myGraphConv(nn.Module):
    """
    from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, in_features, out_features, bias=True):
        super(myGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = torch.mm(input, self.weight)
        if isinstance(adj, dgl.sparse.SparseMatrix):
            adj_dense = adj.to_dense()
            output = torch.mm(adj_dense, support)
        if isinstance(adj, torch.Tensor):
            output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
    

class GCN(nn.Module):
    def __init__(self, feature_number, hid_feats, out_feats):
        super().__init__()
        self.conv1 = myGraphConv(
            in_features=feature_number, out_features=hid_feats)
        self.conv2 = myGraphConv(
            in_features=hid_feats, out_features=out_feats)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        return h



class Attack(nn.Module):
    def __init__(self, label_number):
        super().__init__()
        self.fc_1 = nn.Linear(label_number,128)
        self.fc_2 = nn.Linear(128,2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        h1 = self.fc_1(inputs)
        h = F.relu(h1)
        h = self.dropout(h)
        h = self.fc_2(h)
        h = F.sigmoid(h)
        return h