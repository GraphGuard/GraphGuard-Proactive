import torch
import torch.nn as nn
import torch.nn.functional as F

from net.gcn import myGraphConv

class GIN(nn.Module):
    def __init__(self, feature_number, hid_feats, out_feats, num_layers=2):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList([myGraphConv(feature_number, hid_feats)] +
                                   [myGraphConv(hid_feats, hid_feats) for _ in range(num_layers-1)])
        
        self.mlp = nn.Linear(hid_feats, out_feats)

    def forward(self, adj, inputs):
        h = inputs
        for conv in self.convs:
            h = F.relu(conv(adj, h))
        h = self.mlp(h)
        return h