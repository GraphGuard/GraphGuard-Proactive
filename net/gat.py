import torch
import torch.nn as nn
import torch.nn.functional as F

from net.gcn import myGraphConv

class GAT(nn.Module):
    def __init__(self, feature_number, hid_feats, out_feats, num_heads, num_layers):
        super(GAT, self).__init__()
        self.conv1 = myGraphConv(
            in_features=feature_number, out_features=hid_feats)
        self.conv2 = myGraphConv(
            in_features=hid_feats * num_heads, out_features=out_feats)
        self.attentions = nn.ModuleList([
            myGraphConv(in_features=hid_feats, out_features=hid_feats * num_heads)
            for _ in range(num_layers)
        ])
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, adj, inputs):
        h = self.conv1(adj, inputs)
        h = F.relu(h)

        all_heads = []
        for i in range(self.num_heads):
            head = h
            for j in range(self.num_layers):
                attention = self.attentions[j]
                head = attention(adj, head)
                head = F.relu(head)
                all_heads.append(head)

        h = torch.cat(all_heads, dim=1)
        h = self.conv2(adj, h)
        h = F.relu(h)
        return h