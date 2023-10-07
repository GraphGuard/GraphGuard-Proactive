import copy
import warnings
import dgl.data
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
import numpy as np

from networkx.algorithms.community import greedy_modularity_communities

# Contruct a two-layer GNN model
import dgl.nn as dglnn

# load dataset
def load_data(dataset_name):
    if dataset_name == 'cora':
        data = dgl.data.CoraGraphDataset()
    if dataset_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    if dataset_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.ByteTensor(data.train_mask)
    test_mask = th.ByteTensor(data.test_mask)
    g = data[0]
    return g, features, labels, train_mask, test_mask

# graph partition
def greedy_modularity_graph_partition(input_graph):

    # convert graph type to nx
    # G = nx.karate_club_graph()
    # c = greedy_modularity_communities(input_graph, cutoff=partition_num, best_n=partition_num)
    c = nx.algorithms.community.asyn_lpa_communities(input_graph)

    index_lists = c

    return index_lists


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

dataset_name = 'citeseer'

(g, features, labels, train_mask, test_mask) = load_data(dataset_name)

if dataset_name == 'cora':
    node_number = 2708
    feature_number = 1433
    label_number = 7
if dataset_name == 'citeseer':
    node_number = 3327
    feature_number = 3703
    label_number = 6
if dataset_name == 'pubmed':
    node_number = 19717
    feature_number = 500
    label_number = 3

# print(type(g))
networkx_g = dgl.to_networkx(g)
index_lists = list(greedy_modularity_graph_partition(networkx_g))
# print(type(index_lists[0]))
list_len = len(index_lists)

target_set_index = set()

shadow_set_index = set()

for i in range(list_len):
    if i%2==0:
        target_set_index = target_set_index.union(index_lists[i])
    else:
        shadow_set_index = shadow_set_index.union(index_lists[i])

# print(len(target_set_index))
# print(len(shadow_set_index))

## target set
networkx_target_graph = networkx_g.subgraph(list(target_set_index)).copy()
dgl_target_graph = dgl.from_networkx(networkx_target_graph)
# print(type(dgl_target_graph))
# print(type())
## shadow set
networkx_shadow_graph = networkx_g.subgraph(list(shadow_set_index)).copy()
dgl_shadow_graph = dgl.from_networkx(networkx_shadow_graph)

dgl_shadow_graph_adj = th.FloatTensor(nx.adjacency_matrix(networkx_shadow_graph).todense())
# train a GNN model in target set

# train_mask, test_mask
target_graph_features = features[list(target_set_index)]
target_graph_labels = labels[list(target_set_index)]
# print(target_graph_features)
target_graph_train_mask = train_mask[list(target_set_index)]
# print(target_graph_features.size())
target_graph_test_mask = test_mask[list(target_set_index)]
target_graph_n_features = features.shape[1]
target_graph_n_labels = int(labels.max().item() + 1)

shadow_graph_features = features[list(shadow_set_index)]
shadow_graph_labels = labels[list(shadow_set_index)]
# print(shadow_graph_features)
shadow_graph_train_mask = train_mask[list(shadow_set_index)]
# print(shadow_graph_features.size())
shadow_graph_test_mask = test_mask[list(shadow_set_index)]

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

model = SAGE(in_feats=target_graph_n_features, hid_feats=100, out_feats=target_graph_n_labels)
opt = th.optim.Adam(model.parameters())

print("start training target model")
for epoch in range(100):
    model.train()
    # forward propagation by using all nodes
    # print(target_graph_features.size())
    logits = model(dgl_target_graph, target_graph_features)
    # compute loss
    # print(logits.size())
    # print(target_graph_train_mask)
    # print(target_graph_labels.size())
    loss = F.cross_entropy(logits[target_graph_train_mask], target_graph_labels[target_graph_train_mask])
    # # compute validation accuracy
    # acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    # print(loss.item())

acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
print("target model accuracy is")
print(acc)
# general MIA
#“firstly test whether perfect MIA can unlearned

## graph partition

### training set

### testing set

## train shadow target model on training set

## train attack model using two sets

###create an attack model
class Attack(nn.Module):
    def __init__(self, label_number, hid_feats, out_feats):
        super().__init__()
        self.fc_1 = nn.Linear(label_number,32)
        self.fc_2 = nn.Linear(32,2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, inputs):
        h1 = self.fc_1(inputs)
        h = F.relu(h1)
        # h = self.dropout(h)
        h = self.fc_2(h)
        # h = F.relu(h)
        h = F.sigmoid(h)
        return h
            # ,h1,h

def MLP_evaluate(model, inputs, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(inputs)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

### set training set:
model.eval()
# forward propagation by using all nodes
logits_target = model(dgl_target_graph, target_graph_features)
logits_shadow = model(dgl_shadow_graph, shadow_graph_features)
# print(logits_target.size())
logits_total = th.vstack((logits_target,logits_shadow))
# print(logits_total.size())

# create logits label
member_labels = th.ones([len(target_set_index),1]).long()
nonmember_labels = th.zeros([len(shadow_set_index),1]).long()
# print(member_labels)
logits_labels = th.vstack((member_labels,nonmember_labels)).squeeze(1)
# print(logits_labels.size())

print("start training attack model")

attack_model = Attack(label_number,100,2)
opt_attack = th.optim.Adam(attack_model.parameters())

# print("start training  model")
for epoch in range(1000):
    attack_model.train()
    # forward propagation by using all nodes
    # print(logits_total.size())
    attack_logits= attack_model(logits_total)
    # print(attack_logits.size())
    # print(h1.size())
    # print(attack_logits.size())
    # compute loss
    # print(logits)
    # print(target_graph_train_mask)
    # print(target_graph_labels[target_graph_train_mask])
    # print(attack_logits.size())
    # print(logits_labels.size())
    attack_loss = F.cross_entropy(attack_logits, logits_labels)
    # # compute validation accuracy
    # acc = MLP_evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, th.ones([1, len(target_set_index)]))
    # backward propagation
    opt_attack.zero_grad()
    attack_loss.backward(retain_graph=True)
    opt_attack.step()
    # print(attack_loss.item())

acc = MLP_evaluate(attack_model, logits_total, logits_labels, (th.ones([node_number])>0))
print("attack model accuracy is")
print(acc)

# unlearning samples generation

## define GAE to learn structure.

### G=(A,X)-GNN_learn-> Z -> X'
class GNN_GAE(nn.Module):
    def __init__(self, feature_number, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(
            in_feats=feature_number, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.GraphConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        return h

results_labels = th.ones([len(shadow_set_index),1]).long()

generation_model = GNN_GAE(feature_number,32,16)
opt_generation = th.optim.Adam(generation_model.parameters())

model.eval()
attack_model.eval()
print("start training generation model")

for epoch in range(200):
    generation_model.train()
    # forward propagation by using all nodes
    # print(target_graph_features.size())
    logits = generation_model(dgl_shadow_graph, shadow_graph_features)
    # compute loss
    # print(logits.size())
    # print(target_graph_train_mask)
    # print(target_graph_labels.size())
    # logits_numpy = logits.copy().cpu().detach().numpy()
    new_adj = th.sigmoid(th.matmul(logits, logits.t()))
    GAE_loss = th.sum(th.abs(dgl_shadow_graph_adj - new_adj))
    # src, dst = th.nonzero(new_adj)
    # g_new = dgl.graph((src, dst))
    # g_new.ndata['feat'] = shadow_graph_features
    # results = attack_model(model(g_new,shadow_graph_features))
    # attack_loss = F.cross_entropy(results, results_labels)
    # loss = F.cross_entropy(logits[target_graph_train_mask], target_graph_labels[target_graph_train_mask])
    # # compute validation accuracy
    # acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
    # backward propagation
    opt.zero_grad()
    GAE_loss.backward()
    opt.step()
    print(GAE_loss)

###
# ### G'=(A,X') -GNN_target-> -GNN_attack-> 1
# class GNN_sythetic(nn.Module):
#     def __init__(self, feature_number, hid_feats, out_feats):
#         super().__init__()
#         self.conv1 = dglnn.GraphConv(
#             in_feats=feature_number, out_feats=hid_feats, aggregator_type='mean')
#         self.conv2 = dglnn.GraphConv(
#             in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
#
#     def forward(self, graph, inputs):
#         h = self.conv1(graph, inputs)
#         h = F.relu(h)
#         h = self.conv2(graph, h)
#         h = F.relu(h)
#         return h
#
# results_labels = th.ones([len(shadow_set_index),1]).long()
#
# generation_model = GNN_sythetic(feature_number,32,16)
# opt_generation = th.optim.Adam(generation_model.parameters())
#
# model.eval()
# attack_model.eval()
# print("start training generation model")
# for epoch in range(200):
#     generation_model.train()
#     # forward propagation by using all nodes
#     # print(target_graph_features.size())
#     logits = generation_model(dgl_shadow_graph, shadow_graph_features)
#     # compute loss
#     # print(logits.size())
#     # print(target_graph_train_mask)
#     # print(target_graph_labels.size())
#     # logits_numpy = logits.copy().cpu().detach().numpy()
#     new_adj = th.sigmoid(th.matmul(logits, logits.t()))
#     src, dst = th.nonzero(new_adj)
#     g_new = dgl.graph((src, dst))
#     g_new.ndata['feat'] = shadow_graph_features
#     results = attack_model(model(g_new,shadow_graph_features))
#     attack_loss = F.cross_entropy(results, results_labels)
#     # loss = F.cross_entropy(logits[target_graph_train_mask], target_graph_labels[target_graph_train_mask])
#     # # compute validation accuracy
#     # acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
#     # backward propagation
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
# ## graph partition based on attributes
# ### select nodes whoes label is 0 and the attribute in index ? is 1.
# label_0_in_target_graph = th.Tensor(list(target_set_index))[(target_graph_labels==0)].long()
# label_others_in_target_graph = th.Tensor(list(target_set_index))[(target_graph_labels!=0)].long()
# label_0_index_target_graph_mask = (target_graph_labels==0)
# label_others_index_target_graph_mask = (target_graph_labels!=0)
# # print(label_0_in_target_graph.size())
# # print(labels[label_0_in_target_graph.long().tolist()])
# # print()
#
# # target_node_index = label_0_in_target_graph
# # nontarget_node_index = label_others_in_target_graph
#
# ## add perturbations based on gradients
#
# # unlearn the target GNN model
#
# # define unlearning label
# nonmember_labels = th.zeros([len(target_set_index)]).long()
# member_labels = th.ones([len(target_set_index)]).long()
#
# print("start unlearning target model")
#
# attack_model.eval()
# for epoch in range(10):
#     model.train()
#     # forward propagation by using all nodes
#     # print(target_graph_features.size())
#     logits = model(dgl_target_graph, target_graph_features)
#     attack_logits = attack_model(logits)
#     # compute loss
#     # print(logits.size())
#     # print(target_graph_train_mask)
#     # print(target_graph_labels.size())
#     # accuracy loss
#     acc_loss_forget = F.cross_entropy(logits[label_0_index_target_graph_mask], target_graph_labels[label_0_index_target_graph_mask])
#     acc_loss_remain = F.cross_entropy(logits[label_others_index_target_graph_mask], target_graph_labels[label_others_index_target_graph_mask])
#     # unlearning loss
#     unlearn_loss_forget = F.cross_entropy(attack_logits[label_0_index_target_graph_mask], nonmember_labels[label_0_index_target_graph_mask])
#     # unlearn_loss_remain = F.cross_entropy(attack_logits[label_others_index_target_graph_mask], member_labels[label_others_index_target_graph_mask])
#     loss = 0.4*acc_loss_remain - 0.4*acc_loss_forget + 0.6 * unlearn_loss_forget
#            # + unlearn_loss_remain
#     # loss = F.cross_entropy(logits[target_graph_train_mask], target_graph_labels[target_graph_train_mask])
#     # # compute validation accuracy
#     # acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
#     # backward propagation
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#
# acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
# print("unlearned target model accuracy is")
# print(acc)
#
# # evaluating MIA
#
# acc = MLP_evaluate(attack_model, logits_target, member_labels, label_0_index_target_graph_mask)
# print("original attack model accuracy for unlearned samples is")
# print(acc)
#
# acc = MLP_evaluate(attack_model, logits_target, member_labels, label_others_index_target_graph_mask)
# print("original attack model accuracy for remained samples is")
# print(acc)
#
# model.eval()
# attack_model.eval()
# current_logits_target = model(dgl_target_graph, target_graph_features)
#
# acc = MLP_evaluate(attack_model, current_logits_target, member_labels, label_0_index_target_graph_mask)
# print("current attack model accuracy for unlearned samples is")
# print(acc)
#
# acc = MLP_evaluate(attack_model, current_logits_target, member_labels, label_others_index_target_graph_mask)
# print("current attack model accuracy for remained samples is")
# print(acc)
#
# # evaluate the unlearned GNN
#
# ## train new GNN without unlearned samples
#
# ## dealing with MIA
#
# ## model parameter distribution
#
# ## output distribution
#
#
#
# # train attack model
#
# # check the MIA performance
#
# # add perturbation

# check the proactive MIA performance