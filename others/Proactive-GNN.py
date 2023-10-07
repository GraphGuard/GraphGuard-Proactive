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

import dgl.nn as dglnn

from torch.autograd import Variable
from torch.autograd import grad
# parameter definitions

dataset_name = 'cora'

balanced_factor = 0.1 # factor is between 0 and 1, increasing factor means less consider unlearn performance more consider remain accuracy.



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


# graph partition

def greedy_modularity_graph_partition(input_graph):

    # convert graph type to nx
    # G = nx.karate_club_graph()
    # c = greedy_modularity_communities(input_graph, cutoff=partition_num, best_n=partition_num)
    c = nx.algorithms.community.asyn_lpa_communities(input_graph)

    index_lists = c

    return index_lists

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

## training set

### select a specific types of nodes and considered them as a proactive ones.
#### check the numbers of the nodes having specific attributes in the training set



target_training_features = target_graph_features[target_graph_train_mask.nonzero()].squeeze()
target_training_labels = target_graph_labels[target_graph_train_mask.nonzero()].squeeze()

target_training_features_hist = th.sum(th.where(target_training_features>0,1,0),0)
target_training_max_features_index = th.argmax(target_training_features_hist)
target_training_max_features_num = th.max(target_training_features_hist)





target_training_max_label = th.mode(th.transpose(target_training_labels[target_training_features[:,target_training_max_features_index].nonzero()],0,1)).values

# select the proactive samples from training set:
## feature is target_training_max_features_index
## label is target_training_max_label

target_training_proactive_indexes = th.LongTensor(np.intersect1d(th.transpose((target_training_features[:,target_training_max_features_index] > 0).nonzero(), 0,1).squeeze(), th.transpose((target_training_labels == target_training_max_label).nonzero(), 0,1).squeeze()))

## testing set

### also select a specific types of nodes and they are also proactive ones.

target_testing_features = target_graph_features[target_graph_test_mask.nonzero()].squeeze()
target_testing_labels = target_graph_labels[target_graph_test_mask.nonzero()].squeeze()

target_testing_proactive_indexes = th.LongTensor(np.intersect1d(th.transpose((target_testing_features[:,target_training_max_features_index] > 0).nonzero(), 0,1).squeeze(), th.transpose((target_testing_labels == target_training_max_label).nonzero(), 0,1).squeeze()))

## overall proactive sets

target_proactive_indexes = th.LongTensor(np.union1d(target_training_proactive_indexes, target_testing_proactive_indexes))
# train a GNN model

class GCN(nn.Module):
    def __init__(self, feature_number, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GraphConv(
            in_feats=feature_number, out_feats=hid_feats)
        self.conv2 = dglnn.GraphConv(
            in_feats=hid_feats, out_feats=out_feats)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        return h


# train attack model
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

model_target_unradio = GCN(in_feats=target_graph_n_features, hid_feats=16, out_feats=target_graph_n_labels)
opt = th.optim.Adam(model_target_unradio.parameters())

print("start training target model without radioactive data")
for epoch in range(200):
    model_target_unradio.train()
    # forward propagation by using all nodes
    # print(target_graph_features.size())
    logits = model_target_unradio(dgl_target_graph, target_graph_features)
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

acc = evaluate(model_target_unradio, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
print("target model without radioactive data accuracy is")
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
model_target_unradio.eval()
# forward propagation by using all nodes
logits_target = model_target_unradio(dgl_target_graph, target_graph_features)
logits_shadow = model_target_unradio(dgl_shadow_graph, shadow_graph_features)
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
# check the MIA performance

# add perturbation
## select a random vector (does not work)
### select a random labels
## attack the proactive nodes to make them have these bias to these labels without changing their labels

def evaluate_all_nodes(model, g, features, labels):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits
        labels = labels
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def adding_Perturbations(dataset_name, target_model_without_radioactive, target_proactive_indexes, dgl_target_graph, target_graph_features, target_graph_labels):
    """
        This code is for generating the verification nodes for a target model
        Input:
          target dataset_name
          target model training without radioactive data
          target proactive sample indexes (those selected to be protected)
          target graph structure
          target graph features
          target graph labels

        output:
          modified features
          modified adj matrix
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # # device = 'cpu'
    # # dataset_name = 'cora'
    # SAVE_PATH = target_model_path
    # # SAVE_PATH = './models/' + dataset_name + '_target_model_April_7_correct.pth'

    # if dataset_name == 'cora':
    #     node_number = 2708
    #     feature_number = 1433
    #     label_number = 7
    #     data = dgl.data.CoraGraphDataset()
    # if dataset_name == 'citeseer':
    #     node_number = 3327
    #     feature_number = 3703
    #     label_number = 6
    #     data = dgl.data.CiteseerGraphDataset()
    # if dataset_name == 'pubmed':
    #     node_number = 19717
    #     feature_number = 500
    #     label_number = 3
    #     data = dgl.data.PubmedGraphDataset()

    # gcn_msg = fn.copy_src(src='h', out='m')
    # gcn_reduce = fn.sum(msg='m', out='h')

    # g, features, labels, train_mask, test_mask = load_data(dataset_name)
    #
    # # graph preprocess and calculate normalization factor
    # g = data.graph
    # # add self loop
    # if 1:
    #     g.remove_edges_from(nx.selfloop_edges(g))
    #     g.add_edges_from(zip(g.nodes(), g.nodes()))
    # g = DGLGraph(g)
    # # normalization
    # degs = g.in_degrees().float()
    # norm = th.pow(degs, -0.5)
    # norm[th.isinf(norm)] = 0
    # # if device != 'cpu':
    # #     norm = norm.cuda()
    # g.ndata['norm'] = norm.unsqueeze(1)
    g_scipy = dgl_target_graph.adj()
    dgl_target_graph = Variable(g_scipy.to_dense(), requires_grad=True)
    target_graph_features = Variable(target_graph_features, requires_grad=True)

    #use the target model
    # gcn_Net = GCN(feature_number, label_number)
    # gcn_Net.load_state_dict(th.load(SAVE_PATH))

    # optimizer = th.optim.Adam(gcn_Net.parameters(), lr=1e-2, weight_decay=5e-4)
    # dur = []

    target_model_without_radioactive.eval()

    acc_org = evaluate_all_nodes(target_model_without_radioactive, dgl_target_graph, target_graph_features, target_graph_labels)
    print("Original accuracy is " + str(acc_org))


    g_new = copy.deepcopy(dgl_target_graph)
    features_new = copy.deepcopy(target_graph_features)

    sensitive_nodes_index = target_proactive_indexes.detach().clone()

    # generate the detector mask
    # train_mask_org = train_mask.detach().clone()
    # train_mask_sensitive = test_mask.detach().clone() * 0

    # verify_node_index_list = [55,94,69,32,48]
    # for index in sensitive_nodes_index:
    #     train_mask_sensitive[index] = 1

    # detector_mask = th.ByteTensor(train_mask_sensitive)

    # acc_sensitive_org = evaluate_detector(gcn_Net, g, features, labels, detector_mask)

    target_model_without_radioactive.train()
    logits = target_model_without_radioactive(dgl_target_graph, target_graph_features)
    logp = F.log_softmax(logits, 1)


    ## select a labels
    gradience_matrix = F.nll_loss(logp, target_graph_labels, reduce=False)

    optimizer = th.optim.Adam(target_model_without_radioactive.parameters(), lr=1e-2, weight_decay=5e-4)

    perturb_num = 0
    for i in sensitive_nodes_index:
        perturb_num += 1
        print("perturbing node " + str(i) + "... It is the " + str(perturb_num) + "th node. ")
        optimizer.zero_grad()
        gradience_matrix[i].backward(retain_graph=True)

        gradience_loss_g = grad(gradience_matrix[i], g, create_graph=True)
        added_edges_indexs = th.argsort(gradience_loss_g[0][i], descending = True)

        gradience_loss_features = grad(gradience_matrix[i], features, create_graph = True)
        added_features_indexs = th.argsort(gradience_loss_features[0][i], descending = True)

        last_k = -1
        end_loop_flag = 0
        modified_operation = 0
        for j in range(len(added_edges_indexs)):
            if end_loop_flag == 1:
                break
            for k in range(len(added_features_indexs)):
                if gradience_loss_g[0][i][added_edges_indexs[j]]<=0 and gradience_loss_features[0][i][added_features_indexs[k]]<=0:
                    end_loop_flag = 1
                    break
                if modified_operation > 100:
                    end_loop_flag = 1
                    break
                if k <= last_k:
                    continue
                if gradience_loss_g[0][i][added_edges_indexs[j]]>gradience_loss_features[0][i][added_features_indexs[k]]:
                    # loss for edge is larger than features, so perturb edges
                    g_mody_backup = copy.deepcopy(g_new)
                    if device == 'cpu':
                        g_numpy = g_new.detach().numpy()
                    else:
                        g_numpy = g_new.detach().cpu().numpy()
                    g_numpy[i, j]=1
                    np.where(g_numpy > 0, g_numpy, 1)

                    g_new = nx.from_numpy_matrix(g_numpy)
                    g_new = DGLGraph(g_new)
                    # normalization
                    degs = g_new.in_degrees().float()
                    norm = th.pow(degs, -0.5)
                    norm[th.isinf(norm)] = 0
                    # if device != 'cpu':
                    #     norm = norm.cuda()
                    g_new.ndata['norm'] = norm.unsqueeze(1)
                    g_new = g_new.adjacency_matrix()
                    g_new = Variable(g_new.to_dense(), requires_grad=True)

                    acc_new = evaluate_all_nodes(target_model_without_radioactive, g_new, features_new, target_graph_labels)
                    # acc_sensitive_new = evaluate_detector(gcn_Net, g_new, features_new, labels, detector_mask)

                    if (acc_new != acc_org):
                        g_new = g_mody_backup
                        # print('deduct accuracy from: ' + str(acc_org) + ' to ' + str(acc_new) + ' !')
                    else:
                        modified_operation = modified_operation + 1
                        # print('add edges between ' + str(i) + ' and ' + str(j) + ' !')

                    # indexes have sorted, if edge loss is larger than this feature loss,
                    #   it should also larger than others,
                    #   So go to the next edge perturb, break the inner loop
                    break
                else:
                    last_k = k
                    features_mody_backup = copy.deepcopy(features_new)
                    if device == 'cpu':
                        features_numpy = features_new.detach().numpy()
                    else:
                        features_numpy = features_new.detach().cpu().numpy()
                    features_numpy[i][k] = 1

                    np.where(features_numpy > 0, features_numpy, 1)
                    #normalization
                    sum_of_rows = features_numpy.sum(axis=1)
                    features_numpy = features_numpy / sum_of_rows[:, np.newaxis]

                    features_new = Variable(th.FloatTensor(features_numpy), requires_grad=True)

                    acc_new = evaluate_all_nodes(target_model_without_radioactive, g_new, features_new, target_graph_labels)
                    # acc_sensitive_new = evaluate_detector(gcn_Net, g_new, features_new, labels, detector_mask)

                    if (acc_new != acc_org):
                        features_new = features_mody_backup
                        # print('deduct accuracy from: ' + str(acc_org) + ' to ' + str(acc_new) + ' !')
                    else:
                        modified_operation = modified_operation + 1
                        # print('add features at ' + str(k) + ' !')

    print("perturbation done!")

    if device == 'cpu':
        features_numpy = features_new.detach().numpy()
        g_numpy = g_new.detach().numpy()
    else:
        features_numpy = features_new.detach().cpu().numpy()
        g_numpy = g_new.detach().cpu().numpy()

    return features_numpy, g_numpy, sensitive_nodes_index

# check the proactive MIA performance

# train a GNN model

# train attack model
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

model_target_unradio = GCN(in_feats=target_graph_n_features, hid_feats=16, out_feats=target_graph_n_labels)
opt = th.optim.Adam(model_target_unradio.parameters())

print("start training target model without radioactive data")
for epoch in range(200):
    model_target_unradio.train()
    # forward propagation by using all nodes
    # print(target_graph_features.size())
    logits = model_target_unradio(dgl_target_graph, target_graph_features)
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

acc = evaluate(model_target_unradio, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
print("target model without radioactive data accuracy is")
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
model_target_unradio.eval()
# forward propagation by using all nodes
logits_target = model_target_unradio(dgl_target_graph, target_graph_features)
logits_shadow = model_target_unradio(dgl_shadow_graph, shadow_graph_features)
# print(logits_target.size())
logits_total = th.vstack((logits_target,logits_shadow))
# print(logits_total.size())

# create logits label
member_labels = th.ones([len(target_set_index),1]).long()
nonmember_labels = th.zeros([len(shadow_set_index),1]).long()
# print(member_labels)
logits_labels = th.vstack((member_labels,nonmember_labels)).squeeze(1)
# print(logits_labels.size())

print("start training new attack model")

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
print("new attack model accuracy is")
print(acc)