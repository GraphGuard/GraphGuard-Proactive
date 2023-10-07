import torch
import dgl
import logging
import torch.nn.functional as F
from net.gat import GAT
from net.gcn import GCN
from net.gin import GIN
from net.sage import SAGE


def Evaluation_gnn(model, g, features, labels, print_results=True):
    """
    This code is for evaluate the trained GNN model accuracy.

    Inputs:
    target_model: The model for evaluation
    Testing graph:
    g:
    features:
    labels:
    train_mask:
    test_mask:

    Output:
    accuracy: accuracy of the model
    """
    model = model.to('cpu')
    model.eval()
    with torch.no_grad():
        if isinstance(g, dgl.DGLGraph):
            logits = model(g.adjacency_matrix(), features)
        elif isinstance(features, torch.Tensor):
            logits = model(g, features)
        # get probabilities from logits
        probs = F.softmax(logits, dim=1)
        # get the true label (probabilities) list
        prob_list = probs[:, labels].diag()
        _, indices = torch.max(logits, dim=1)
        # get the true label (probabilities) list
        correct = torch.sum(indices == labels)
        # if print_results:
        #     # show the accuracy
        #     print(f"The evaluation accuracy of the model is:{correct.item() * 1.0 / len(labels)}")

    return correct.item() * 1.0 / len(labels), logits, prob_list


def Train_gnn_model(params, g, features, labels, train_mask, test_mask):
    """
    This function is for train a GNN model.

    Inputs:
    model_type: the GNN model architecture
    dataset with following values:
    g: graph in DGL graph format
    features: features in Tensor format
    labels: labels in Tensor format
    train_mask: training mask as Boolean Tensor
    test_mask: testing mask as Boolean Tensor

    Outputs:
    trained_model: well-trained GNN model
    """
    model_type = params['model']
    hidden_feats = int(params['net_params']['hidden'])
    epochs = int(params['epochs'])
    step = int(params['step'])
    num_labels = int(params['net_params']['num_labels'])
    device = params['device']
    if model_type == 'GCN':
        model = GCN(feature_number=features.size()[
                    1], hid_feats=hidden_feats, out_feats=num_labels)
    elif model_type == 'GraphSage':
        model = SAGE(feature_number=features.size()[
                     1], hid_feats=hidden_feats, out_feats=num_labels)
    elif model_type == 'GIN':
        model = GIN(feature_number=features.size()[
                     1], hid_feats=hidden_feats, out_feats=num_labels)
    elif model_type == 'GAT':
        model = GAT(feature_number=features.size()[
                     1], hid_feats=hidden_feats, out_feats=num_labels)
    opt = torch.optim.Adam(model.parameters())

    # set training_mask to be all True!
    train_mask = torch.logical_or(train_mask, torch.logical_not(train_mask))
    # print features and its type, size, and nonzero elements with information
    # print(f'features: {features}')
    # print(f'type: {type(features)}')
    # print(f'size: {features.size()}')
    # print(f'nonzero elements: {torch.nonzero(features)}')
    # start training
    model = model.to(device)
    model.train()
    total = 0
    total_correct = 0
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    if isinstance(g, dgl.DGLGraph):
            g = g.adjacency_matrix()
    else:
        breakpoint()
    for epoch in range(epochs):
        logits = model(g, features)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        total += len(labels)
        total_correct += correct
        if (epoch+1) % step == 0:
            # print(f"Epoch:{epoch+1}, the training accuracy of the model is:{correct.item() * 1.0 / len(labels)}")
            logging.info(
                f"Epoch:{epoch+1}, the training accuracy of the model is:{correct.item() * 1.0 / len(labels)}")
    return model
