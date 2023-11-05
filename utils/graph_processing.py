import random
import dgl
import networkx
import numpy as np
import torch
import scipy.sparse as sp
import torch
import logging

def load_data(dataset_name):
    """
    This function is for load the dataset provided by DGL.

    Input:
    dataset_name: the dataset need to be loaded

    Output:
    g: graph in DGL graph format
    features: features in Tensor format
    labels: labels in Tensor format
    train_mask: training mask as Boolean Tensor
    test_mask: testing mask as Boolean Tensor
    num_class
    """
    if dataset_name == 'cora':
        data = dgl.data.CoraGraphDataset()
    if dataset_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    if dataset_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    if dataset_name == 'lastfm':
        data = dgl.data.LastFMGraphDataset()
    if dataset_name == 'flickr':
        data = dgl.data.FlickrDataset()
    if dataset_name == 'reddit':
        data = dgl.data.RedditDataset(self_loop=True)
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    num_class = data.num_classes
    return g, features, labels, train_mask, test_mask, num_class

def subgraph_generation(g, features, labels, train_mask, test_mask, params):
    logging.info("Start to convert DGL graph to networkx graph")
    networkx_g = dgl.to_networkx(g)
    logging.info(f"Number of nodes in the original graph: {networkx_g.number_of_nodes()}")
    index_lists = list(networkx.algorithms.community.asyn_lpa_communities(networkx_g))
    list_len = len(index_lists)
    dataset_name = params['dataset']
    num_subsets = int(params['num_subsets'])
    logging.info(f"Number of subsets: {num_subsets}")
    # split the nodes into num_subsets subsets
    sets = [set() for _ in range(num_subsets)]
    node_count = 0  
    for i in range(list_len):
        if dataset_name == 'flickr' and len(index_lists[i]) > 70000:
            continue
        elif node_count > 5500:
            break
        else:
            sets[i % num_subsets] = sets[i % num_subsets].union(index_lists[i])
            node_count += len(index_lists[i])
    # combine the sets into one set
    selected_set = set(item for sublist in sets for item in sublist)
    logging.info(f"Number of nodes in the subgraph: {len(list(selected_set))}")
    networkx_graph = networkx_g.subgraph(list(selected_set)).copy()
    g = dgl.from_networkx(networkx_graph)
    features = features[list(selected_set)]
    labels = labels[list(selected_set)]
    train_mask = train_mask[list(selected_set)]
    test_mask = test_mask[list(selected_set)]
    logging.info("Subgraph generation completed")
    return g, features, labels, train_mask, test_mask


def Graph_partition(g, features, labels, train_mask, test_mask):
    """
    This function is for use greedy modularity graph partition technique to divide a large graph to TWO subgraph.
    
    Input:
    Input graph with detailed information:
    g: graph in DGL graph format
    features: features in Tensor format
    labels: labels in Tensor format
    train_mask: training mask as Boolean Tensor
    test_mask: testing mask as Boolean Tensor
    
    Output:
    For both subgraph, the function will return below values:
    g: graph in DGL graph format
    features: features in Tensor format
    labels: labels in Tensor format
    train_mask: training mask as Boolean Tensor
    test_mask: testing mask as Boolean Tensor
    """
    networkx_g = dgl.to_networkx(g)

    index_lists = list(networkx.algorithms.community.asyn_lpa_communities(networkx_g))
    list_len = len(index_lists)

    target_set_index = set()

    shadow_set_index = set()

    for i in range(list_len):
        if i % 2 == 0:
            target_set_index = target_set_index.union(index_lists[i])
        else:
            shadow_set_index = shadow_set_index.union(index_lists[i])

    networkx_target_graph = networkx_g.subgraph(list(target_set_index)).copy()
    target_g = dgl.from_networkx(networkx_target_graph)

    networkx_shadow_graph = networkx_g.subgraph(list(shadow_set_index)).copy()
    shadow_g = dgl.from_networkx(networkx_shadow_graph)

    target_features = features[list(target_set_index)]
    target_labels = labels[list(target_set_index)]
    target_train_mask = train_mask[list(target_set_index)]
    target_test_mask = test_mask[list(target_set_index)]
    target_graph_n_features = features.shape[1]
    target_graph_n_labels = int(labels.max().item() + 1)

    shadow_features = features[list(shadow_set_index)]
    shadow_labels = labels[list(shadow_set_index)]
    shadow_train_mask = train_mask[list(shadow_set_index)]
    shadow_test_mask = test_mask[list(shadow_set_index)]

    return target_g, target_features, target_labels, target_train_mask, target_test_mask, \
        shadow_g, shadow_features, shadow_labels, shadow_train_mask, shadow_test_mask


def Identify_proactive_nodes(features, labels, proactive_features_index, proactive_label):
    """
    This function is for identifying the proactive nodes. In our cases, we select the nodes with specific features and label.

    Inputs:
    The graph searching for proactive nodes:
    features:
    labels:

    proactive_features: the proactive nodes should have this feature
    proactive_label: the proactive nodes should have this label

    Output:
    proactive_node_index: a list for the proactive node index in this graph
    """
    proactive_node_index = torch.LongTensor(np.intersect1d(
        torch.transpose((features[:, proactive_features_index] > 0).nonzero(), 0, 1).squeeze(),
        torch.transpose((labels == proactive_label).nonzero(), 0, 1).squeeze()))

    return proactive_node_index

def Select_proactive_node(features, labels):
    """
    This function is for select the maximum numbers of a specific (feature, label) pairs.

    Inputs:
    features: the feature of the graph
    labels: the label of the graph

    Outputs:
    proactive_features_index: selected feature
    proactive_label: selected label
    """
    # target_training_features = target_graph_features[target_graph_train_mask.nonzero()].squeeze()
    # target_training_labels = target_graph_labels[target_graph_train_mask.nonzero()].squeeze()

    features_hist = torch.sum(torch.where(features > 0, 1, 0), 0)
    proactive_features_index = torch.argmax(features_hist)
    max_features_num = torch.max(features_hist)

    proactive_label = torch.mode(torch.transpose(labels[features[:, proactive_features_index].nonzero()], 0, 1)).values

    return proactive_features_index, proactive_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # Add a small epsilon value to rowsum
    epsilon = 1e-10
    rowsum_with_epsilon = rowsum + epsilon
    r_inv = np.power(rowsum_with_epsilon, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.tensor(mx)
    return mx

def preprocess(adj, features, labels):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.
    """
    labels = torch.LongTensor(labels)
    if sp.issparse(features):
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(features)
    # adj = torch.FloatTensor(adj.todense())
        
def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))
