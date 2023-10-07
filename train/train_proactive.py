import numpy as np
import torch
import scipy    
import warnings
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import logging
import networkx
import matplotlib.pyplot as pyplot
from dgl import DGLGraph
from torch.autograd import grad
from train.train_gnn import Evaluation_gnn
from utils.graph_processing import normalize



def Generate_proactive(model, proactive_node_index, g, features, labels, proactive_target_label):
    #dataset_name, target_model_without_radioactive, target_proactive_indexes, dgl_target_graph, target_graph_features, target_graph_labels):
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # g_scipy = g.adj()
    # dgl_target_graph = Variable(g_scipy.to_dense(), requires_grad=True)
    dgl_target_graph = Variable(g.adjacency_matrix().to_dense(), requires_grad=True)
    # print(dgl_target_graph)
    target_graph_features = Variable(features, requires_grad=True)
    #use the target model
    # gcn_Net = GCN(feature_number, label_number)
    # gcn_Net.load_state_dict(th.load(SAVE_PATH))
    # optimizer = th.optim.Adam(gcn_Net.parameters(), lr=1e-2, weight_decay=5e-4)
    # dur = []
    model.eval()

    acc_org = Evaluation_gnn(model, dgl_target_graph, target_graph_features, labels, print_results=False)
    logging.info("Original accuracy is " + str(acc_org))
    g_new = copy.deepcopy(dgl_target_graph)
    features_new = copy.deepcopy(target_graph_features)

    sensitive_nodes_index = proactive_node_index.detach().clone()

    # generate the detector mask
    # train_mask_org = train_mask.detach().clone()
    # train_mask_sensitive = test_mask.detach().clone() * 0

    # verify_node_index_list = [55,94,69,32,48]
    # for index in sensitive_nodes_index:
    #     train_mask_sensitive[index] = 1

    # detector_mask = th.ByteTensor(train_mask_sensitive)

    # acc_sensitive_org = evaluate_detector(gcn_Net, g, features, labels, detector_mask)

    model.train()
    logits = model(dgl_target_graph, target_graph_features)
    logp = F.log_softmax(logits, 1)


    ## select a labels
    proactive_target_labels = copy.deepcopy(labels) * 0 + int(proactive_target_label)
    gradience_matrix = F.nll_loss(logp, proactive_target_labels, reduce=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    perturb_num = 0
    for i in sensitive_nodes_index:
        perturb_num += 1
        logging.info("perturbing node " + str(i) + "... It is the " + str(perturb_num) + "th node. ")
        optimizer.zero_grad()
        gradience_matrix[i].backward(retain_graph=True)

        gradience_loss_g = grad(gradience_matrix[i], dgl_target_graph, create_graph=True)
        # print(gradience_loss_g)
        added_edges_indexs = torch.argsort(gradience_loss_g[0][i])

        gradience_loss_features = grad(gradience_matrix[i], target_graph_features, create_graph = True)
        added_features_indexs = torch.argsort(gradience_loss_features[0][i])

        last_k = -1
        end_loop_flag = 0
        modified_operation = 0
        for j in range(len(added_edges_indexs)):
            if end_loop_flag == 1:
                break
            for k in range(len(added_features_indexs)):
                if gradience_loss_g[0][i][added_edges_indexs[j]]>=0 and gradience_loss_features[0][i][added_features_indexs[k]]>=0:
                    end_loop_flag = 1
                    break
                if modified_operation > 10:
                    end_loop_flag = 1
                    break
                if k <= last_k:
                    continue
                if gradience_loss_g[0][i][added_edges_indexs[j]]<gradience_loss_features[0][i][added_features_indexs[k]]:
                    # loss for edge is smaller than features, so perturb edges
                    g_mody_backup = copy.deepcopy(g_new)
                    if device == 'cpu':
                        g_numpy = g_new.detach().numpy()
                    else:
                        g_numpy = g_new.detach().cpu().numpy()
                    g_numpy[i, j]=1
                    np.where(g_numpy > 0, g_numpy, 1)

                    g_new = networkx.from_numpy_matrix(g_numpy)
                    g_new = DGLGraph(g_new)
                    # normalization
                    degs = g_new.in_degrees().float()
                    norm = torch.pow(degs, -0.5)
                    norm[torch.isinf(norm)] = 0
                    # if device != 'cpu':
                    #     norm = norm.cuda()
                    g_new.ndata['norm'] = norm.unsqueeze(1)
                    g_new = g_new.adjacency_matrix()
                    g_new = Variable(g_new.to_dense(), requires_grad=True)

                    acc_new = Evaluation_gnn(model, g_new, features_new, labels, print_results=False)
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

                    features_new = Variable(torch.FloatTensor(features_numpy), requires_grad=True)

                    acc_new = Evaluation_gnn(model, g_new, features_new, labels, print_results=False)
                    # acc_sensitive_new = evaluate_detector(gcn_Net, g_new, features_new, labels, detector_mask)

                    if (acc_new != acc_org):
                        features_new = features_mody_backup
                        # print('deduct accuracy from: ' + str(acc_org) + ' to ' + str(acc_new) + ' !')
                    else:
                        modified_operation = modified_operation + 1
                        # print('add features at ' + str(k) + ' !')

    logging.info("perturbation done!")
    # if device == 'cpu':
    #     features_numpy = features_new.detach().numpy()
    #     g_numpy = g_new.detach().numpy()
    # else:
    #     features_numpy = features_new.detach().cpu().numpy()
    #     g_numpy = g_new.detach().cpu().numpy()

    return features_new, g_new, sensitive_nodes_index

def Generate_proactive_features(proattribute_method,model, proactive_node_index_target, target_g, target_features, target_labels):
    """
        This code is for generating the verification nodes for a target model
        Input:
          name of the proactive attribute generation method
          target model training without radioactive data
          target proactive sample indexes (those selected to be protected)
          target graph structure
          target graph features
          target graph labels

        output:
          modified features
          selected attribute trigger indexes
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # g_scipy = g.adj()
    # dgl_target_graph = Variable(g_scipy.to_dense(), requires_grad=True)
    dgl_target_graph = Variable(target_g.adjacency_matrix().to_dense(), requires_grad=True)
    # print(dgl_target_graph)
    target_graph_features = Variable(target_features, requires_grad=True)


    #use the target model
    # gcn_Net = GCN(feature_number, label_number)
    # gcn_Net.load_state_dict(th.load(SAVE_PATH))

    # optimizer = th.optim.Adam(gcn_Net.parameters(), lr=1e-2, weight_decay=5e-4)
    # dur = []

    model.eval()

    acc_org,_,_ = Evaluation_gnn(model, dgl_target_graph, target_graph_features, target_labels, print_results=False)
    logging.info("Original accuracy is " + str(acc_org))
    # g_new = copy.deepcopy(dgl_target_graph)
    # features_new = copy.deepcopy(target_graph_features)
    #
    # sensitive_nodes_index = proactive_node_index_target.detach().clone()

    # generate the detector mask
    # train_mask_org = train_mask.detach().clone()
    # train_mask_sensitive = test_mask.detach().clone() * 0

    # verify_node_index_list = [55,94,69,32,48]
    # for index in sensitive_nodes_index:
    #     train_mask_sensitive[index] = 1

    # detector_mask = th.ByteTensor(train_mask_sensitive)

    # acc_sensitive_org = evaluate_detector(gcn_Net, g, features, labels, detector_mask)

    model.train()
    logits = model(dgl_target_graph, target_graph_features)
    logp = F.log_softmax(logits, 1)


    ## select a labels
    # proactive_target_labels = copy.deepcopy(labels) * 0 + int(proactive_target_label)
    logging.info(f"Target graph features size: {target_graph_features.size()}")
    logging.info(f"Target labels size: {target_labels.size()}")
    logging.info(f"Logp size: {logp.size()}")
    gradience_matrix = F.nll_loss(logp, target_labels, reduce=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    if proattribute_method == 'max' or proattribute_method == 'min':
        optimizer.zero_grad()
        loss = torch.sum(gradience_matrix[proactive_node_index_target])
        loss.backward(retain_graph=True)
        gradience_loss_features = grad(loss, target_graph_features, create_graph=True)
        gradience_loss_features_sum = torch.sum(gradience_loss_features[0][proactive_node_index_target],0)
        logging.info(f"Gredient loss features sum size: {gradience_loss_features_sum.size()}")
        added_features_indexs = torch.argsort(gradience_loss_features_sum) #descending=False
        # print(added_features_indexs[0:10].size())
        if proattribute_method == 'min':
            proattribute_index = added_features_indexs[0:5]
        if proattribute_method == 'max':
            proattribute_index = added_features_indexs[-6:-1]

        # print(target_graph_proactive_features.size())
        # print(proattribute_index)
        # print(proactive_node_index_target)
        model.eval()
        target_graph_proactive_features = target_graph_features.detach().clone()
        for i in proactive_node_index_target.tolist():
            for k in proattribute_index.tolist():
                with torch.no_grad():
                    target_graph_proactive_features[i][k]=1
        # normalization
        # ============== Removed by Xiangwen ================= #
        '''
        1. Seems the normalization is not necessary, because the features are already normalized.
        2. This normalization will cause the accuracy to be 0.
        '''
        # sum_of_rows = target_graph_proactive_features.sum(axis=1)
        # set all non zero values in target_graph_proactive_features to 1
        target_graph_proactive_features[target_graph_proactive_features!=0] = 1
        # target_graph_proactive_features = target_graph_proactive_features / sum_of_rows[:, np.newaxis]
        target_graph_proactive_features = normalize(target_graph_proactive_features)
        # ============== Removed by Xiangwen ================= #
        logging.info(f"Type of target_graph_proactive_features: {type(target_graph_proactive_features)}")
        # features_new = Variable(torch.FloatTensor(features_numpy), requires_grad=True)
        # print(added_features_indexs[-11:-1].size())
        # print(added_features_indexs[0])
        # print(added_features_indexs[1])
        # print(added_features_indexs.size())
        # print(gradience_loss_features[0][sensitive_nodes_index])
        # print(gradience_loss_features[0][sensitive_nodes_index].size())
        # num = 0
        #
        # for i in sensitive_nodes_index:
        #     num = num + 1
        #     print("perturbing node " + str(i) + "... It is the " + str(num) + "th node. ")
        #     optimizer.zero_grad()
        #     gradience_matrix[i].backward(retain_graph=True)
        #     gradience_loss_features = grad(gradience_matrix[i], dgl_target_graph, create_graph=True)
        #
        #     added_features_indexs = torch.argsort(gradience_loss_features[0][i])

    return target_graph_proactive_features, proattribute_index

def Evaluation_proactive_mia(target_model, new_target_model, shadow_model,
                             target_g, target_features,
                             # shadow_g, shadow_features,
                             proactive_target_g, proactive_target_features,
                             proactive_node_index_target, # proactive_node_index_shadow,
                             proactive_target_label, proactive_label):
    """
    This function is for evaluate the proactive MIA by compare the distribution between:
    # model(target nodes)  vs model(shadow nodes) [is similar as "model(target nodes)  vs shadow_model(target nodes)"]
    # new_model(perturb target nodes) vs model(perturb target nodes)
    model(target nodes)  vs shadow_model(target nodes)
    new_model(perturb target nodes) vs shadow_model(perturb target nodes)

    Inputs:
    target_model:
    new_target_model:

    target_g:
    target_features:
    shadow_g:
    shadow_features:
    proactive_target_g:
    proactive_target_features:

    proactive_node_index_target:
    proactive_node_index_shadow:

    proactive_target_label: the label selected to be perturb
    proactive_label: the label of the proactive nodes

    Outputs:
    four distribution as shown as the description above.
    """

    # create proactive_index_mask
    proactive_node_index_mask = torch.zeros([target_features.size()[0], 1]).bool()
    proactive_node_index_mask[proactive_node_index_target, 0] = True
    # print(proactive_node_index_mask.size())
    proactive_node_index_mask = proactive_node_index_mask.squeeze()
    # print(proactive_node_index_mask.size())

    # # create shadow_index_mask
    # shadow_node_index_mask = torch.zeros([shadow_features.size()[0], 1]).bool()
    # shadow_node_index_mask[proactive_node_index_shadow] = True
    # shadow_node_index_mask = shadow_node_index_mask.squeeze()

    # calculating the output distribution
    # print(proactive_label)
    dis_tm_t = target_model(target_g.adjacency_matrix(),target_features)
    # print(dis_tm_t)
    dis_tm_t = dis_tm_t[:,proactive_label]
    # print(dis_tm_t)
    # dis_tm_s = target_model(shadow_g.adjacency_matrix(),shadow_features)
    # dis_tm_s = dis_tm_s[:,proactive_label]

    dis_sm_t = shadow_model(target_g.adjacency_matrix(),target_features)
    dis_sm_t = dis_sm_t[:, proactive_label]

    dis_nm_pt = new_target_model(proactive_target_g,proactive_target_features)
    dis_nm_pt = dis_nm_pt[:,proactive_label]

    dis_sm_pt = shadow_model(proactive_target_g,proactive_target_features)
    dis_sm_pt = dis_sm_pt[:,proactive_label]

    # discuss whether they are the same distribution via Kolmogorov–Smirnov test
    # print(dis_tm_t)
    # print(dis_tm_t.size())
    # print(proactive_node_index_mask.size())
    # print(dis_tm_t[proactive_node_index_mask])
    # print(dis_tm_t[proactive_node_index_mask].size())
    # calculating the output distribution corresponding to proactive labels
    dis_tm_t = np.squeeze(dis_tm_t[proactive_node_index_mask].detach().numpy())
    dis_sm_t = np.squeeze(dis_sm_t[proactive_node_index_mask].detach().numpy())
    dis_nm_pt = np.squeeze(dis_nm_pt[proactive_node_index_mask].detach().numpy())
    dis_sm_pt = np.squeeze(dis_sm_pt[proactive_node_index_mask].detach().numpy())

    # print(dis_tm_t)
    # print(type(dis_tm_t))
    # print(dis_tm_s.shape)
    original_mia = scipy.stats.ks_2samp(dis_tm_t, dis_sm_t)
    logging.info(f"Original mia p value is:{original_mia}")
    pyplot.style.use('seaborn-deep')
    bins = np.linspace(0.5, 1, 200)
    # d1a1,d1a2,d1a3 = scipy.stats.foldnorm.fit(dis_tm_t)
    # d2a1,d2a2,d2a3 = scipy.stats.foldnorm.fit(dis_sm_t)
    # y1 = scipy.stats.foldnorm.pdf(bins,d1a1,d1a2,d1a3)
    # y2 = scipy.stats.foldnorm.pdf(bins, d2a1,d2a2,d2a3)
    pyplot.hist(dis_tm_t, bins, alpha=0.5, label='member')
    pyplot.hist(dis_sm_t, bins, alpha=0.5, label='non-member')
    # pyplot.plot(bins, y1, 'b--', linewidth=2)
    # pyplot.plot(bins, y2, 'g--', linewidth=2)
    pyplot.legend(loc='upper right')
    pyplot.show()

    proactive_mia = scipy.stats.ks_2samp(dis_nm_pt, dis_sm_pt)
    logging.info(f"proactive mia p value is:{proactive_mia}")
    bins = np.linspace(0, 0.5, 200)
    # d3a1, d3a2, d3a3 = scipy.stats.foldnorm.fit(dis_nm_pt)
    # d4a1, d4a2, d4a3 = scipy.stats.foldnorm.fit(dis_sm_pt)
    # y3 = scipy.stats.foldnorm.pdf(bins, d3a1, d3a2, d3a3)
    # y4 = scipy.stats.foldnorm.pdf(bins, d4a1, d4a2, d4a3)
    pyplot.hist(dis_nm_pt, bins, alpha=0.5, label='member')
    pyplot.hist(dis_sm_pt, bins, alpha=0.5, label='non-member')
    # pyplot.plot(bins, y3, 'b--', linewidth=2)
    # pyplot.plot(bins, y4, 'g--', linewidth=2)
    pyplot.legend(loc='upper right')
    pyplot.show()
    # discuss whether they are the same distribution


    return dis_tm_t, dis_sm_t, dis_nm_pt, dis_sm_pt
