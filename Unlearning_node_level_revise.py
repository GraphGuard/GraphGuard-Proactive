import argparse
import numpy as np
import os
import json
import time
import torch
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import copy
import logging
from net.estimateadj import ProGNN
from net.gcn import GCN
from train.train_gnn import Evaluation_gnn, Train_gnn_model
from train.train_mia import Baseline_mia, MIA_evaluation,  Unlearning_MIA_evaluation, Pro_mia, Pro_MIA_evaluation
from train.train_proactive import Generate_proactive_features
from utils.graph_processing import Graph_partition, Identify_proactive_nodes, Select_proactive_node, load_data, \
    normalize, subgraph_generation
import torch.nn as nn
# Set the seed for PyTorch
torch.manual_seed(42)
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for DGL
dgl.random.seed(42)
# class GNN_sythetic(nn.Module):
#     def __init__(self, feature_number, hid_feats, out_feats):
#         super().__init__()
#         self.conv1 = dglnn.GraphConv(
#             in_feats=feature_number, out_feats=hid_feats)
#         self.conv2 = dglnn.GraphConv(
#             in_feats=hid_feats, out_feats=out_feats)
#
#     def forward(self, graph, inputs):
#         h = self.conv1(graph, inputs)
#         h = F.relu(h)
#         h = self.conv2(graph, h)
#         h = F.relu(h)
#         return h

class GNN_sythetic(nn.Module):
    def __init__(self, feature_number, hid_feats, out_feats):
        super().__init__()
        self.fc1 = nn.Linear(
            feature_number, hid_feats)
        self.fc2 = nn.Linear(
            hid_feats, out_feats)

    def forward(self, inputs):
        h = self.fc1(inputs)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        return h

# config logging
def config_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # Create a stream handler to display logs on the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    stream_handler.setFormatter(log_format)
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return file_handler, stream_handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,choices=['cora', 'pubmed', 'flickr', 'citeseer'], default='cora')
    parser.add_argument('--model', required=False, type=str, choices=['GCN', 'GAT', 'GIN', 'GraphSage'], default="GCN")    
    parser.add_argument('--epochs', required=False, default=300)
    parser.add_argument('--device', required=False, type=str, default="0")
    args = parser.parse_args()
    # Get current available GPU
    device = torch.device("cuda:" + args.device) if torch.cuda.is_available() else torch.device("cpu")

    # if torch.backends.mps.is_available():
    #     logging.info("Using MPS")
    #     device = torch.device("mps")

    # read config file
    config_file = 'config/' + args.dataset.lower() + '.json'
    with open(config_file) as f:
        config = json.load(f)
    # set up logging
    # create folder name by config params
    folder_name = 'log/' + config['params']['dataset'] + '_' + args.model + '_' + \
                  'Unlearning_with_defence' + '_' + config['params']['proattribute_method'] + '_' + config['params']['inject_only_proactive'] + '_' + \
                  config['params']['feature_inference_only'] + '_' + str(
        config['params']['proattribute_max_from']) + '_' + \
                  str(config['params']['proattribute_max_to'])
    # set up logging
    if not os.path.exists(folder_name):
        logging.info("Create folder: " + folder_name)
        os.makedirs(folder_name)
    timestamp_log_str = time.strftime("%Y%m%d_%H%M%S")
    current_log_folder = folder_name + '/' + timestamp_log_str
    os.mkdir(current_log_folder)
    log_file_name = current_log_folder + "/" + 'results.log'
    file_handler, stream_handler = config_logging(log_path=log_file_name)
    # parse config file
    params = config['params']
    # add device to params
    params['device'] = device
    dataset_name = params['dataset']
    num_classes = params['net_params']['num_labels']
    hidden_feats = params['net_params']['hidden']
    target_model_type = args.model
    proattribute_method = params['proattribute_method']
    inject_only_proactive = eval(params['inject_only_proactive'])
    feature_inference_only = eval(params['feature_inference_only'])
    # graph partition
    target_model_type = params['model']
    proattribute_method = params['proattribute_method']
    inject_only_proactive = eval(params['inject_only_proactive'])
    feature_inference_only = eval(params['feature_inference_only'])
    # load graph data
    g, features, labels, train_mask, test_mask, num_classes = load_data(dataset_name)
    # check params exist num_subsets
    if 'num_subsets' in params:
        logging.info("Start to split the graph into subsets")
        g, features, labels, train_mask, test_mask = subgraph_generation(g, features, labels, train_mask, test_mask,
                                                                         params)
    else:
        logging.info("Do not need to split the graph into subsets")
    # normalize features
    if dataset_name == 'citeseer':
        partition = True
    else:
        partition = False
    target_g, target_features, target_labels, target_train_mask, target_test_mask, \
        shadow_g, shadow_features, shadow_labels, shadow_train_mask, shadow_test_mask = \
        Graph_partition(g, features, labels, train_mask, test_mask,  recorded_partition = partition)
    # select a proper proactive feature and label index

    logging.info("Select the proactive feature and label index")
    proactive_features_index, proactive_label = Select_proactive_node(target_features, target_labels)

    # select the proactive nodes
    logging.info("Identify the target proactive nodes")
    proactive_node_index_target = Identify_proactive_nodes(target_features, target_labels, proactive_features_index,
                                                           proactive_label)
    logging.info('Total ' + str(len(target_labels)) + ' nodes in target set')
    logging.info('Generate ' + str(proactive_node_index_target.size()) + ' proactive node in target set')
    # print(target_features.size())
    logging.info("Identify the shadow proactive nodes")
    proactive_node_index_shadow = Identify_proactive_nodes(shadow_features, shadow_labels, proactive_features_index,
                                                           proactive_label)
    logging.info('Total ' + str(len(shadow_labels)) + ' nodes in shadow set')
    logging.info('Generate ' + str(proactive_node_index_shadow.size()) + ' proactive node in shadow set')
    # evaluate the MIA on original proactive nodes
    target_evaluation_mask = torch.zeros([target_features.size()[0], 1]).bool()
    shadow_evaluation_mask = torch.zeros([shadow_features.size()[0], 1]).bool()
    target_evaluation_mask[proactive_node_index_target, 0] = True
    shadow_evaluation_mask[proactive_node_index_shadow, 0] = True

    # Train: traget_model (target feature (without proactive node index)) vs new_target_model (proactive feature)
    # eval: target_model (proactive feature) -> 0-nonmember vs new_target_model (proactive feature) -> 1-member

    proactive_nodex_index_target_mask = torch.zeros(target_g.number_of_nodes(), dtype=torch.bool)
    proactive_nodex_index_target_mask[proactive_node_index_target] = True
    proactive_nodex_index_shadow_mask = torch.zeros(shadow_g.number_of_nodes(), dtype=torch.bool)
    proactive_nodex_index_shadow_mask[proactive_node_index_shadow] = True
    # train target GNN model
    logging.info("Start train the target GNN model")
    target_model = Train_gnn_model(params, target_g,
                                   target_features,
                                   target_labels,
                                   target_train_mask,
                                   target_test_mask)

    target_eval_acc, target_logits, target_prob_list = Evaluation_gnn(target_model, target_g, target_features,
                                                                      target_labels)
    logging.info(f"The evaluation accuracy of the target model is:{target_eval_acc}")
    # inti an eyes graph based on target graph
    eyes_g = dgl.DGLGraph()
    eyes_g.add_nodes(target_g.number_of_nodes())
    eyes_g.add_edges(torch.tensor(range(target_g.number_of_nodes())), torch.tensor(range(target_g.number_of_nodes())))
    target_eval_acc, targe_feat_eye_g_logits, targe_feat_eye_g_prob_list = Evaluation_gnn(target_model, eyes_g,
                                                                                          target_features,
                                                                                          target_labels)

    # train shadow GNN model
    logging.info("Start train the shadow GNN model")
    shadow_model = Train_gnn_model(params, shadow_g,
                                   shadow_features,
                                   shadow_labels,
                                   shadow_train_mask,
                                   shadow_test_mask)
    shadow_eval_acc, shadow_logits, _ = Evaluation_gnn(shadow_model, shadow_g, shadow_features, shadow_labels)
    logging.info(f"The evaluation accuracy of the shadow model is:{shadow_eval_acc}")
    # create masked graph, features, labels
    target_g_masked = target_g.subgraph(proactive_nodex_index_target_mask)
    target_features_masked = target_features[proactive_nodex_index_target_mask]
    target_labels_masked = target_labels[proactive_nodex_index_target_mask]
    logging.info("Start train the target GNN model without proactive node index")
    target_model_no_proa_nodeidx = Train_gnn_model(params, target_g_masked,
                                                   target_features_masked,
                                                   target_labels_masked,
                                                   target_train_mask,
                                                   target_test_mask)
    # evalue the target model without proactive node index
    # TODO replace target_g_masked with eyes_g
    target_acc_no_proa_nodeidx, target_feat_g_non_proa_logits, target_feat_g_non_proa_prob_list = Evaluation_gnn(
        target_model_no_proa_nodeidx,
        target_g,
        target_features,
        target_labels)
    logging.info(
        f"The evaluation accuracy of the target model without proactive node index is:{target_acc_no_proa_nodeidx}")
    # TODO: save the target_logits_non_pro , change target_g to eyes_g
    _, target_feat_eye_g_non_pro_logits, target_feat_eye_g_non_pro_prob_list = Evaluation_gnn(
        target_model_no_proa_nodeidx,
        eyes_g, target_features,
        target_labels)

    logging.info("Generate the proactive features")
    proactive_target_features, proactive_attribute_trigger_index = Generate_proactive_features(proattribute_method,
                                                                                               target_model,
                                                                                               proactive_node_index_target,
                                                                                               target_g,
                                                                                               target_features,
                                                                                               target_labels)
    # train GNN model and attack (with proactive)
    logging.info("Start train the New Target GNN model with proactive feature:")
    if dataset_name == 'citeseer':
        params['epochs']=3000
    new_target_model = Train_gnn_model(params, target_g,
                                       proactive_target_features, target_labels,
                                       target_train_mask, target_test_mask)
    # TODO: change the target_g to eyes_g
    new_target_eval_acc, new_target_pro_feat_eye_g_logits, new_target_pro_feat_eye_g_prob_list = Evaluation_gnn(
        new_target_model,
        eyes_g,
        proactive_target_features,
        target_labels)
    logging.info(f"The evaluation accuracy of the new target model is:{new_target_eval_acc}")
    _, target_pro_feat_eye_g_non_pro_logits, target_pro_feat_eye_g_non_pro_prob_list = Evaluation_gnn(
        target_model_no_proa_nodeidx,
        eyes_g, proactive_target_features,
        target_labels)


    # Now save all the logits and labels
    # Save true label probability
    np.save(f'{current_log_folder}/target_prob_list.npy', target_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_prob_list.npy', targe_feat_eye_g_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_g_non_proa_prob_list.npy',
            target_feat_g_non_proa_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_non_pro_prob_list.npy',
            target_feat_eye_g_non_pro_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/new_target_pro_feat_eye_g_prob_list.npy',
            new_target_pro_feat_eye_g_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_pro_feat_eye_g_non_pro_prob_list.npy',
            target_pro_feat_eye_g_non_pro_prob_list.detach().numpy())
    # save target, target_no_proa logit and new target logit
    np.save(f'{current_log_folder}/target_logits.npy', target_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_logits.npy', targe_feat_eye_g_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_g_non_proa_logits.npy', target_feat_g_non_proa_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_non_pro_logits.npy',
            target_feat_eye_g_non_pro_logits.detach().numpy())
    np.save(f'{current_log_folder}/new_target_pro_feat_eye_g_logits.npy',
            new_target_pro_feat_eye_g_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_pro_feat_eye_g_non_pro_logits.npy',
            target_pro_feat_eye_g_non_pro_logits.detach().numpy())

    # new evaluation the proactive MIA
    ## use original graph
    try:
        logits_mem = target_model(target_g.adjacency_matrix(), proactive_target_features)
        logits_pro_mem = new_target_model(target_g.adjacency_matrix(), proactive_target_features)
    except:
        logits_mem = target_model(target_g, proactive_target_features)
        logits_pro_mem = new_target_model(target_g, proactive_target_features)
    logging.info("=============Final Results=============")
    logging.info("============Trigger Injection (second value high means injected)=============")
    _, indices = torch.max(logits_mem[proactive_node_index_target], dim=1)
    labels = copy.deepcopy(indices)
    labels = torch.where((labels != -1), proactive_label, labels)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits member:{correct.item() * 1.0 / len(labels)}")
    _, indices = torch.max(logits_pro_mem[proactive_node_index_target], dim=1)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits pro member:{correct.item() * 1.0 / len(labels)}")
    ## use shadow graph
    # shadow graph features
    shadow_graph_proactive_features = copy.deepcopy(shadow_features)
    if inject_only_proactive:
        for i in proactive_node_index_shadow.numpy():
            for j in proactive_attribute_trigger_index.numpy():
                with torch.no_grad():
                    shadow_graph_proactive_features[i][j] = 1
    else:
        for i in range(len(shadow_graph_proactive_features)):
            for j in proactive_attribute_trigger_index.numpy():
                with torch.no_grad():
                    shadow_graph_proactive_features[i][j] = 1
    # normalization
    shadow_graph_proactive_features[shadow_graph_proactive_features != 0] = 1
    target_graph_proactive_features = normalize(shadow_graph_proactive_features)
    if feature_inference_only:
        src_idx = torch.tensor(range(len(shadow_graph_proactive_features)), dtype=torch.int64)
        shadow_g = dgl.DGLGraph()
        shadow_g.add_nodes(len(shadow_graph_proactive_features))
        shadow_g.add_edges(src_idx, src_idx)
    try:
        logits_non_mem = target_model(shadow_g.adjacency_matrix(), shadow_graph_proactive_features)
        logits_pro_non_mem = new_target_model(shadow_g.adjacency_matrix(), shadow_graph_proactive_features)
    except:
        logits_non_mem = target_model(shadow_g, shadow_graph_proactive_features)
        logits_pro_non_mem = new_target_model(shadow_g, shadow_graph_proactive_features)
    if inject_only_proactive:
        _, indices = torch.max(logits_non_mem[proactive_node_index_shadow], dim=1)
    else:
        _, indices = torch.max(logits_non_mem, dim=1)

    labels = copy.deepcopy(indices)
    labels = torch.where((labels != -1), proactive_label, labels)
    # print(labels)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits non-member:{correct.item() * 1.0 / len(labels)}")
    if inject_only_proactive:
        _, indices = torch.max(logits_pro_non_mem[proactive_node_index_shadow], dim=1)
    else:
        _, indices = torch.max(logits_pro_non_mem, dim=1)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits pro non-member:{correct.item() * 1.0 / len(labels)}")
    # target_mode vs new_target_model
    # target_model(small value) vs new_target_model(large value) -> threshold

    # # evaluate the proactive MIA
    # Evaluation_proactive_mia(target_model, new_target_model, shadow_model,
    #                          target_g, target_features,
    #                          # shadow_g, shadow_features,
    #                          target_g, proactive_target_features,
    #                          proactive_node_index_target, # proactive_node_index_shadow,
    #                          target_labels, proactive_label)

    attack_model = Baseline_mia(params, target_g, shadow_g, target_model, target_features, shadow_features)
    attack_acc = MIA_evaluation(attack_model, target_model,
                                target_g, target_features, shadow_g, shadow_features, target_evaluation_mask,
                                shadow_evaluation_mask)
    logging.info(f"Baseline MIA Acc:{attack_acc}")

    '''
        Unlearning Start
    '''


    # unlearning samples generation

    ## define GAE to learn structure.

    ### G=(A,X)-GNN_learn-> Z -> A'

    ### G'=(A',X) -GNN_target-> -GNN_attack-> 1


    # results_labels = torch.ones(shadow_g.number_of_nodes(), dtype=torch.long).squeeze(1)
    # breakpoint()
    results_labels = torch.ones(target_g.number_of_nodes(), dtype=torch.long)
    # generation_model = GNN_sythetic(features.shape[1], 32, 16)
    # generation_model = GNN_sythetic(features.shape[1], 256, (features.shape[0]*(features.shape[0]+1)//2))
    generation_model = GNN_sythetic(features.shape[1], 256, 256)
    opt_generation = torch.optim.Adam(generation_model.parameters())

    new_target_model.eval()

    label_0_index_target_graph_mask = proactive_nodex_index_target_mask  # .squeeze(1)
    label_others_index_target_graph_mask = torch.logical_not(label_0_index_target_graph_mask)

    # new_attack_model = Baseline_mia(params, target_g, shadow_g, new_target_model, proactive_target_features, shadow_features)
    new_attack_model = Pro_mia(params, target_g, new_target_model, proactive_target_features, label_0_index_target_graph_mask, label_others_index_target_graph_mask
                                    )
    new_attack_model.train()
    print("start training generation model")
    for epoch in range(200):
        generation_model.train()
        # forward propagation by using all nodes
        # print(target_graph_features.size())
        # logits = generation_model(target_g, proactive_target_features)
        logits = generation_model(proactive_target_features)
        # compute loss
        # print(logits.size())
        # print(target_graph_train_mask)
        # print(target_graph_labels.size())
        # logits_numpy = logits.copy().cpu().detach().numpy()
        new_adj = torch.sigmoid(torch.matmul(logits, logits.t()))
        src = torch.nonzero(new_adj)[:, 0]
        dst = torch.nonzero(new_adj)[:, 1]
        g_new = dgl.graph((src, dst))
        g_new.ndata['feat'] = proactive_target_features
        output = new_target_model(g_new.adjacency_matrix(), proactive_target_features)
        results = new_attack_model(output)
        # print(results.size())
        # print(results_labels.size())
        # print(results_labels)
        # print(results)
        generation_loss = F.cross_entropy(results, results_labels)
        # loss = F.cross_entropy(logits[target_graph_train_mask], target_graph_labels[target_graph_train_mask])
        # # compute validation accuracy
        # acc = evaluate(model, dgl_target_graph, target_graph_features, target_graph_labels, target_graph_test_mask)
        # backward propagation
        # if (epoch + 1) % 100 == 0:
        #     print(f"Epoch:{epoch+1}, the loss is:{generation_loss}")
        opt_generation.zero_grad()
        generation_loss.backward()
        opt_generation.step()
        # print("Epoch:")
        # print(epoch)

    # label_0_in_target_graph = proactive_node_index_target.long()
    # label_others_in_target_graph = th.Tensor(list(target_set_index))[(target_graph_labels != 0)].long()


    ## graph partition based on attributes
    ### select nodes whoes label is 0 and the attribute in index ? is 1.
    # label_0_in_shadow_graph = th.Tensor(list(shadow_set_index))[(shadow_graph_labels == 0)].long()
    # label_others_in_shadow_graph = th.Tensor(list(shadow_set_index))[(shadow_graph_labels != 0)].long()
    label_0_index_shadow_graph_mask = proactive_nodex_index_shadow_mask#.squeeze(1)
    label_others_index_shadow_graph_mask = torch.logical_not(label_0_index_shadow_graph_mask)
    # print(label_0_in_shadow_graph.size())
    # print(labels[label_0_in_shadow_graph.long().tolist()])
    # print()

    # shadow_node_index = label_0_in_shadow_graph
    # nonshadow_node_index = label_others_in_shadow_graph

    ## add perturbations based on gradients

    # unlearn the target GNN model

    # define unlearning label
    nonmember_labels = torch.zeros([target_g.number_of_nodes()]).long()
    member_labels = torch.ones([target_g.number_of_nodes()]).long()

    print("start unlearning target model")

    unlearning_opt = torch.optim.Adam(new_target_model.parameters())
    new_attack_model.train()

    random_target_label = torch.randint(0,num_classes-1,[target_g.number_of_nodes()])
    if dataset_name == 'citeseer':
        this_epoches = 4
    else:
        this_epoches = 2

    for epoch in range(this_epoches):
        new_target_model.train()
        # forward propagation by using all nodes
        # print(shadow_graph_features.size())
        if isinstance(g_new, dgl.DGLGraph):
            logits = new_target_model(g_new.adjacency_matrix(), proactive_target_features)
        elif isinstance(proactive_target_features, torch.Tensor):
            logits = new_target_model(g_new, proactive_target_features)
        # logits = new_target_model(g_new, shadow_features)
        attack_logits = new_attack_model(logits)
        # compute loss
        # print(logits.size())
        # print(shadow_graph_train_mask)
        # print(shadow_graph_labels.size())
        # accuracy loss
        acc_loss_forget = F.cross_entropy(logits[label_0_index_target_graph_mask],
                                          random_target_label[label_0_index_target_graph_mask])
        acc_loss_remain = F.cross_entropy(logits[label_others_index_target_graph_mask],
                                          target_labels[label_others_index_target_graph_mask])
        # unlearning loss
        unlearn_loss_forget = F.cross_entropy(attack_logits[label_0_index_target_graph_mask],
                                              nonmember_labels[label_0_index_target_graph_mask])
        # unlearn_loss_remain = F.cross_entropy(attack_logits[label_others_index_shadow_graph_mask], member_labels[label_others_index_shadow_graph_mask])
        loss = 1 * acc_loss_remain + 1 * acc_loss_forget + 0 * unlearn_loss_forget
        # + unlearn_loss_remain
        # loss = F.cross_entropy(logits[shadow_graph_train_mask], shadow_graph_labels[shadow_graph_train_mask])
        # # compute validation accuracy
        # acc = evaluate(model, dgl_shadow_graph, shadow_graph_features, shadow_graph_labels, shadow_graph_test_mask)
        # backward propagation
        # print(unlearn_loss_forget)
        # print(acc_loss_forget)
        # print(acc_loss_remain)


        unlearning_opt.zero_grad()
        loss.backward()
        unlearning_opt.step()

    unlearned_target_eval_acc, unlearned_target_pro_feat_logits, unlearned_target_pro_feat_prob_list = Evaluation_gnn(new_target_model, target_g, proactive_target_features, target_labels, target_train_mask)
    print("unlearned target model accuracy is")
    print(unlearned_target_eval_acc)

    # evaluating MIA

    # define unlearning label
    target_nonmember_labels = torch.zeros([target_g.number_of_nodes()]).long()
    target_member_labels = torch.ones([target_g.number_of_nodes()]).long()

    new_target_model.eval()
    attack_model.eval()
    # acc,_,_ = Evaluation_gnn(new_target_model, target_g, target_features, target_labels) #model(dgl_target_graph, target_graph_features)
    # mia_attack_acc = Unlearning_MIA_evaluation(attack_model, new_target_model, target_g, target_features, label_0_index_target_graph_mask)

    # attack_acc = MIA_evaluation(new_attack_model, new_target_model,
    #                             target_g, proactive_target_features, shadow_g, shadow_features, torch.logical_and(label_0_index_target_graph_mask,target_test_mask),
    #                             torch.zeros(shadow_g.number_of_nodes(), dtype=torch.bool))

    # attack_acc = Pro_MIA_evaluation(new_attack_model, new_target_model,
    #                             target_g, proactive_target_features, target_g, proactive_target_features,
    #                             label_0_index_target_graph_mask, label_others_index_target_graph_mask)
    # balanced_mia_attack_acc = 0.5*mia_attack_acc + 0.5*attack_acc
    # new_attack_model = Pro_mia(params, target_g, new_target_model, proactive_target_features,
    #                            label_0_index_target_graph_mask, label_others_index_target_graph_mask
    #                            )
    attack_acc = MIA_evaluation(new_attack_model, new_target_model,
                                target_g, proactive_target_features, shadow_g, shadow_features, target_evaluation_mask,
                                shadow_evaluation_mask)

    print("======Reproduce Results  E3&4==========")
    print("Unlearned model accuracy is")
    print(unlearned_target_eval_acc)
    print("MIA attack successful rates for unlearned samples is")
    print(attack_acc)

    # close the log file handler
    logging.info("============End=============")
    stream_handler.close()
    file_handler.close()

